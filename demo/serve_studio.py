"""Local mesh generation and Gaussian PLY asset studio; no external upload."""
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json,uuid,threading,subprocess,sys,io,shutil,time,base64
import numpy as np
from PIL import Image,ImageOps
from plyfile import PlyData
from wardrobe_backend import handle_get, handle_post
import avatar_backend
ROOT=Path(__file__).resolve().parent
STORE=ROOT/'studio_results';STORE.mkdir(exist_ok=True)
lock=threading.Lock();active=None
def save(out,m):
    temp=out/'record.tmp';temp.write_text(json.dumps(m,ensure_ascii=False),encoding='utf8');temp.replace(out/'record.json')
def describe(path):
    ply=PlyData.read(path);v=ply['vertex'].data
    required=['x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    if not all(k in v.dtype.names for k in required):raise ValueError('需要包含颜色、透明度、尺度和旋转的 Gaussian PLY；普通点云或网格请使用网格入口。')
    if not 0<len(v)<=3_000_000:raise ValueError('高斯数量须在 1 至 300 万之间')
    for k in required:
        if not np.isfinite(v[k]).all():raise ValueError('PLY 包含无效数值')
    xyz=np.column_stack([v[k] for k in ['x','y','z']]);lo,hi=np.quantile(xyz,[.01,.99],axis=0)
    return {'gaussians':len(v),'center':((lo+hi)/2).tolist(),'radius':float(max(np.linalg.norm(hi-lo)/2,.01)),'sh_degree':0 if 'f_rest_0' not in v.dtype.names else 1}
def run(out,m):
    global active
    try:
        with (out/'run.log').open('w',encoding='utf8') as f:
            if m['kind']=='gaussian':
                refs=out/'references';refs.mkdir(exist_ok=True)
                uploads=sorted((out/'uploads').glob('*.png')) if (out/'uploads').exists() else [out/'upload.png']
                for i,upload in enumerate(uploads):
                    p=subprocess.run([sys.executable,'-u',str(ROOT/'prepare_lhm_photo.py'),str(upload),str(refs/f'{i:02d}.png')],stdout=f,stderr=subprocess.STDOUT)
                    if p.returncode:raise RuntimeError('照片预处理失败，请查看日志')
                shutil.copy2(refs/'00.png',out/'reference.png')
                def linux_path(path):
                    path=Path(path).resolve().as_posix();return '/mnt/'+path[0].lower()+path[2:]
                command=['wsl','-d','Ubuntu-22.04','--','/opt/lhm-venv/bin/python','-u',linux_path(ROOT/'run_lhm_verify.py'),linux_path(refs),linux_path(out)]
                if m.get('motion'):command.append(m['motion'])
                p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT.parent)
            else:
                p=subprocess.run([sys.executable,'-u',str(ROOT/'run_mesh.py'),str(out/'upload.png'),str(out)],stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError('模型运行失败，请查看日志')
        if m['kind']=='gaussian':
            verification=json.loads((out/'verification.json').read_text())
            if not verification.get('finite'):raise RuntimeError('生成文件数值无效')
            m.update(status='complete',metrics={**describe(out/'gaussian.ply'),**verification},files=['gaussian.ply','verification.json','reference.png','input.png','run.log'])
            if m.get('motion'):
                animation=json.loads((out/'animation.json').read_text(encoding='utf8'))
                if not animation.get('finite') or animation.get('frames',0)<2:raise RuntimeError('动画验证失败')
                m['animation']=animation
                m['files'] += ['animation.mp4','animation.json','animation_poster.png','animation_middle.png','animation_last.png']
            m['source']=f"本机 LHM++ {verification.get('input_count',1)} 图联合推理 / 标准姿态；未观测区域由 AI 推测"
            m['files'] += [str(p.relative_to(out)).replace('\\','/') for p in sorted((out/'references').glob('*.png'))]
        else:
            m.update(status='complete',metrics=json.loads((out/'mesh.json').read_text()),files=['human.glb','human.ply','mesh.json','input.png','foreground.png','run.log'])
    except Exception as e:m.update(status='failed',error=str(e))
    finally:
        save(out,m)
        with lock:active=None
def seed():
    p=ROOT.parent/'example_multi_images(1).ply';out=STORE/'team_sample'
    if p.exists() and not (out/'record.json').exists():
        out.mkdir(exist_ok=True);shutil.copy2(p,out/'gaussian.ply')
        save(out,{'id':out.name,'name':'组员提供的高斯人体','kind':'gaussian','status':'complete','created':time.time(),'source':'导入现有 PLY，生成方法待确认（LHM++ 文件命名线索）','metrics':describe(out/'gaussian.ply'),'files':['gaussian.ply']})
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT),**kw)
    def reply(self,m,code=200):
        b=json.dumps(m,ensure_ascii=False).encode();self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        if avatar_backend.handle_get(self,sys.modules[__name__]):return
        if handle_get(self,sys.modules[__name__]):return
        if self.path=='/api/studio/jobs':
            jobs=[]
            for p in STORE.glob('*/record.json'):
                m=json.loads(p.read_text(encoding='utf8'));log=p.parent/'run.log';m['log']=log.read_text(encoding='utf8',errors='replace')[-5000:] if log.exists() else '';jobs.append(m)
            self.reply({'active':active,'jobs':sorted(jobs,key=lambda m:m['created'],reverse=True)});return
        if self.path=='/':self.path='/studio.html'
        return super().do_GET()
    def do_POST(self):
        if avatar_backend.handle_post(self,sys.modules[__name__]):return
        if handle_post(self,sys.modules[__name__]):return
        global active
        if self.path not in ['/api/studio/import','/api/studio/generate']:self.reply({'error':'Not found'},404);return
        if self.headers.get('Origin') not in (None,'http://127.0.0.1:8770','http://localhost:8770'):self.reply({'error':'Origin denied'},403);return
        try:
            size=int(self.headers.get('Content-Length',0));isgs=self.path.endswith('import')
            if not 0<size<=(250 if isgs else 80)*1024**2:raise ValueError('文件过大或为空')
            with lock:
                if not isgs and active:raise ValueError('已有模型在生成，请等待完成')
                ident=uuid.uuid4().hex
                if not isgs:active=ident
            out=STORE/ident;out.mkdir();data=self.rfile.read(size)
            if isgs:
                (out/'gaussian.ply').write_bytes(data);metrics=describe(out/'gaussian.ply')
                m={'id':ident,'name':'导入的高斯模型','kind':'gaussian','status':'complete','created':time.time(),'source':'本地导入；未执行生成模型','metrics':metrics,'files':['gaussian.ply']};save(out,m)
            else:
                backend=self.headers.get('X-Model-Backend','triposr')
                if backend not in ['triposr','lhm']:raise ValueError('未知生成模型')
                raw_images=[data];motion=None
                if self.headers.get('Content-Type','').startswith('application/json'):
                    payload=json.loads(data)
                    motion=payload.get('motion') or None
                    if motion not in [None,'Dance_I','TaiChi','Jump']:raise ValueError('未知预设动作')
                    if motion and backend!='lhm':raise ValueError('动作驱动仅适用于 LHM++')
                    encoded=payload.get('images',[])
                    if not 1<=len(encoded)<=4:raise ValueError('请选择 1-4 张照片')
                    raw_images=[base64.b64decode(s.split(',',1)[-1],validate=True) for s in encoded]
                if backend=='triposr' and len(raw_images)!=1:raise ValueError('TripoSR 只支持单图；多图请选择 LHM++')
                uploads=out/'uploads';uploads.mkdir()
                for i,raw in enumerate(raw_images):
                    if len(raw)>20*1024**2:raise ValueError('单张图片须小于 20MB')
                    img=Image.open(io.BytesIO(raw))
                    if img.width*img.height>30_000_000:raise ValueError('图片像素数量过大')
                    img.load();ImageOps.exif_transpose(img).save(uploads/f'{i:02d}.png')
                shutil.copy2(uploads/'00.png',out/'upload.png')
                m={'id':ident,'name':'LHM++ 照片生成高斯' if backend=='lhm' else '照片生成的人体网格','kind':'gaussian' if backend=='lhm' else 'mesh','status':'running','created':time.time(),'source':'本机 LHM++ 单图推理 / 标准姿态；背面由 AI 推测' if backend=='lhm' else '本机 TripoSR 单图推理；背面由 AI 推测','files':[],'motion':motion};save(out,m);threading.Thread(target=run,args=(out,m),daemon=True).start()
            self.reply(m,202)
        except Exception as e:
            if not self.path.endswith('import') and 'ident' in locals():
                with lock:
                    if active==ident:active=None
            self.reply({'error':str(e)},400)
if __name__=='__main__':
    seed();print('http://127.0.0.1:8770/studio.html',flush=True);ThreadingHTTPServer(('127.0.0.1',8770),Handler).serve_forever()
