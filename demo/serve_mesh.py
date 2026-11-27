"""Loopback photo upload and serialized CUDA reconstruction."""
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json,subprocess,sys,threading,uuid,io
from PIL import Image
ROOT=Path(__file__).resolve().parent
lock=threading.Lock();state={'running':False,'result':None,'error':None,'job':None}
def infer(job):
    out=ROOT/'mesh_results'/job
    try:
        with (out/'run.log').open('w',encoding='utf-8') as log:
            p=subprocess.run([sys.executable,'-u',str(ROOT/'run_mesh.py'),str(out/'upload.png'),str(out)],stdout=log,stderr=subprocess.STDOUT)
        with lock:
            if p.returncode==0:state['result']='mesh_results/'+job
            else:state['error']='生成失败，请查看运行日志并更换清晰全身照片。'
    except Exception as e:
        with lock:state['error']=str(e)
    finally:
        with lock:state['running']=False
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT),**kw)
    def reply(self,data,code=200):
        b=json.dumps(data).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        if self.path=='/api/mesh/status':
            with lock:s=state.copy()
            p=ROOT/'mesh_results'/str(s['job'])/'run.log'
            s['log']=p.read_text(encoding='utf-8',errors='replace')[-3500:] if p.exists() else ''
            self.reply(s);return
        return super().do_GET()
    def do_POST(self):
        if self.path!='/api/mesh/generate':self.reply({'error':'Not found'},404);return
        if self.headers.get('Origin') not in (None,'http://127.0.0.1:8766','http://localhost:8766'):self.reply({'error':'Origin denied'},403);return
        n=int(self.headers.get('Content-Length','0'))
        if not 0<n<=20*1024*1024:self.reply({'error':'图片须小于 20MB'},400);return
        with lock:
            if state['running']:self.reply({'error':'已有任务正在生成，请稍候'},409);return
            state['running']=True
        try:
            img=Image.open(io.BytesIO(self.rfile.read(n)));img.load()
            if img.width*img.height>30_000_000:raise ValueError('图片分辨率过大')
            job=uuid.uuid4().hex;out=ROOT/'mesh_results'/job;out.mkdir(parents=True)
            img.save(out/'upload.png')
            with lock:state.update(job=job,result=None,error=None)
            threading.Thread(target=infer,args=(job,),daemon=True).start();self.reply({'job':job},202)
        except Exception as e:
            with lock:state['running']=False
            self.reply({'error':str(e)},400)
if __name__=='__main__':
    print('http://127.0.0.1:8766/mesh.html',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8766),Handler).serve_forever()
