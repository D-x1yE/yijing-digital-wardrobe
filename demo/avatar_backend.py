"""Versioned avatar workflow. Model outputs require explicit human review before V1."""
from pathlib import Path
import json,threading,time,uuid,subprocess,sys,shutil,re,urllib.request
from wardrobe_backend import _save_image,_write
ROOT=Path(__file__).resolve().parent
STORE=ROOT/'avatar_sessions';STORE.mkdir(exist_ok=True)
MUTEX=threading.RLock()
PRESETS={'Dance_I','TaiChi','Jump'}

def load(ident):
 if not re.fullmatch('[a-f0-9]{32}',ident or ''):raise ValueError('无效项目编号')
 return json.loads((STORE/ident/'session.json').read_text(encoding='utf8'))
def persist(s):_write(STORE/s['id']/'session.json',s)
def linux(p):
 p=Path(p).resolve().as_posix();return '/mnt/'+p[0].lower()+p[2:]
def capabilities():
 try:
  with urllib.request.urlopen('http://127.0.0.1:8791/health',timeout=.3) as r:qwen_ready=r.status==200
 except Exception:qwen_ready=False
 return {'qwen_ready':qwen_ready,'fashn_weights_ready':(Path('D:/AvatarPipeline/weights/fashn/model.safetensors').exists()),'qwen_weights_ready':Path('D:/AvatarPipeline/weights/qwen/Qwen3.5-4B-Q4_K_M.gguf').exists()}

def launch(studio,s,kind,source,motion=None):
 with studio.lock:
  if studio.active:raise ValueError('GPU 正忙，请等待当前任务完成')
  job=uuid.uuid4().hex;studio.active=job
 s['busy']=True;s['stage']={'tryon':'生成二维试衣候选','rebuild':'重建三维版本','animate':'生成动作视频'}[kind];s['error']=None
 s['task']={'id':job,'kind':kind,'started':time.time()};persist(s)
 threading.Thread(target=worker,args=(studio,s['id'],job,kind,source,motion),daemon=True).start()

def worker(studio,sid,job,kind,source,motion):
 directory=STORE/sid;out=studio.STORE/job;out.mkdir(exist_ok=True)
 try:
  if kind=='tryon':
   with MUTEX:s=load(sid);candidate=s['candidate']
   edited=directory/'candidates'/candidate['id']/'edited';edited.mkdir(parents=True,exist_ok=True)
   command=['wsl','-d','Ubuntu-22.04','--','/opt/fashn-venv/bin/python','-u',linux(ROOT/'run_fashn.py'),linux(directory/source),linux(directory/candidate['garment']),linux(edited),candidate['category'],str(candidate['seed']),candidate['photo_type']]
   with (out/'run.log').open('w',encoding='utf8') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
   expected=[edited/f'{i:02d}.png' for i in range(len(s['versions'][s['current']]['images']))]
   if not all(p.exists() for p in expected):raise RuntimeError('试衣未输出完整视图')
   with MUTEX:
    s=load(sid);s['candidate'].update(status='review',images=[p.relative_to(ROOT).as_posix() for p in expected],review={'identity':False,'garment':False,'views':False});s['stage']='请检查候选图并确认，尚未重建 V1';persist(s)
  else:
   uploads=out/'uploads';uploads.mkdir(exist_ok=True)
   for i,p in enumerate(sorted((directory/source).glob('*.png'))):shutil.copy2(p,uploads/f'{i:02d}.png')
   if not list(uploads.glob('*.png')):raise RuntimeError('缺少版本参考照片')
   record={'id':job,'name':'对话衣橱 / '+kind,'kind':'gaussian','status':'running','created':time.time(),'source':'LHM++ / 版本化衣橱','files':[],'motion':motion}
   studio.save(out,record);studio.run(out,record)
   if record['status']!='complete':raise RuntimeError(record.get('error','模型运行失败'))
   with MUTEX:
    s=load(sid)
    if kind=='rebuild':
     key='V'+str(len(s['versions']));parent=s['current'] if s['versions'] else None
     s['versions'][key]={'id':key,'parent':parent,'job':job,'source':source,'images':[(directory/source/p.name).relative_to(ROOT).as_posix() for p in sorted((directory/source).glob('*.png'))],'animations':{},'created':time.time()};s['current']=key;s['display']={'mode':'view','version':key,'view':'front'}
     if s.get('candidate') and source.startswith('candidates/'):
      s['candidate']['status']='accepted';s['candidate']['version']=key
      s['versions'][key]['tryon']={k:s['candidate'][k] for k in ['id','garment','category','photo_type','seed','review']}
    else:s['versions'][s['task']['version']]['animations'][motion]=job
    s['stage']='完成';persist(s)
 except Exception as e:
  with MUTEX:s=load(sid);s['error']=str(e);s['stage']='任务失败';persist(s)
 finally:
  with MUTEX:s=load(sid);s['busy']=False;persist(s)
  with studio.lock:
   if studio.active==job:studio.active=None

def qwen(text,s):
 system='你是工具分类器。比较/对比=compare；撤销/恢复原版=undo；跳舞/太极/跳跃=animate；查看背面/侧面/正面=view；新建人物=reconstruct；要求新换衣且已附商品图=tryon，否则提示上传并选none。没有商品图不影响比较、查看、动作、撤销。已有版本时不重复reconstruct。reply用简短中文说明将调用的工具，不声称生成已完成。version只能选已有的单个版本；比较时选V0之外的目标版本，未指定取最新版本。motion舞蹈=Dance_I，太极=TaiChi，跳跃=Jump。view默认front。category上装=tops，下装=bottoms，连衣裙=one-pieces。不要批准试衣检查。项目状态:'+json.dumps({'current':s['current'],'versions':list(s['versions']),'busy':s['busy'],'has_garment':s.get('_has_garment',False),'recent_user_requests':[m['content'] for m in s.get('messages',[]) if m['role']=='user'][-3:]},ensure_ascii=False)
 payload={'model':'Qwen3.5-4B','messages':[{'role':'system','content':system}]+[{'role':'user','content':text}],'versions':list(s['versions']),'max_tokens':400,'temperature':.1,'response_format':{'type':'json_object'},'chat_template_kwargs':{'enable_thinking':False}}
 req=urllib.request.Request('http://127.0.0.1:8791/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=180) as r:d=json.load(r)
  value=json.loads(d['choices'][0]['message']['content']);assert value.get('action') in {'none','reconstruct','tryon','view','compare','undo','animate'};return value
 except Exception as e:raise RuntimeError('Qwen 对话服务未就绪或返回无效工具指令；可使用页面按钮。'+str(e))

def handle_get(h,studio):
 if h.path=='/api/avatar/state':
  sessions=[]
  for p in STORE.glob('*/session.json'):
   try:sessions.append(json.loads(p.read_text(encoding='utf8')))
   except ValueError:pass
  h.reply({'sessions':sorted(sessions,key=lambda x:x['created'],reverse=True),'capabilities':capabilities(),'active':studio.active});return True
 return False

def handle_post(h,studio):
 if h.path not in {'/api/avatar/create','/api/avatar/action'}:return False
 try:
  if h.headers.get('Origin') not in {None,'http://127.0.0.1:8770','http://localhost:8770'}:raise ValueError('Origin denied')
  length=int(h.headers.get('Content-Length',0))
  if not 0<length<100*1024**2:raise ValueError('请求大小无效')
  p=json.loads(h.rfile.read(length))
  with MUTEX:
   if h.path.endswith('create'):
    people=p.get('people',[])
    if not isinstance(people,list) or not 1<=len(people)<=4:raise ValueError('请选择1–4张人物照片')
    ident=uuid.uuid4().hex;directory=STORE/ident;directory.mkdir()
    for i,im in enumerate(people):_save_image(im,directory/'original'/f'{i:02d}.png')
    plan=qwen(str(p.get('intent') or '请用照片建立原始三维人物V0'),{'current':None,'versions':{},'busy':False,'messages':[]})
    if plan['action']!='reconstruct':raise ValueError('Qwen未选择建模工具：'+plan.get('reply',''))
    s={'initial_plan':plan,'id':ident,'name':str(p.get('name','我的数字衣橱'))[:80],'created':time.time(),'versions':{},'current':None,'candidate':None,'messages':[],'busy':False,'stage':'已保存原图'};persist(s);launch(studio,s,'rebuild','original');h.reply(s,202);return True
   s=load(p.get('id'));action=p.get('action')
   if s.get('busy') and action not in {'chat','view','compare'}:raise ValueError('当前项目正在执行任务')
   if action=='chat':
    # Slow model request outside the project lock.
    pass
   elif action=='tryon':
    submit_tryon(studio,s,p)
   elif action=='approve':
    c=s.get('candidate')
    if not c or c['status']!='review':raise ValueError('没有待确认的试衣结果')
    if not all(p.get('review',{}).get(k) is True for k in ['identity','garment','views']):raise ValueError('请逐项检查身份、衣服细节和视角一致性')
    if s['current']!=c['base']:raise ValueError('当前版本已改变，请切回候选对应版本再确认')
    c['review']=p['review'];launch(studio,s,'rebuild',f"candidates/{c['id']}/edited")
   elif action=='reject':
    if not s.get('candidate') or s['candidate']['status']!='review':raise ValueError('没有待确认结果')
    s['candidate']['status']='rejected';s['stage']='候选已拒绝，原版本保留';persist(s)
   else:execute(studio,s,action,p)
  if action=='chat':
   instruction=qwen(str(p.get('text',''))[:2000],{**s,'_has_garment':bool(p.get('garment'))})
   with MUTEX:
    s=load(s['id']);
    if instruction['action']=='tryon':submit_tryon(studio,s,{**p,'category':instruction.get('category',p.get('category','tops'))})
    else:execute(studio,s,instruction['action'],instruction)
    s=load(s['id']);s['messages']+= [{'role':'user','content':str(p.get('text',''))[:2000]},{'role':'assistant','content':str(instruction.get('reply',''))[:2000]}];persist(s)
   h.reply({'session':s,'instruction':instruction});return True
  h.reply({'session':s},202)
 except Exception as e:h.reply({'error':str(e)},400)
 return True


def submit_tryon(studio,s,p):
 if not s['current']:raise ValueError('请先完成 V0')
 if not capabilities()['fashn_weights_ready']:raise ValueError('FASHN权重尚未下载完成')
 category=p.get('category','tops');photo_type=p.get('photo_type','flat-lay')
 if category not in {'tops','bottoms','one-pieces'} or photo_type not in {'flat-lay','model'}:raise ValueError('衣物类别或图片类型无效')
 cid=uuid.uuid4().hex;directory=STORE/s['id'];_save_image(p['garment'],directory/'candidates'/cid/'garment.png')
 if s.get('candidate'):s.setdefault('candidate_history',[]).append(s['candidate'])
 s['candidate']={'id':cid,'base':s['current'],'garment':f'candidates/{cid}/garment.png','category':category,'photo_type':photo_type,'seed':int(p.get('seed',42)),'status':'generating'};launch(studio,s,'tryon',s['versions'][s['current']]['source'])

def execute(studio,s,action,p):
 if action=='none':return
 if action=='reconstruct':raise ValueError('当前已有原始版本，无需重复建立V0')
 version=p.get('version') or s['current']
 if action=='compare' and version=='V0' and len(s['versions'])>1:version=list(s['versions'])[-1]
 if version not in s['versions']:raise ValueError('目标版本不存在')
 if action=='undo':
  if s['busy']:raise ValueError('请等待任务结束后撤销')
  parent=s['versions'][s['current']]['parent']
  if parent:s['current']=parent
  s.pop('display',None)
 elif action=='select':
  if s['busy']:raise ValueError('请等待任务结束')
  s['current']=version;s.pop('display',None)
 elif action=='animate':
  if s['busy']:raise ValueError('项目正在执行任务')
  motion=p.get('motion','Dance_I')
  if motion not in PRESETS:raise ValueError('未知动作')
  if motion not in s['versions'][version]['animations']:
   launch(studio,s,'animate',s['versions'][version]['source'],motion);s['task']['version']=version
  s['display']={'mode':'animation','version':version,'motion':motion}
 elif action in {'view','compare'}:s['display']={'mode':action,'version':version,'view':p.get('view','front') if p.get('view') in {'front','side','back'} else 'front'}
 else:raise ValueError('未知操作')
 persist(s)
