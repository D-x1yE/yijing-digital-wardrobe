"""Local CPU-only Qwen3.5-4B tool planner; OpenAI-compatible narrow endpoint."""
import json,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from llama_cpp import Llama
MODEL='/mnt/d/AvatarPipeline/weights/qwen/Qwen3.5-4B-Q4_K_M.gguf'
model=Llama(model_path=MODEL,n_ctx=4096,n_threads=6,n_threads_batch=6,n_gpu_layers=0,chat_format='chatml',verbose=False)
LOCK=threading.Lock()
class Handler(BaseHTTPRequestHandler):
 def reply(self,d,status=200):
  raw=json.dumps(d,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 def do_GET(self):self.reply({'status':'ok','model':'Qwen3.5-4B','backend':'llama-cpp-python CPU Q4_K_M','vision':False})
 def do_POST(self):
  try:
   n=int(self.headers.get('Content-Length',0))
   if not 0<n<=100000:raise ValueError('Invalid request size')
   p=json.loads(self.rfile.read(n))
   schema={'type':'object','properties':{'reply':{'type':'string'},'action':{'type':'string','enum':['none','reconstruct','tryon','view','compare','undo','animate']},'category':{'type':'string','enum':['tops','bottoms','one-pieces']},'version':{'type':['string','null'],'enum':[None]+[v for v in p.get('versions',['V0','V1']) if isinstance(v,str) and len(v)<12]},'motion':{'type':'string','enum':['Dance_I','TaiChi','Jump']},'view':{'type':'string','enum':['front','side','back']}},'required':['reply','action','version','motion','view','category'],'additionalProperties':False}
   with LOCK:
    result=model.create_chat_completion(messages=p['messages'],temperature=.1,max_tokens=min(int(p.get('max_tokens',400)),500),response_format={'type':'json_object','schema':schema})
   self.reply(result)
  except Exception as e:self.reply({'error':str(e)},500)
print('Qwen CPU service http://127.0.0.1:8791',flush=True)
ThreadingHTTPServer(('127.0.0.1',8791),Handler).serve_forever()
