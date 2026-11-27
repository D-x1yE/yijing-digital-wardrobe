"""Local-only demo server with a real, serialized GPU inference endpoint."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import subprocess
import sys
import threading
import os

ROOT=Path(__file__).resolve().parent
state={'running':False,'last_exit':None}
lock=threading.Lock()

def inference():
    try:
        with (ROOT/'real_inference.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u',str(ROOT/'run_real_model.py'),'--samples','3','--views','17'],stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT.parent))
        with lock: state['last_exit']=result.returncode
    finally:
        with lock: state['running']=False

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs): super().__init__(*args,directory=str(ROOT),**kwargs)
    def reply(self,obj,code=200):
        data=json.dumps(obj).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if self.path=='/api/status':
            log=ROOT/'real_inference.log'
            self.reply({**state,'log':log.read_text(encoding='utf-8',errors='replace')[-4500:] if log.exists() else ''});return
        return super().do_GET()
    def do_POST(self):
        if self.path!='/api/generate': self.reply({'error':'Not found'},404);return
        # Restrict API to same-origin browser requests and loopback server binding.
        origin=self.headers.get('Origin')
        if origin and origin not in ('http://127.0.0.1:8765','http://localhost:8765'):
            self.reply({'error':'Origin denied'},403);return
        with lock:
            if state['running']: self.reply({'running':True},409);return
            state['running']=True
        threading.Thread(target=inference,daemon=True).start();self.reply({'running':True},202)

if __name__=='__main__':
    print('Open http://127.0.0.1:8765/real.html',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8765),Handler).serve_forever()
