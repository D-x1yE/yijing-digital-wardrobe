from pathlib import Path
import subprocess,sys,time,webbrowser,urllib.request
ROOT=Path(__file__).resolve().parent
FLAGS=getattr(subprocess,'CREATE_NO_WINDOW',0)
def ready(url):
 try:
  with urllib.request.urlopen(url,timeout=1) as r:return r.status==200
 except Exception:return False
if not ready('http://127.0.0.1:8770/api/avatar/state'):
 if ready('http://127.0.0.1:8770/api/studio/jobs'):raise SystemExit('8770 正在运行旧版服务，请关闭旧服务后重试。')
 with (ROOT/'avatar_server.log').open('ab') as log:
  subprocess.Popen([sys.executable,'-u',str(ROOT/'serve_studio.py')],cwd=ROOT,stdout=log,stderr=log,creationflags=FLAGS)
 for _ in range(60):
  if ready('http://127.0.0.1:8770/api/avatar/state'):break
  time.sleep(.5)
if not ready('http://127.0.0.1:8791/health'):
 lock=ROOT/'qwen_starting.txt'
 if not lock.exists() or time.time()-lock.stat().st_mtime>180:
  lock.write_text(str(time.time()))
  p=(ROOT/'serve_qwen.py').resolve().as_posix();linux='/mnt/'+p[0].lower()+p[2:]
  with (ROOT/'qwen_service.log').open('ab') as log:
   subprocess.Popen(['wsl','-d','Ubuntu-22.04','--','/opt/fashn-venv/bin/python','-u',linux],cwd=ROOT,stdout=log,stderr=log,creationflags=FLAGS)
webbrowser.open('http://127.0.0.1:8770/avatar.html')
