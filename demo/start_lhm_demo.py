from pathlib import Path
import subprocess,sys,time,webbrowser,urllib.request
ROOT=Path(__file__).resolve().parent
URL='http://127.0.0.1:8770/'
def ready():
 try:
  with urllib.request.urlopen(URL+'api/studio/jobs',timeout=1) as r:return r.status==200
 except Exception:return False
if not ready():
 log=(ROOT/'lhm_demo_server.log').open('ab')
 subprocess.Popen([sys.executable,'-u',str(ROOT/'serve_studio.py')],cwd=ROOT,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 for _ in range(60):
  if ready():break
  time.sleep(.5)
if ready():webbrowser.open(URL+'lhm.html')
else:raise SystemExit('服务启动失败，请查看 lhm_demo_server.log')
