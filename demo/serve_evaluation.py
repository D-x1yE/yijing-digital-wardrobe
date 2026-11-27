from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import base64,json,re
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evaluation_render_20261123'
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT),**kw)
    def do_POST(self):
        if self.path!='/api/capture':self.send_error(404);return
        try:
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<20*1024**2:raise ValueError('Invalid size')
            data=json.loads(self.rfile.read(size));a=data['angle'];r=data['ratio']
            evaluation=data.get('evaluation',OUT.name)
            if not isinstance(evaluation,str) or not re.fullmatch(r'evaluation_[A-Za-z0-9_]+',evaluation):raise ValueError('Invalid evaluation')
            target=ROOT/evaluation
            if not (target/'evaluation.json').is_file():raise ValueError('Unknown evaluation')
            if a not in (0,90,180) or r not in (1,2):raise ValueError('Invalid view')
            raw=base64.b64decode(data.pop('data').split(',',1)[1],validate=True)
            if not raw.startswith(bytes([137,80,78,71])):raise ValueError('PNG required')
            (target/f'webgl_{a}_{r}x.png').write_bytes(raw)
            (target/f'webgl_{a}_{r}x.json').write_text(json.dumps(data,indent=2),encoding='utf8')
            self.send_response(200);self.end_headers();self.wfile.write(b'OK')
        except Exception as e:self.send_error(400,str(e))
ThreadingHTTPServer(('127.0.0.1',8771),Handler).serve_forever()
