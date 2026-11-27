"""Verified-size parallel ranged download for large public setup artifacts."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests,hashlib
def fetch(url,path,size,sha=None,workers=12):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size==size:return path
    parts=path.with_name(path.name+'.parts');parts.mkdir(exist_ok=True)
    starts=list(range(0,size,4*1024**2))
    def get(start):
        end=min(start+4*1024**2,size)-1;p=parts/str(start)
        if p.exists() and p.stat().st_size==end-start+1:return
        for attempt in range(4):
            try:
                s=requests.Session();s.trust_env=False
                r=s.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,40));r.raise_for_status()
                if len(r.content)!=end-start+1:raise ValueError('range response size')
                p.write_bytes(r.content);return
            except Exception:
                if attempt==3:raise
    with ThreadPoolExecutor(max_workers=workers) as ex:list(ex.map(get,starts))
    h=hashlib.sha256()
    with path.open('wb') as f:
        for start in starts:b=(parts/str(start)).read_bytes();h.update(b);f.write(b)
    if sha and h.hexdigest()!=sha:raise ValueError('SHA256 mismatch')
    for start in starts:(parts/str(start)).unlink()
    parts.rmdir();print('ready',path,size,flush=True);return path
