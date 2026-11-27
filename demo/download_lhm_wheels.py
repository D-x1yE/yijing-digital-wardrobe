"""Download pinned Linux wheels in verified HTTP ranges for WSL setup."""
import requests,re,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent.parent/'vendor/lhm_wheels';ROOT.mkdir(exist_ok=True)
packages=[('torch','2.3.0'),('torchvision','0.18.0'),('xformers','0.0.26.post1'),('triton','2.3.0'),('nvidia-cuda-nvcc-cu12','12.1.105')]
s=requests.Session();s.trust_env=False
packages += [('nvidia-cuda-nvrtc-cu12','12.1.105'),('nvidia-cuda-runtime-cu12','12.1.105'),('nvidia-cuda-cupti-cu12','12.1.105'),('nvidia-cudnn-cu12','8.9.2.26'),('nvidia-cublas-cu12','12.1.3.1'),('nvidia-cufft-cu12','11.0.2.54'),('nvidia-curand-cu12','10.3.2.106'),('nvidia-cusolver-cu12','11.4.5.107'),('nvidia-cusparse-cu12','12.1.0.106'),('nvidia-nccl-cu12','2.20.5'),('nvidia-nvtx-cu12','12.1.105')]
known={
 'torch':('torch-2.3.0-cp310-cp310-manylinux1_x86_64.whl','43/e5/2ddae60ae999b224aceb74490abeb885ee118227f866cb12046f0481d4c9',779141888),
 'torchvision':('torchvision-0.18.0-cp310-cp310-manylinux1_x86_64.whl','d4/7e/d41b771dbffa927b9cc37372b1e18c881348cd18a0e4ad73f2c6bdf56c0e',6954946),
 'xformers':('xformers-0.0.26.post1-cp310-cp310-manylinux2014_x86_64.whl','32/e7/27003645ef99e7571fb6964cd2f39da3f1b3f3011aa00bb2d3ac9b790757',222745400)}
def fetch(pkg):
    name,ver=pkg;ss=requests.Session();ss.trust_env=False
    if name in known:
        fn,sub,size=known[name];f={'filename':fn,'url':'https://files.pythonhosted.org/packages/'+sub+'/'+fn,'size':size}
    else:
        cache=ROOT/(name+'.metadata.json')
        if cache.exists():data=__import__('json').loads(cache.read_text())
        else:
            for retry in range(5):
                try:
                    data=ss.get(f'https://pypi.org/pypi/{name}/{ver}/json',timeout=(20,45)).json();cache.write_text(__import__('json').dumps(data));break
                except Exception:
                    if retry==4:raise
        choices=[f for f in data['urls'] if 'manylinux' in f['filename'] and 'x86_64' in f['filename'] and ('cp310' in f['filename'] or 'py3-' in f['filename'])]
        f=choices[0]
    target=ROOT/f['filename'];size=f['size']
    if target.exists() and target.stat().st_size==size:return
    parts=ROOT/(f['filename']+'.parts');parts.mkdir(exist_ok=True)
    def chunk(start):
        end=min(start+8*1024**2,size)-1;p=parts/str(start)
        if p.exists() and p.stat().st_size==end-start+1:return
        for attempt in range(5):
            try:
                rs=requests.Session();rs.trust_env=False
                r=rs.get(f['url'].replace('https://files.pythonhosted.org/','https://mirrors.aliyun.com/pypi/'),headers={'Range':f'bytes={start}-{end}'},timeout=(20,35));r.raise_for_status()
                if len(r.content)!=end-start+1:raise ValueError('range size')
                p.write_bytes(r.content);return
            except Exception:
                if attempt==4:raise
    starts=list(range(0,size,8*1024**2))
    with ThreadPoolExecutor(max_workers=12) as e:list(e.map(chunk,starts))
    h=hashlib.sha256()
    with target.open('wb') as out:
        for start in starts:b=(parts/str(start)).read_bytes();h.update(b);out.write(b)
    if 'digests' in f:assert h.hexdigest()==f['digests']['sha256']
    for start in starts:(parts/str(start)).unlink()
    parts.rmdir();print('verified',name,size,flush=True)
with ThreadPoolExecutor(max_workers=3) as p:list(p.map(fetch,packages))
