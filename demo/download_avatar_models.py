from pathlib import Path
import requests,concurrent.futures,hashlib,time
ROOT=Path('D:/AvatarPipeline/weights')
ITEMS=[('fashn-ai/fashn-vton-1.5','model.safetensors','fashn/model.safetensors',1943668048,'d6cd38286885bc29fa487ea9383f80ffeb95862e7747c630d42c5d3c05bdd35a'),('fashn-ai/DWPose','yolox_l.onnx','fashn/dwpose/yolox_l.onnx',216746733,'7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411'),('fashn-ai/DWPose','dw-ll_ucoco_384.onnx','fashn/dwpose/dw-ll_ucoco_384.onnx',134399116,'724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843'),('unsloth/Qwen3.5-4B-GGUF','Qwen3.5-4B-Q4_K_M.gguf','qwen/Qwen3.5-4B-Q4_K_M.gguf',2740937888,'00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4'),('unsloth/Qwen3.5-4B-GGUF','mmproj-F16.gguf','qwen/mmproj-F16.gguf',672423616,'cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864')]
for repo,name,rel,total,sha in ITEMS:
 p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():continue
 part=p.with_suffix('.part');start=part.stat().st_size if part.exists() else 0
 print('START',name,start,total,flush=True)
 ranges=[(a,min(a+64*1024**2,total)-1) for a in range(start,total,64*1024**2)]
 def fetch(bounds):
  a,b=bounds;chunk=p.with_name(name+'.chunk'+str(a));n=chunk.stat().st_size if chunk.exists() else 0
  for attempt in range(5):
   try:
    if n==b-a+1:return chunk
    r=requests.get(f'https://modelscope.cn/models/{repo}/resolve/master/{name}',headers={'Range':f'bytes={a+n}-{b}'},stream=True,timeout=(20,90));r.raise_for_status()
    if r.status_code!=206:raise RuntimeError('Server did not honor byte range')
    with chunk.open('ab') as f:
     for data in r.iter_content(2**20):f.write(data)
    n=chunk.stat().st_size
    if n!=b-a+1:raise RuntimeError('Incomplete chunk')
    print('CHUNK',name,a,flush=True);return chunk
   except Exception:
    n=chunk.stat().st_size if chunk.exists() else 0
    if attempt==4:raise
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:chunks=list(pool.map(fetch,ranges))
 with part.open('ab') as out:
  for ch in chunks:
   with ch.open('rb') as f:
    for data in iter(lambda:f.read(2**20),b''):out.write(data)
 digest=hashlib.sha256()
 with part.open('rb') as f:
  for data in iter(lambda:f.read(2**20),b''):digest.update(data)
 if digest.hexdigest()!=sha:raise RuntimeError('Checksum mismatch '+name)
 part.replace(p)
 for ch in chunks:ch.unlink()
 print('VERIFIED',name,flush=True)
