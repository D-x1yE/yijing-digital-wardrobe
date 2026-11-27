"""Fetch published LHM++ inference assets from ModelScope, verify SHA256."""
import requests,json,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'vendor/LHM-plusplus-main/pretrained_models'
def download(task):
    repo,item,target=task;size=item['Size'];target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.stat().st_size==size:return str(target)+' exists'
    part=target.with_name(target.name+'.partial')
    for attempt in range(5):
        try:
            s=requests.Session();s.trust_env=False
            offset=part.stat().st_size if part.exists() else 0
            url=f'https://modelscope.cn/api/v1/models/{repo}/repo'
            if size>100*1024**2:
                chunks=target.with_name(target.name+'.chunks');chunks.mkdir(exist_ok=True)
                def segment(start):
                    end=min(start+8*1024**2,size)-1;q=chunks/str(start)
                    if q.exists() and q.stat().st_size==end-start+1:return
                    for retry in range(5):
                        try:
                            ss=requests.Session();ss.trust_env=False
                            rr=ss.get(url,params={'Revision':'master','FilePath':item['Path']},headers={'Range':f'bytes={start}-{end}'},timeout=(20,40));rr.raise_for_status()
                            if rr.status_code!=206 or len(rr.content)!=end-start+1:raise ValueError('invalid range')
                            q.write_bytes(rr.content);return
                        except Exception:
                            if retry==4:raise
                starts=range(0,size,8*1024**2)
                with ThreadPoolExecutor(max_workers=24) as ranges:
                    for _ in ranges.map(segment,starts):pass
                with part.open('wb') as f:
                    for start in starts:
                        f.write((chunks/str(start)).read_bytes())
            else:
              with s.get(url,params={'Revision':'master','FilePath':item['Path']},headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(25,90)) as r:
                r.raise_for_status()
                if offset and r.status_code!=206:offset=0
                with part.open('ab' if offset else 'wb') as f:
                    for chunk in r.iter_content(4*1024**2):f.write(chunk)
            if part.stat().st_size!=size:raise ValueError('size mismatch')
            h=hashlib.sha256()
            with part.open('rb') as f:
                for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
            if item.get('Sha256') and h.hexdigest()!=item['Sha256']:raise ValueError('hash mismatch')
            part.replace(target);return str(target)+' verified'
        except Exception as e:print('retry',item['Path'],attempt,str(e),flush=True)
    raise RuntimeError(item['Path'])
if __name__=='__main__':
    tasks=[]
    for name in ['LHMPP-700M-SMPLX-FREE','LHMPP-Prior']:
        files=json.loads((ROOT/f'tmp/{name}-files.json').read_text())['Data']['Files']
        for i in files:
            p=i['Path']
            selected=name!='LHMPP-Prior' and p in ['config.json','model.safetensors']
            selected|=name=='LHMPP-Prior' and (p.startswith(('human_model_files/smplx/','dense_sample_points/','voxel_grid/')) or p in ['BiRefNet-general-epoch_244.pth','arcface_resnet18.pth']) and not p.endswith('.zip')
            if i['Type']=='blob' and selected:
                out=DEST/('Damo_XR_Lab/'+name if name!='LHMPP-Prior' else '')/p
                tasks.append(('Damo_XR_Lab/'+name,i,out))
    print('download bytes',sum(t[1]['Size'] for t in tasks),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for res in pool.map(download,tasks):print(res,flush=True)
