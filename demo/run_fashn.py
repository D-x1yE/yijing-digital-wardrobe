"""Real FASHN VTON execution; CPU pose/parser leave VRAM for diffusion."""
import os,sys,json,time
from pathlib import Path
os.environ.setdefault('HF_HOME','/mnt/d/AvatarPipeline/hf-cache')
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
import torch
from PIL import Image
from fashn_vton import TryOnPipeline
class LocalPipeline(TryOnPipeline):
 def _setup_pose_model(self):
  from fashn_vton.dwpose import DWposeDetector
  self.pose_model=DWposeDetector(checkpoints_dir=str(Path(self.weights_dir)/'dwpose'),device='cpu')
 def _setup_hp_model(self):
  from fashn_human_parser import FashnHumanParser
  self.hp_model=FashnHumanParser(model_id='/mnt/d/AvatarPipeline/weights/parser',device='cpu')
if __name__=='__main__':
 source,garment,out,category,seed,photo_type=sys.argv[1:];out=Path(out);out.mkdir(exist_ok=True,parents=True)
 started=time.time();torch.set_num_threads(4)
 pipeline=LocalPipeline('/mnt/d/AvatarPipeline/weights/fashn',device='cuda')
 paths=sorted(Path(source).glob('*.png'))
 if not 1<=len(paths)<=4:raise ValueError('Expected 1–4 views')
 for i,p in enumerate(paths):
  print(f'STAGE: FASHN view {i+1}/{len(paths)}',flush=True)
  result=pipeline(person_image=Image.open(p).convert('RGB'),garment_image=Image.open(garment).convert('RGB'),category=category,garment_photo_type=photo_type,num_samples=1,num_timesteps=20,seed=int(seed),segmentation_free=False)
  result.images[0].save(out/f'{i:02d}.png')
 (out/'fashn.json').write_text(json.dumps({'model':'fashn-ai/fashn-vton-1.5','views':len(paths),'seconds':time.time()-started,'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30,'seed':int(seed),'category':category,'photo_type':photo_type}),encoding='utf8')
