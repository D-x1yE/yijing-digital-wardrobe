"""Single image -> TripoSR neural field -> colored triangle mesh (local CUDA)."""
from pathlib import Path
import sys, time, json, types
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'vendor/TripoSR-main'))
import numpy as np
import torch
from PIL import Image, ImageOps
from skimage.measure import marching_cubes

# torchmcubes returns coordinates in z,y,x order; keep upstream's conversion.
def cpu_mc(level, threshold):
    v,f,_,_=marching_cubes(level.detach().cpu().numpy(),level=threshold)
    return torch.from_numpy(v[:,[2,1,0]].copy()),torch.from_numpy(f.copy().astype(np.int64))
sys.modules['torchmcubes']=types.SimpleNamespace(marching_cubes=cpu_mc)
from tsr.models.tokenizers import image as image_module
# The full checkpoint already contains DINO weights; load its config offline.
image_module.hf_hub_download=lambda **kw:str(ROOT.parent/'vendor/TripoSR-weights/dino_config.json')
from tsr.system import TSR
from tsr.utils import resize_foreground

def main():
    source=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
    started=time.time();torch.set_num_threads(6)
    print('STAGE: preprocessing photo',flush=True)
    img=ImageOps.exif_transpose(Image.open(source)).convert('RGBA')
    img.thumbnail((2048,2048))
    if img.getextrema()[3][0]==255:
        import rembg
        img=rembg.remove(img,session=rembg.new_session('u2netp',providers=['CPUExecutionProvider']))
    if np.count_nonzero(np.asarray(img)[...,3]>128)<100:
        raise ValueError('No foreground found. Try a clear full-body photo or transparent PNG.')
    img.save(out/'foreground.png')
    img=resize_foreground(img,0.85)
    arr=np.array(img).astype(np.float32)/255
    img=Image.fromarray(((arr[:,:,:3]*arr[:,:,3:]+0.5*(1-arr[:,:,3:]))*255).astype('uint8'))
    img.save(out/'input.png')
    print('STAGE: loading TripoSR checkpoint',flush=True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU unavailable')
    model=TSR.from_pretrained(str(ROOT.parent/'vendor/TripoSR-weights'),'config.yaml','model.ckpt').eval().to('cuda')
    model.renderer.set_chunk_size(4096)
    print('STAGE: inferring complete 3D field',flush=True)
    with torch.no_grad():code=model([img],device='cuda')
    print('STAGE: extracting full colored mesh',flush=True)
    mesh=model.extract_mesh(code,True,resolution=192)[0]
    # Remove detached tiny floaters without manufacturing hidden geometry.
    parts=mesh.split(only_watertight=False)
    if len(parts):mesh=max(parts,key=lambda m:len(m.faces))
    mesh.export(out/'human.glb');mesh.export(out/'human.ply')
    # Viewer uses normalized coordinates and the actual generated triangles.
    v=np.asarray(mesh.vertices);center=(v.min(0)+v.max(0))/2
    v=(v-center)/np.max(np.ptp(v,axis=0))*2
    faces=np.asarray(mesh.faces)
    packed=np.concatenate([v[faces].reshape(-1,3),np.asarray(mesh.vertex_normals)[faces].reshape(-1,3),np.asarray(mesh.visual.vertex_colors)[faces,:3].reshape(-1,3)/255],axis=1).astype('<f4')
    packed.tofile(out/'triangles.bin')
    meta={'vertices':len(mesh.vertices),'faces':len(mesh.faces),'seconds':round(time.time()-started,1),'model':'TripoSR','backside':'AI inferred from single image','watertight':bool(mesh.is_watertight),'gpu_peak_gb':round(torch.cuda.max_memory_allocated()/2**30,2)}
    (out/'mesh.json').write_text(json.dumps(meta),encoding='utf-8')
    print('DONE: '+json.dumps(meta),flush=True)
if __name__=='__main__':main()
