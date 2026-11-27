"""Real GPS-Gaussian pretrained inference; no synthetic human geometry."""
import argparse
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'vendor/GPS-Gaussian-main'), str(ROOT / 'demo/runtime')]
from config.stereo_human_config import ConfigStereoHuman
from lib.network import RtStereoHumanModel
from lib.human_loader import StereoHumanDataset
from lib.utils import get_novel_calib
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def gather(data):
    arrays = {k: [] for k in ('xyz', 'rgb', 'rot', 'scale', 'opacity')}
    for view in ('lmain', 'rmain'):
        d = data[view]
        v = d['pts_valid'][0]
        arrays['xyz'].append(d['xyz'][0][v])
        for key, field, channels in [('rgb','img',3),('rot','rot_maps',4),('scale','scale_maps',3),('opacity','opacity_maps',1)]:
            a = d[field][0].permute(1,2,0).reshape(-1,channels)[v]
            arrays[key].append(a * .5 + .5 if key == 'rgb' else a)
    arrays = {k: torch.cat(v).contiguous() for k,v in arrays.items()}
    valid = torch.isfinite(arrays['xyz']).all(1) & torch.isfinite(arrays['scale']).all(1)
    return {k: v[valid] for k,v in arrays.items()}

def render(data, g, bg):
    d = data['novel_view']
    settings = GaussianRasterizationSettings(
        image_height=int(d['height'][0]), image_width=int(d['width'][0]),
        tanfovx=math.tan(float(d['FovX'][0])*.5), tanfovy=math.tan(float(d['FovY'][0])*.5),
        bg=torch.tensor(bg,device='cuda',dtype=torch.float32), scale_modifier=1.,
        viewmatrix=d['world_view_transform'][0], projmatrix=d['full_proj_transform'][0],
        sh_degree=0, campos=d['camera_center'][0], prefiltered=False, debug=False)
    result = GaussianRasterizer(settings)(means3D=g['xyz'], means2D=torch.zeros_like(g['xyz']),
        colors_precomp=g['rgb'], opacities=g['opacity'], scales=g['scale'], rotations=g['rot'])
    return result[0]

def export_cloud(g, out):
    from plyfile import PlyData, PlyElement
    a = {k:v.detach().cpu().numpy() for k,v in g.items()}
    np.savez_compressed(out/'gaussians.npz', **a)
    n = len(a['xyz'])
    names = ['x','y','z','nx','ny','nz'] + ['f_dc_'+str(i) for i in range(3)] + ['opacity'] + ['scale_'+str(i) for i in range(3)] + ['rot_'+str(i) for i in range(4)]
    values = np.concatenate([a['xyz'],np.zeros((n,3)),(a['rgb']-.5)/.28209479177387814,
        np.log(np.clip(a['opacity'],1e-6,1-1e-6)/(1-np.clip(a['opacity'],1e-6,1-1e-6))),
        np.log(np.maximum(a['scale'],1e-8)),a['rot']],axis=1).astype('f4')
    vertices = np.empty(n, dtype=[(name,'f4') for name in names])
    for i,name in enumerate(names): vertices[name]=values[:,i]
    PlyData([PlyElement.describe(vertices,'vertex')]).write(str(out/'human_gaussians.ply'))
    # Compact true 3D point preview. Full anisotropic Gaussians remain in NPZ/PLY.
    ids=np.flatnonzero((a['opacity'][:,0]>.1)&np.isfinite(a['xyz']).all(1))[::4]
    xyz=a['xyz'][ids]
    center=np.median(xyz,axis=0)
    scale=max(float(np.percentile(np.linalg.norm(xyz-center,axis=1),95)),1e-6)
    preview=np.column_stack([(xyz-center)/scale,np.clip(a['rgb'][ids],0,1)]).astype('<f4')
    (out/'points.bin').write_bytes(preview.tobytes())
    return n,len(ids)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--samples',type=int,default=1)
    parser.add_argument('--views',type=int,default=9)
    args=parser.parse_args()
    torch.manual_seed(0)
    factory=ConfigStereoHuman();factory.load(str(ROOT/'vendor/GPS-Gaussian-main/config/stage2.yaml'))
    cfg=factory.get_cfg();cfg.defrost();cfg.raft.corr_implementation='reg'
    cfg.dataset.use_processed_data=False;cfg.dataset.use_hr_img=False
    cfg.dataset.test_data_root=str(ROOT/'vendor/minimal_data');cfg.freeze()
    dataset=StereoHumanDataset(cfg.dataset,phase='test')
    model=RtStereoHumanModel(cfg,with_gs_render=True).cuda().eval()
    model.load_state_dict(torch.load(ROOT/'vendor/gps_network_only.pth',map_location='cpu',weights_only=True),strict=True)
    reports=[]
    for idx in range(min(args.samples,len(dataset))):
        torch.cuda.empty_cache()
        start=time.perf_counter();data=dataset.get_test_item(idx,[0,1])
        name=data['name'];out=ROOT/'demo/real_results'/name;out.mkdir(parents=True,exist_ok=True)
        for view in ('lmain','rmain'):
            for k,v in data[view].items(): data[view][k]=v.cuda().unsqueeze(0)
        preprocess=time.perf_counter()-start
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode(): data,_,_=model(data,is_train=False);g=gather(data)
        torch.cuda.synchronize();model_ms=(time.perf_counter()-start)*1000
        count,preview=export_cloud(g,out)
        for v in (0,1):
            src=ROOT/'vendor/minimal_data/img'/name/f'{v}.jpg'
            Image.open(src).save(out/f'input_{v}.jpg')
        timings=[]
        for i,ratio in enumerate(np.linspace(0,1,args.views)):
            data=get_novel_calib(data,cfg.dataset,ratio=float(ratio),intr_key='intr_ori',extr_key='extr_ori')
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode(): image=render(data,g,cfg.dataset.bg_color)
            torch.cuda.synchronize();timings.append((time.perf_counter()-start)*1000)
            arr=(image.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype('uint8')
            Image.fromarray(arr).save(out/f'view_{i:02d}.png')
            print(name,'view',i,'ratio',round(float(ratio),3),'saved',flush=True)
        report={'sample':name,'model':'GPS-Gaussian official pretrained network', 'parameters':sum(p.numel() for p in model.parameters()),
            'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'resolution':list(arr.shape[:2]),
            'preprocess_seconds':preprocess,'network_and_gather_ms':model_ms,'render_ms':timings,
            'peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2,'gaussians':count,'preview_points':preview,
            'views':args.views,'quality_metrics':None,'notes':'No target-view ground truth in this minimal extraction. First-run timings, not paper benchmark. Camera interpolation between two inputs; not complete 360 reconstruction.'}
        (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8');reports.append(report)
        del g,data
        print(json.dumps(report),flush=True)
    dest=ROOT/'demo/real_results/manifest.json';dest.write_text(json.dumps(reports,indent=2),encoding='utf-8')

if __name__=='__main__': main()
