"""Controlled legacy-vs-fit preprocessing ablation, using one loaded checkpoint.

Usage: python ablate_lhm_preprocessing.py manifest.json output_dir
Manifest is a list of {name, input}; paths may be relative to the manifest.
"""
from pathlib import Path
import sys,json,hashlib,time,gc

# Share the already-validated model and numerical adapters with the render test.
base=Path(__file__).with_name('evaluate_lhm_render.py')
exec(compile(base.read_text(encoding='utf8').split('import hashlib, traceback')[0],str(base),'exec'))
from core.structures.camera import Camera
manifest=source
cases=json.loads(manifest.read_text(encoding='utf8'))
motion=scope['_build_synthetic_motion_seq']({'render_size':420})
sp={k:v.to('cuda') for k,v in motion['smplx_params'].items()}
report={'seed':42,'model':'LHMPP-700M-SMPLX-FREE','cases':[],
        'precision':'BF16 backbone / FP32 Gaussian geometry',
        'note':'No matched ground truth; compare appearance only. Same camera for all cases.'}
camera_views=None
with torch.inference_mode():
 for case in cases:
  torch.manual_seed(42);np.random.seed(42)
  path=Path(case['input']);path=path if path.is_absolute() else manifest.parent/path
  dest=out/case['name'];dest.mkdir(exist_ok=True)
  image=Image.open(path).convert('RGB');image.save(dest/'input.png')
  tensor=torch.from_numpy(np.asarray(image).copy()/255.).permute(2,0,1).float()[None,None].to('cuda')
  ts=time.perf_counter()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   result=model.infer_single_view(tensor,None,None,render_c2ws=motion['render_c2ws'].to('cuda'),render_intrs=motion['render_intrs'].to('cuda'),render_bg_colors=motion['render_bg_colors'].to('cuda'),smplx_params=sp,ref_imgs_bool=torch.ones(1,1,device='cuda',dtype=torch.bool),return_pred_shape=True)
  gs_list,query,neutral,features,*extras=result
  pred_shape=result[-1]
  merged=type(model).smplx_params_with_pred_shape_betas(sp,pred_shape)
  canosp=scope['build_tpose_smplx_params'](motion,neutral,merged['betas'],'cuda',torch.float32)
  gs=model.inference_gs(gs_list,query,canosp,motion['render_c2ws'].to('cuda'),motion['render_intrs'].to('cuda'),motion['render_bg_colors'].to('cuda'),features)
  gs.save_ply(str(dest/'gaussian.ply'))
  torch.cuda.synchronize()
  seconds=time.perf_counter()-ts
  xyz=gs.xyz.detach().float().cpu().numpy()
  if not np.isfinite(xyz).all():raise RuntimeError('Nonfinite Gaussian positions')
  h,w,focal=700,420,700.
  if camera_views is None:
   lo,hi=np.quantile(xyz,[.01,.99],axis=0);center=(lo+hi)/2
   distance=float(max((hi[1]-lo[1])*focal/h,(hi[0]-lo[0])*focal/w)*1.35)
   camera_views=[]
   for angle in [0,90,180]:
    a=math.radians(angle);eye=center+np.array([math.sin(a)*distance,0,math.cos(a)*distance]);forward=center-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,np.array([0.,1.,0.]));right/=np.linalg.norm(right);down=np.cross(forward,right)
    mat=np.eye(4,dtype=np.float32);mat[:3,:3]=np.stack([right,down,forward],axis=1);mat[:3,3]=eye
    camera_views.append({'angle':angle,'c2w_opencv':mat.tolist()})
  K=torch.eye(4,device='cuda');K[0,0]=K[1,1]=focal;K[0,2]=w/2;K[1,2]=h/2
  model.renderer.device=torch.device('cuda')
  for view in camera_views:
   camera=Camera.from_c2w(torch.tensor(view['c2w_opencv'],device='cuda'),K,h,w)
   rendered=model.renderer.forward_single_view(gs,camera,torch.ones(3,device='cuda'),features=None,patch_size=model.neural_renderer_patch_size)
   arr=rendered['comp_rgb'].detach().float().cpu().numpy()
   if not np.isfinite(arr).all():raise RuntimeError('Nonfinite render')
   Image.fromarray((arr.clip(0,1)*255).round().astype('uint8')).save(dest/f"official_gs_{view['angle']}.png")
  report['cases'].append({'name':case['name'],'input':str(path),'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'gaussians':len(xyz),'seconds_infer_export':seconds})
  report['views']=camera_views
  (out/'ablation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
  del result,tensor,gs_list,query,neutral,features,extras,gs,rendered
  gc.collect();torch.cuda.empty_cache()
  print('CASE_DONE '+case['name'],flush=True)
print('ABLATION_DONE',flush=True)
