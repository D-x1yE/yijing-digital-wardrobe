"""Local LHM++ single-image GS validation using official export functions.

Offline DINO initialization is followed by strict full-checkpoint loading.
Point-transformer varlen attention uses installed xFormers block-diagonal attention
instead of a separately compiled flash-attn wheel; no learned layers are changed.
"""
from __future__ import annotations
import os,sys,json,time,math,ast,types,gc,argparse
from pathlib import Path
from typing import Any
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent/'vendor/LHM-plusplus-main'
source=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=True)
os.chdir(REPO);sys.path.insert(0,str(REPO))
os.environ['TORCHDYNAMO_DISABLE']='1'
os.environ['CUDA_HOME']='/opt/lhm-cuda'
os.environ['PATH']='/opt/lhm-cuda/bin:/opt/lhm-venv/bin:'+os.environ.get('PATH','')
os.environ['MAX_JOBS']='2'
os.environ['TORCH_CUDA_ARCH_LIST']='8.9'
import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator
from safetensors.torch import load_file
started=time.time();torch.set_num_threads(4);torch._dynamo.config.disable=True
Accelerator()
print('STAGE: CUDA validation',torch.__version__,torch.cuda.get_device_name(0),flush=True)
import xformers.ops as xops
flash=types.ModuleType('flash_attn')
def varlen(qkv,cu_seqlens,max_seqlen,dropout_p=0,softmax_scale=None,**kw):
    if not torch.isfinite(qkv).all():raise RuntimeError('Nonfinite point attention QKV')
    lens=(cu_seqlens[1:]-cu_seqlens[:-1]).tolist()
    bias=xops.fmha.BlockDiagonalMask.from_seqlens(lens)
    q,k,v=qkv.unbind(1)
    return xops.memory_efficient_attention(q[None],k[None],v[None],attn_bias=bias,p=dropout_p,scale=softmax_scale)[0]
flash.flash_attn_varlen_qkvpacked_func=varlen
sys.modules['flash_attn']=flash
print('STAGE: import model',flush=True)
from core.models.encoders.dinov2_wrapper import Dinov2Wrapper
original=Dinov2Wrapper._build_dinov2
Dinov2Wrapper._build_dinov2=staticmethod(lambda name,modulation_dim=None,**kw:original(name,modulation_dim,pretrained=False))
from core.models import model_dict
model_name=os.environ.get('LHM_EVAL_MODEL','LHMPP-700M-SMPLX-FREE')
if model_name not in ['LHMPP-700M-SMPLX-FREE','LHMPP-700M-PixelShuffle']:raise ValueError('Unsupported evaluation model')
weights=REPO/'pretrained_models/Damo_XR_Lab'/model_name
cfg=json.loads((weights/'config.json').read_text())
print('STAGE: instantiate network',flush=True)
model=model_dict['human_lrm_a4o'](**cfg)
model.eval()
print('STAGE: strict checkpoint load',flush=True)
state=load_file(str(weights/'model.safetensors'),device='cpu')
if hasattr(model,'id_face_net'):
    # Training-only identity network is excluded from the main checkpoint;
    # its constructor has strictly loaded the separately verified ArcFace asset.
    for k,v in model.id_face_net.state_dict().items():state['id_face_net.'+k]=v
print(model.load_state_dict(state,strict=True),flush=True);del state;gc.collect()
print('STAGE: memory optimization (BF16 neural layers, FP32 sparse convolution and body geometry)',flush=True)
for name,child in model.named_children():
    if name not in ['renderer','id_face_net']:child.bfloat16()
# These modules are used for training losses / RGB refinement, not PLY export.
if hasattr(model,'id_face_net'):model.id_face_net.to('cpu')
if model.neural_renderer is not None:model.neural_renderer.to('cpu')
print('STAGE: transfer to GPU',flush=True)
for name,child in model.named_children():
    if name not in ['id_face_net','neural_renderer']:child.to('cuda')
import spconv.pytorch as spconv
from spconv.pytorch.conv import SparseConvolution
def sparse_dtype(mod,args):
    data=args[0]
    if data.features.dtype!=mod.weight.dtype:
        return (data.replace_feature(data.features.to(mod.weight.dtype)),)+args[1:]
for mod in model.modules():
    if isinstance(mod,SparseConvolution):
        mod.float();mod.register_forward_pre_hook(sparse_dtype)
    elif isinstance(mod,torch.nn.LayerNorm) and mod.weight is not None:
        mod.register_forward_pre_hook(lambda m,a:(a[0].to(m.weight.dtype),)+a[1:])
for name in ['encoder','transformer','shape_head']:
    module=getattr(model,name,None)
    if module is not None:
        module.register_forward_pre_hook(lambda mod,args,n=name:print('FORWARD: '+n,flush=True))
        def finish(mod,args,result,n=name):
            def check(value):
                if isinstance(value,torch.Tensor) and value.is_floating_point():
                    if not torch.isfinite(value).all():raise RuntimeError('Nonfinite output in '+n)
                elif isinstance(value,(list,tuple)):
                    for v in value:check(v)
                elif isinstance(value,dict):
                    for v in value.values():check(v)
            check(result);print('FINISHED: '+n,flush=True)
        module.register_forward_hook(finish)
original_query=model.renderer.get_query_points
def fp32_tree(value):
    if isinstance(value,torch.Tensor) and value.is_floating_point():return value.float()
    if __import__('dataclasses').is_dataclass(value):
        return __import__('dataclasses').replace(value,**{k:fp32_tree(v) for k,v in value.items()})
    if isinstance(value,dict):return {k:fp32_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [fp32_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(fp32_tree(v) for v in value)
    return value
def logged_query(*args,**kwargs):
    print('FORWARD: body query points',flush=True)
    with torch.autocast('cuda',enabled=False):result=original_query(*fp32_tree(args),**fp32_tree(kwargs))
    print('FINISHED: body query points',flush=True)
    return result
model.renderer.get_query_points=logged_query
def verify_values(value,label):
    if isinstance(value,torch.Tensor) and value.is_floating_point():
        if not torch.isfinite(value).all():raise RuntimeError('Nonfinite tensor: '+label)
    elif isinstance(value,dict):
        for k,v in value.items():verify_values(v,label+'.'+str(k))
    elif isinstance(value,(tuple,list)):
        for i,v in enumerate(value):verify_values(v,label+'.'+str(i))
original_forward_gs=model.renderer.forward_gs
def checked_forward_gs(*args,**kwargs):
    verify_values(kwargs,'forward_gs inputs')
    with torch.autocast('cuda',enabled=False):result=original_forward_gs(*fp32_tree(args),**fp32_tree(kwargs))
    for gs in result[0]:
        for k in ['offset_xyz','xyz','opacity','scaling','rotation','shs']:
            if hasattr(gs,k):verify_values(getattr(gs,k),'forward_gs output.'+k)
    verify_values(result[1:],'forward_gs body outputs')
    print('FINISHED: finite Gaussian attribute prediction',flush=True)
    return result
model.renderer.forward_gs=checked_forward_gs
original_cano=model.renderer.inference_cano_gs
def checked_cano(*args,**kwargs):
    with torch.autocast('cuda',enabled=False):result=original_cano(*fp32_tree(args),**fp32_tree(kwargs))
    for gs in result:
        for k in ['xyz','opacity','scaling','rotation','shs']:verify_values(getattr(gs,k),'canonical output.'+k)
    return result
model.renderer.inference_cano_gs=checked_cano
# Reuse the author's pure synthetic-camera and canonical-export functions without
# importing unused GUI, video, remote download and pose-estimation dependencies.
tree=ast.parse((REPO/'scripts/inference/to_gs_ply.py').read_text())
defs=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name not in ['main','setup_loaders_and_inputs']]
scope={'torch':torch,'np':np,'math':math,'os':os,'Any':Any,'Path':Path,'argparse':argparse,'contextmanager':__import__('contextlib').contextmanager,'json':json,'SimpleNamespace':SimpleNamespace}
exec(compile(ast.Module(body=defs,type_ignores=[]),str(REPO/'scripts/inference/to_gs_ply.py'),'exec'),scope)

import hashlib, traceback
from core.structures.camera import Camera
def digest_file(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
seed=42;torch.manual_seed(seed);np.random.seed(seed)
images=[Image.open(source).convert('RGB')]
images[0].save(out/'input.png')
tensor=torch.stack([torch.from_numpy(np.asarray(img).copy()/255.0).permute(2,0,1).float() for img in images]).to('cuda')
motion=scope['_build_synthetic_motion_seq']({'render_size':420})
sp={k:v.to('cuda') for k,v in motion['smplx_params'].items()}
report={'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'model':model_name,'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
'checkpoint_sha256':digest_file(weights/'model.safetensors'),
'seed':seed,'precision':'BF16 backbone / FP32 Gaussian geometry / FP32 neural head',
'height':700,'width':420,'views':[],'limitations':['Canonical pose, not a calibrated ground-truth photograph; no reconstruction-accuracy PSNR can be claimed.','CUDA GS vs WebGL have different splat kernels and thresholding.','Adapted xformers attention and mixed precision shared by all three outputs.']}
def timed(name,fn):
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();ts=time.perf_counter()
 val=fn();torch.cuda.synchronize();report[name]={'seconds':time.perf_counter()-ts,'peak_allocated_gb':torch.cuda.max_memory_allocated()/2**30};return val
with torch.inference_mode():
 def infer():
  with torch.autocast('cuda',dtype=torch.bfloat16):
   return model.infer_single_view(tensor[None],None,None,render_c2ws=motion['render_c2ws'].to('cuda'),render_intrs=motion['render_intrs'].to('cuda'),render_bg_colors=motion['render_bg_colors'].to('cuda'),smplx_params=sp,ref_imgs_bool=torch.ones(1,1,device='cuda',dtype=torch.bool),return_pred_shape=True)
 outputs=timed('predict',infer)
 gs_list,query,neutral,features,image_latents,motion_emb,pos_emb,pred_shape=outputs
 print('OUTPUTS',[(type(x).__name__,getattr(x,'shape',None)) for x in outputs],flush=True)
 merged=type(model).smplx_params_with_pred_shape_betas(sp,pred_shape)
 canosp=scope['build_tpose_smplx_params'](motion,neutral,merged['betas'],'cuda',torch.float32)
 gs=model.inference_gs(gs_list,query,canosp,motion['render_c2ws'].to('cuda'),motion['render_intrs'].to('cuda'),motion['render_bg_colors'].to('cuda'),features)
 gs.save_ply(str(out/'gaussian.ply'))
 if os.environ.get('LHM_EVAL_SAVE_STATE','1')=='1':
  torch.save({'gs_list':gs_list,'query':query,'canosp':canosp,'features':features,'image_latents':image_latents,'motion_emb':motion_emb,'pos_emb':pos_emb,'report':report},out/'render_state.pt')
 report['ply_sha256']=digest_file(out/'gaussian.ply')
 xyz=gs.xyz.detach().float().cpu().numpy();lo,hi=np.quantile(xyz,[.01,.99],axis=0);center=(lo+hi)/2
 print('BOUNDS',lo,hi,'FEATURES',features.shape,flush=True)
 for name in ['encoder','transformer','shape_head']:
  module=getattr(model,name,None)
  if module is not None:module.to('cpu')
 del outputs,tensor;gc.collect();torch.cuda.empty_cache()
 # Camera frame uses OpenCV +z forward, +y down; same converted for Three.js.
 h,w=700,420;focal=700.;distance=float(max((hi[1]-lo[1])*focal/h,(hi[0]-lo[0])*focal/w)*1.35)
 report['gaussians']=int(len(xyz))
 K=torch.eye(4,device='cuda');K[0,0]=K[1,1]=focal;K[0,2]=w/2;K[1,2]=h/2
 bg=torch.ones(3,device='cuda');model.renderer.device=torch.device('cuda')
 for angle in [0,90,180]:
  a=math.radians(angle);eye=center+np.array([math.sin(a)*distance,0,math.cos(a)*distance]);forward=center-eye;forward=forward/np.linalg.norm(forward)
  right=np.cross(forward,np.array([0.,1.,0.]));right/=np.linalg.norm(right);down=np.cross(forward,right)
  mat=np.eye(4,dtype=np.float32);mat[:3,:3]=np.stack([right,down,forward],axis=1);mat[:3,3]=eye
  reference=os.environ.get('LHM_EVAL_CAMERAS')
  if reference:
   fixed=next(v for v in json.loads(Path(reference).read_text())['views'] if v['angle']==angle)
   mat=np.array(fixed['c2w_opencv'],dtype=np.float32);eye=np.array(fixed['eye']);center=np.array(fixed['target'])
  camera=Camera.from_c2w(torch.tensor(mat,device='cuda'),K,h,w)
  view={'angle':angle,'eye':eye.tolist(),'target':center.tolist(),'camera_up':[0,1,0],'c2w_opencv':mat.tolist(),'focal':focal,'width':w,'height':h}
  rendered=timed(f'gs_{angle}',lambda:model.renderer.forward_single_view(gs,camera,bg,features=features[0].float(),patch_size=model.neural_renderer_patch_size))
  def png(x,path):
   arr=x.detach().float().cpu().numpy();Image.fromarray((arr.clip(0,1)*255).round().astype('uint8')).save(path)
  png(rendered['comp_rgb'],out/f'official_gs_{angle}.png')
  png(rendered['comp_mask'].expand(-1,-1,3),out/f'alpha_{angle}.png')
  try:
   if next(model.neural_renderer.parameters()).device.type!='cuda':model.neural_renderer.float().to('cuda')
   with torch.autocast('cuda',enabled=False):
    nr,nm=timed(f'neural_{angle}',lambda:model.neural_renderer(fp32_tree(image_latents),rendered['comp_features'].permute(2,0,1)[None,None],fp32_tree(motion_emb),h,w,pos_emb_list=fp32_tree(pos_emb)))
   print('NEURAL_SHAPES',nr.shape,getattr(nm,'shape',None),flush=True)
   if not torch.isfinite(nr).all():raise RuntimeError('Neural render contains NaN/Inf')
   rgb=nr[0,0];rgb=rgb.permute(1,2,0) if rgb.shape[0]==3 else rgb
   png(rgb,out/f'official_neural_{angle}.png');view['neural_status']='complete'
  except Exception as e:
   traceback.print_exc();view['neural_status']='failed';view['error']=str(e)
   model.neural_renderer.to('cpu');torch.cuda.empty_cache()
  report['views'].append(view)
  (out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
  del rendered;gc.collect();torch.cuda.empty_cache()
report['total_seconds']=time.time()-started
(out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print('EVALUATION_DONE',flush=True)
