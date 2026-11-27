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
weights=REPO/'pretrained_models/Damo_XR_Lab/LHMPP-700M-SMPLX-FREE'
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
motion=scope['_build_synthetic_motion_seq']({'render_size':420})
sources=sorted(source.glob('*.png')) if source.is_dir() else [source]
if not 1<=len(sources)<=4:raise ValueError('Expected 1-4 reference images')
images=[Image.open(p).convert('RGB') for p in sources]
images[0].save(out/'input.png')
tensor=torch.stack([torch.from_numpy(np.asarray(img).copy()/255.0).permute(2,0,1).float() for img in images]).to('cuda')
print('STAGE: reference images '+str(len(images)),flush=True)
print('STAGE: photo to Gaussian prediction',flush=True)
# Optional animation captures the official reconstruction outputs for skeletal warping.
animation_motion = sys.argv[3] if len(sys.argv)>3 else None
captured = {}
if animation_motion:
    original_infer = model.infer_single_view
    def capture_infer(*args, **kwargs):
        result = original_infer(*args, **kwargs)
        captured['outputs'] = result
        return result
    model.infer_single_view = capture_infer
infer_start=time.time()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    scope['run_tpose_export'](model,tensor,motion,'cuda',str(out/'gaussian.ply'))
torch.cuda.synchronize()
from plyfile import PlyData
ply=PlyData.read(out/'gaussian.ply');v=ply['vertex'].data
metrics={'model':'LHMPP-700M-SMPLX-FREE','input':str(source),'gaussians':len(v),'seconds_total':round(time.time()-started,2),'seconds_infer_export':round(time.time()-infer_start,2),'gpu_peak_gb':round(torch.cuda.max_memory_allocated()/2**30,3),'strict_checkpoint':True,'attention_backend':'xformers block diagonal varlen','precision':'BF16 neural / FP32 sparse convolution and body geometry','finite':all(np.isfinite(v[n]).all() for n in v.dtype.names)}
(out/'verification.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
metrics['input_count']=len(sources)
(out/'verification.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
if not metrics['finite']:raise RuntimeError('Output contains NaN/Inf: validation failed')
print('DONE: '+json.dumps(metrics),flush=True)

if animation_motion:
    from lhm_animation import render_animation
    render_animation(model, captured['outputs'], scope, motion, animation_motion, out, started)
