import importlib,json,sys
modules=['torch','torchvision','xformers','pytorch3d','torch_scatter','spconv.pytorch','pointops_cuda','diff_gaussian_rasterization','accelerate','transformers','diffusers','timm','smplx']
results={}
for name in modules:
    try:
        m=importlib.import_module(name);results[name]=getattr(m,'__version__','import OK')
    except Exception as e:results[name]='ERROR: '+repr(e)
import torch
results['cuda_available']=torch.cuda.is_available()
if torch.cuda.is_available():
    results['gpu']=torch.cuda.get_device_name(0)
    results['tensor_test']=(torch.ones(16,device='cuda')*2).sum().item()
print(json.dumps(results,indent=2))
