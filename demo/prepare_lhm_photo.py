"""Local foreground extraction and portrait padding for LHM++ reference images."""
from pathlib import Path
import sys,json,time
import numpy as np
from PIL import Image,ImageOps
started=time.time()
img=ImageOps.exif_transpose(Image.open(sys.argv[1])).convert('RGBA')
original_size=img.size
img.thumbnail((2048,2048))
method='provided alpha'
if img.getextrema()[3][0]==255:
    import rembg
    print('STAGE: local foreground extraction',flush=True)
    # Only use a verified, installed full-size model; never silently download
    # weights during a generation job. Keep the working lightweight fallback.
    model_home=Path(__import__('os').environ.get('U2NET_HOME',str(Path.home()/'.u2net')))
    method='u2net' if (model_home/'u2net.onnx').exists() else 'u2netp'
    print('STAGE: foreground model '+method,flush=True)
    img=rembg.remove(img,session=rembg.new_session(method,providers=['CPUExecutionProvider']))
a=np.asarray(img)[...,3];ys,xs=np.where(a>100)
if len(xs)<100:raise ValueError('Foreground not found')
# Preserve soft hair/clothing boundaries around the confident foreground.
soft_y,soft_x=np.where(a>16)
box=(max(0,int(soft_x.min())-2),max(0,int(soft_y.min())-2),min(img.width,int(soft_x.max())+3),min(img.height,int(soft_y.max())+3))
crop=img.crop(box)
crop_size=crop.size
# Fit in both directions, including upscaling small uploads. thumbnail() only
# downsizes and left small subjects surrounded by excessive white padding.
scale=min(454/crop.width,756/crop.height)
crop=crop.resize((max(1,int(crop.width*scale)),max(1,int(crop.height*scale))),Image.Resampling.LANCZOS)
canvas=Image.new('RGBA',(504,840),'white');canvas.alpha_composite(crop,((504-crop.width)//2,(840-crop.height)//2))
target=Path(sys.argv[2]);target.parent.mkdir(parents=True,exist_ok=True);canvas.convert('RGB').save(target)
(target.with_suffix('.preprocess.json')).write_text(json.dumps({'foreground_model':method,'original_size':original_size,'network_input_size':canvas.size,'crop_box':box,'foreground_crop_size':crop_size,'fitted_crop_size':crop.size,'fit_scale':scale,'resampling':'LANCZOS; fit includes upscaling','seconds':round(time.time()-started,2)},ensure_ascii=False,indent=2),encoding='utf8')
print('STAGE: LHM reference photo ready',flush=True)
