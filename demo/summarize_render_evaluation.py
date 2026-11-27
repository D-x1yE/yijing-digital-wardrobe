"""Create original-resolution comparisons; no ground-truth accuracy claim."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent/'evaluation_render_20261123'
report=json.loads((ROOT/'evaluation.json').read_text(encoding='utf8'))
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
rows=[];measurements=[]
labels=['官方高斯 RGB','官方神经渲染','网页 PLY · 1×','网页 PLY · 2×降采样']
for angle in [0,90,180]:
    files=[f'official_gs_{angle}.png',f'official_neural_{angle}.png',f'webgl_{angle}_1x.png',f'webgl_{angle}_2x.png']
    imgs=[Image.open(ROOT/f).convert('RGB') for f in files]
    normalized=[im.resize((420,700),Image.Resampling.LANCZOS) if im.size!=(420,700) else im for im in imgs]
    row=Image.new('RGB',(1680,755),'#edf0f4');draw=ImageDraw.Draw(row)
    for i,im in enumerate(normalized):
        row.paste(im,(420*i,55));draw.text((420*i+12,12),labels[i],font=font,fill='#152638')
    row.save(ROOT/f'comparison_{angle}.png');rows.append(row)
    # Same crop in each image; based only on fixed GS foreground alpha.
    alpha=np.asarray(Image.open(ROOT/f'alpha_{angle}.png'))[...,0]>128
    yy,xx=np.where(alpha)
    if len(xx):
        top=int(yy.min());bottom=int(yy.max());cx=int((xx.min()+xx.max())/2)
        box=(max(0,cx-80),max(0,top-10),min(420,cx+80),min(700,top+int((bottom-top)*.34)))
        crop=Image.new('RGB',(1280,510),'#edf0f4');cd=ImageDraw.Draw(crop)
        for i,im in enumerate(normalized):
            part=im.crop(box);part.thumbnail((310,450),Image.Resampling.NEAREST)
            # Nearest-neighbor enlargement deliberately reveals original pixels.
            factor=min(310/(box[2]-box[0]),450/(box[3]-box[1]))
            part=im.crop(box).resize((int((box[2]-box[0])*factor),int((box[3]-box[1])*factor)),Image.Resampling.NEAREST)
            crop.paste(part,(i*320,55));cd.text((i*320+6,12),labels[i],font=small,fill='#152638')
        crop.save(ROOT/f'crop_{angle}.png')
    a=np.asarray(normalized[0],dtype=float)
    for i in [1,2,3]:
        b=np.asarray(normalized[i],dtype=float)
        mask=(np.min(a,axis=2)<245)|(np.min(b,axis=2)<245)
        mae=float(np.abs(a-b)[mask].mean()) if mask.any() else 0
        measurements.append({'angle':angle,'comparison':labels[0]+' vs '+labels[i],'foreground_union_mae_0_255':mae})
report['browser_captures']=[json.loads((ROOT/f'webgl_{a}_{r}x.json').read_text(encoding='utf8')) for a in [0,90,180] for r in [1,2]]
for capture in report['browser_captures']:
    assert capture['ply_sha256']==report['ply_sha256']
    assert capture['splatRenderCount']==capture['splatCount']==157626
    assert capture['fadeInComplete']==1
    assert capture['width']==420*capture['ratio'] and capture['height']==700*capture['ratio']
report['pairwise_differences']=measurements
(ROOT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
summary=Image.new('RGB',(1680,755*3),'white')
for i,row in enumerate(rows):summary.paste(row,(0,i*755))
summary.save(ROOT/'comparison_all.png')
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LHM++ 三路渲染评估</title><style>body{max-width:1500px;margin:35px auto;padding:0 24px;font:16px/1.8 system-ui;color:#203044;background:#f5f7fa}img{width:100%;background:white;border-radius:10px}section{background:white;padding:22px;margin:20px 0;border-radius:12px}h1{font-size:30px}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}code{word-break:break-all}a{color:#146b8f}</style><h1>LHM++ 同输入、同权重、同高斯渲染对照</h1><p>2026-11-23 · RTX 4060 Laptop 8GB · LHMPP-700M-SMPLX-FREE</p><section><h2>对照条件</h2><p>一张官方全身人物照片，同一次神经网络预测的 160,000 个高斯。规范姿态、白背景、420×700 像素；正、侧、背三套相机参数共享。网页 2× 以 840×1400 渲染，展示对比时降采样到 420×700。</p><p>CUDA 使用官方高斯光栅器与神经渲染头，网页采用现有 GaussianSplats3D 参数。CUDA 正文图和神经输出没有套用掩码去除异常像素。神经头采用 FP32，骨干沿用已验证的混合精度和 xFormers 适配；不是官方全精度完整流水线的逐字复现。</p><p><b>没有匹配相机和姿态的真实照片作为真值，因此本次只能归因渲染分支的观感差异，不能报告重建准确率提升或宣称泛化表现。</b></p></section>'''
html=html.replace('2026-11-23 ·','2026-11-24 ·')
html+='<section><h2>本轮结论</h2><p><b>在这一张官方样本上，网页 PLY 与官方纯高斯非常接近；神经渲染主要平滑纹理和边缘，没有呈现清晰度的质变。</b>头发、眼镜和衣服褶皱仍有有限的细节。网页提高到 2× 可改善部分边缘观感，但不能补出重建时缺失的信息。</p><p>评估页已关闭渐显动画并等待完整高斯排序，排除了早期截图腿脚缺失的采集问题。此改动仅在评估页，尚未合入项目主网页。PLY 有 160,000 点；网页沿用 alpha 阈值 1，实际载入并排序 157,626 点。不同光栅内核与透明度阈值使两路不可能逐像素一致。</p><p>三视角网页 1× 与 CUDA 的前景并集平均绝对差约 2.47–2.93 / 255。这只是两种输出的接近程度，不是对真实人体的准确率，不能把数值越小解释为模型更好。</p><p>当前应继续检查项目实际输入的预处理、权重、导出及相机分辨率设置，再用多张自己的照片复测。单凭这一张官方样本，无法解释此前所有演示差距，也不能证明模型在用户照片上的稳定性。</p></section>'
for angle,name in [(0,'正面'),(90,'侧面'),(180,'背面')]:
    html+=f'<section><h2>{name} · {angle}°</h2><a href="comparison_{angle}.png"><img src="comparison_{angle}.png"></a><h3>头部与上身同位置放大（最近邻显示原始像素）</h3><img src="crop_{angle}.png"></section>'
html+='<section><h2>运行数据</h2><table><tr><th>阶段</th><th>秒</th><th>PyTorch 峰值已分配显存 GB</th></tr>'
for key,value in report.items():
    if isinstance(value,dict) and 'seconds' in value:html+=f'<tr><td>{key}</td><td>{value["seconds"]:.3f}</td><td>{value["peak_allocated_gb"]:.3f}</td></tr>'
html+='</table><p>阶段峰值包含仍驻留的模型或特征；不是该阶段单独新增显存。第一次调用可能包含 CUDA 编译/初始化，不能据此宣传稳态延迟。</p><p><a href="metrics.json">完整参数、哈希和差异数据</a> · <a href="input.png">固定输入</a> · <a href="webgl.html">交互检查网页高斯</a></p></section></html>'
(ROOT/'index.html').write_text(html,encoding='utf8')
print(json.dumps(measurements,ensure_ascii=False,indent=2))
