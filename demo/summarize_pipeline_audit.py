"""Build the local evidence report for the 2026-11-24 pipeline audit."""
from pathlib import Path
import json,hashlib,html
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from plyfile import PlyData

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evaluation_user_20261124'
report=json.loads((OUT/'evaluation.json').read_text(encoding='utf8'))
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
labels=['官方高斯 RGB','官方神经渲染','网页 PLY · 1×','网页 PLY · 2×降采样']
rows=[];comparisons=[];captures=[]
for angle in [0,90,180]:
 images=[]
 for filename in [f'official_gs_{angle}.png',f'official_neural_{angle}.png',f'webgl_{angle}_1x.png',f'webgl_{angle}_2x.png']:
  im=Image.open(OUT/filename).convert('RGB')
  images.append(im.resize((420,700),Image.Resampling.LANCZOS) if im.size!=(420,700) else im)
 row=Image.new('RGB',(1680,755),'#edf0f4');draw=ImageDraw.Draw(row)
 for i,im in enumerate(images):row.paste(im,(420*i,55));draw.text((420*i+12,12),labels[i],font=font,fill='#152638')
 row.save(OUT/f'comparison_{angle}.png');rows.append(row)
 alpha=np.asarray(Image.open(OUT/f'alpha_{angle}.png'))[...,0]>128
 yy,xx=np.where(alpha);top=int(yy.min());bottom=int(yy.max());cx=int((xx.min()+xx.max())/2)
 box=(max(0,cx-80),max(0,top-10),min(420,cx+80),min(700,top+int((bottom-top)*.34)))
 crop=Image.new('RGB',(1280,510),'#edf0f4');cd=ImageDraw.Draw(crop)
 for i,im in enumerate(images):
  part=im.crop(box);factor=min(310/part.width,450/part.height)
  part=part.resize((int(part.width*factor),int(part.height*factor)),Image.Resampling.NEAREST)
  crop.paste(part,(i*320,55));cd.text((i*320+6,12),labels[i],font=small,fill='#152638')
 crop.save(OUT/f'crop_{angle}.png')
 a=np.asarray(images[0],dtype=float)
 for i in [1,2,3]:
  b=np.asarray(images[i],dtype=float);mask=(a.min(2)<245)|(b.min(2)<245)
  comparisons.append({'angle':angle,'other':labels[i],'foreground_union_mae_0_255':float(np.abs(a-b)[mask].mean())})
 for ratio in [1,2]:
  capture=json.loads((OUT/f'webgl_{angle}_{ratio}x.json').read_text(encoding='utf8'))
  assert capture['ply_sha256']==report['ply_sha256']
  assert capture['splatRenderCount']==capture['splatCount'] and 0<capture['splatCount']<=report['gaussians']
  assert capture['fadeInComplete']==1
  assert (capture['width'],capture['height'])==(420*ratio,700*ratio)
  view=next(v for v in report['views'] if v['angle']==angle)
  assert np.allclose(capture['eye'],view['eye'],atol=1e-6)
  captures.append(capture)
report['pairwise_differences']=comparisons;report['browser_captures']=captures

ply=PlyData.read(OUT/'gaussian.ply')['vertex'].data
rest=[n for n in ply.dtype.names if n.startswith('f_rest_')]
report['export_audit']={'count':len(ply),'properties':list(ply.dtype.names),
 'all_properties_finite':all(bool(np.isfinite(ply[n]).all()) for n in ply.dtype.names),
 'rest_sh_property_count':len(rest),
 'max_abs_rest_sh':max((float(np.abs(ply[n]).max()) for n in rest),default=0.),
 'actual_ply_sha256':hashlib.sha256((OUT/'gaussian.ply').read_bytes()).hexdigest()}
assert report['export_audit']['all_properties_finite']
assert report['export_audit']['actual_ply_sha256']==report['ply_sha256']

def foreground_size(path):
 a=np.asarray(Image.open(path).convert('RGB'));yy,xx=np.where(a.min(2)<240)
 return {'image_size':[a.shape[1],a.shape[0]],'nonwhite_bbox':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],'nonwhite_height':int(yy.max()-yy.min()+1)}
ablation=json.loads((OUT/'small_ablation/ablation.json').read_text(encoding='utf8'))
for case in ablation['cases']:case['input_extent']=foreground_size(OUT/'small_ablation'/case['name']/'input.png')
report['preprocessing_ablation']=ablation
(OUT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')

for angle in [0,90,180]:
 sheet=Image.new('RGB',(840,755),'#edf0f4');d=ImageDraw.Draw(sheet)
 for i,(name,label) in enumerate([('legacy','修复前：小图保持原尺寸'),('fixed','修复后：等比例放大到输入范围')]):
  sheet.paste(Image.open(OUT/'small_ablation'/name/f'official_gs_{angle}.png'),(i*420,55));d.text((i*420+8,12),label,font=small,fill='#152638')
 sheet.save(OUT/f'preprocess_result_{angle}.png')
sheet=Image.new('RGB',(1008,900),'#edf0f4');d=ImageDraw.Draw(sheet)
for i,(name,label) in enumerate([('legacy','修复前 · 504×840'),('fixed','修复后 · 504×840')]):
 sheet.paste(Image.open(OUT/'small_ablation'/name/'input.png'),(i*504,60));d.text((i*504+15,15),label,font=font,fill='#152638')
sheet.save(OUT/'preprocess_inputs.png')

maes=[x['foreground_union_mae_0_255'] for x in comparisons if x['other']==labels[2]]
before=ablation['cases'][0]['input_extent']['nonwhite_height'];after=ablation['cases'][1]['input_extent']['nonwhite_height']
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LHM++ 项目流程排查</title><style>body{max-width:1440px;margin:32px auto;padding:0 24px;background:#f3f6f9;color:#203044;font:16px/1.85 system-ui}section{background:white;margin:20px 0;padding:24px;border-radius:12px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}a{color:#086a8e}code{overflow-wrap:anywhere}h1{font-size:30px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style><h1>LHM++ 项目流程排查</h1><p>2026-11-24 · RTX 4060 Laptop 8GB · 本地推理与浏览器实测</p>
<section><h2>结论</h2><p><b>实际上传样本的模糊在官方高斯渲染中已经出现。网页不是该样本丢失脸部与 T 恤文字细节的主要环节。</b>神经渲染改变和平滑纹理，但仍不能还原原有文字。本结论基于一张历史上传照片与上一轮一张官方照片，不构成数据集准确率评估。</p>
<p>另有一个已确认的工程问题：预处理使用 thumbnail()，小图只缩小不放大，导致人物占比偏小。现已修复为双向等比例适配；网页渐显和截图完整排序修复也已合入主查看器。</p>
<p><b>需要区分模型版本：</b>项目实际加载 LHMPP-700M-SMPLX-FREE。本地官方 app.py 默认 LHMPP-700M-PixelShuffle，但这份权重尚未安装、尚未对照实测。不能将本轮 SMPLX-FREE 的结果推广为 PixelShuffle 的效果，也不能声称更换后必然改善。</p></section>'''
page+=f'<section><h2>同照片、同权重、同一次预测</h2><p>输入为历史任务 <code>1df7ab658db24e3dac4d22b85fe40f99/references/00.png</code>，继续使用网页已实际送入模型的参考图，排除额外预处理变量。原上传 1024×1024，经原流程裁剪为 504×840，故此样本不受“小图不放大”问题影响。种子 42；BF16 骨干、FP32 几何与神经头，xFormers 注意力适配；非官方全精度逐字复现。</p><p>每次预测 160,000 个高斯；网页沿用 alpha 阈值 1，实际载入并排序 {captures[0]["splatCount"]:,} 点。全部属性有限，PLY 哈希一致，相机、白底与分辨率对齐。网页 1× 与官方高斯的前景并集平均绝对差 {min(maes):.2f}–{max(maes):.2f}/255，仅说明两个渲染器的接近程度，不是重建准确率。</p><p>这份权重配置 gs_use_rgb=true，官方导出 PLY 不含高阶球谐属性，颜色只含 DC/RGB 转换结果；因此这个输出无需网页计算高阶球谐。未发现普通点云冒充高斯、NaN 或导出属性缺失。</p></section>'
for angle,name in [(0,'正面'),(90,'侧面'),(180,'背面')]:
 page+=f'<section><h2>{name} · {angle}°</h2><a href="comparison_{angle}.png"><img src="comparison_{angle}.png" alt="四路同相机对照"></a><h3>原始像素局部放大</h3><img src="crop_{angle}.png" alt="同位置头部与衣服放大"></section>'
page+=f'<section><h2>小图预处理修复</h2><p>历史上传图为 197×405，固定输出仍为 504×840。修复前非白前景高 {before} 像素，修复后 {after} 像素。修改仅做等比例插值适配，不是超分辨率，也不会恢复输入本来没有的纹理。</p><img src="preprocess_inputs.png" alt="同一张照片修复前后模型输入"><p>同一份模型加载、两次推理前均重置随机种子、同一套输出相机。旧输入直接取自已保存的历史参考图，新输入使用当前已安装的完整 U²-Net 重新预处理；因此不是只替换一行缩放参数的严格隔离实验，抠图实现或权重状态也可能形成混杂。输出外观有变化，但清晰度没有质变，不能据此量化改善比例。</p><img src="preprocess_result_0.png" alt="修复前后正面重建"><img src="preprocess_result_90.png" alt="修复前后侧面重建"></section>'
page+='''<section><h2>已核对与后续优先级</h2><table><tr><th>环节</th><th>结果</th></tr>
<tr><td>上传</td><td>前端 FileReader 传原文件；后端 EXIF 转正后存 PNG，没有前端有损压缩或缩略图替代上传。</td></tr>
<tr><td>预处理</td><td>确认小图不放大问题，已修复。原有透明通道会保留；不透明图使用完整 U²-Net，缺失时回退轻量版。半身图、遮挡、低清晰度仍需使用约束。</td></tr>
<tr><td>推理</td><td>同一 SMPLX-FREE 权重严格加载；当前预测张量已经包含文字与脸部的细节损失。模型与输入质量仍是后续检查重点。</td></tr>
<tr><td>导出</td><td>沿用官方规范姿态和 save_ply；16 万高斯、属性有限。普通 PLY 不携带神经渲染头使用的隐藏特征。</td></tr>
<tr><td>网页</td><td>主查看器关闭渐显；导出 PNG 前等待完整排序；保留原 1×/1.5×/2× 显示选项。</td></tr>
<tr><td>下一优先级</td><td>固定一组清晰、完整全身、同服装的正侧背照片，再对照 PixelShuffle。先实测再判断换模型；不要通过照片后期锐化宣称三维重建精度提升。</td></tr></table></section>'''
page+='<section><h2>证据与限制</h2><p>规范姿态与输入照片姿态不同，没有匹配的真实新视角真值，因此不计算“重建准确率”的 PSNR/SSIM。模型随机采样及 GPU 运算可能存在非确定性；本次小图实验只作质性检查。所有推理与渲染在本机完成。</p><p><a href="metrics.json">参数、哈希与网页完整性验证</a> · <a href="small_ablation/ablation.json">小图推理记录</a> · <a href="input.png">实际送入模型的照片</a> · <a href="webgl.html">交互对照</a> · <a href="../evaluation_render_20261123/index.html">上一轮官方样本</a></p></section></html>'
(OUT/'index.html').write_text(page,encoding='utf8')
print(json.dumps({'webgl_mae':maes,'small_foreground_heights':[before,after],'export':report['export_audit']},ensure_ascii=False))
