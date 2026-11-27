"""Preset SMPL-X animation with official LHM++ skinning and gsplat rendering.
No synthetic image animation: each video frame comes from posed 3D Gaussians.
"""
from pathlib import Path
import json,time,os
import numpy as np
import torch
import imageio.v2 as imageio

PRESETS = {
    'Dance_I': {'label':'舞蹈','step':4,'fps':15,'frames':72},
    'TaiChi': {'label':'太极','step':2,'fps':15,'frames':72},
    'Jump': {'label':'跳跃','step':1,'fps':30,'frames':76},
}

def render_animation(model, outputs, scope, synthetic_motion, preset, out, started):
    if preset not in PRESETS: raise ValueError('Unknown motion preset')
    from gsplat.rendering import rasterization
    repo=Path(__file__).resolve().parent.parent/'vendor/LHM-plusplus-main'
    cfg=PRESETS[preset]
    paths=sorted((repo/'motion_video'/preset/'smplx_params').glob('*.json'))[::cfg['step']][:cfg['frames']]
    if len(paths)<2:raise RuntimeError('Motion assets are missing')
    gs_attrs,query,neutral_transform=outputs[:3]
    pred_shape=outputs[7] if len(outputs)==8 else None
    base_sp={k:v.to('cuda') for k,v in synthetic_motion['smplx_params'].items()}
    merged=type(model).smplx_params_with_pred_shape_betas(base_sp,pred_shape)
    betas=merged['betas'].float()
    frames_limit=int(os.environ.get('LHM_ANIMATION_FRAMES',len(paths)))
    paths=paths[:frames_limit]
    print('STAGE: animation '+preset+' frames='+str(len(paths)),flush=True)
    render_started=time.time()
    # Preserve source camera projection; use a bounded output size on an 8 GiB GPU.
    raw=json.loads(paths[0].read_text());w,h=raw['img_size_wh']
    ratio=min(512/h,512/w)
    width=max(16,int(round(w*ratio/16))*16);height=max(16,int(round(h*ratio/16))*16)
    writer=imageio.get_writer(str(out/'animation.mp4'),fps=cfg['fps'],codec='libx264',quality=8,
                            pixelformat='yuv420p',macro_block_size=16,ffmpeg_params=['-movflags','+faststart'])
    frame_diffs=[];prev=None;first_xyz=None;xyz_change=0.;alpha_counts=[]
    try:
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            for i,path in enumerate(paths):
                motion=scope['_build_motion_seq_from_pose_json'](str(path),{'render_size':420})
                sp=scope['build_animation_frame_smplx_params'](motion,neutral_transform,betas,torch.device('cuda'),torch.float32)
                single=model.renderer._get_single_batch_data(model.renderer.get_single_view_smpl_data(sp,0),0)
                posed,_=model.renderer.animate_gs_model(gs_attrs[0],query['neutral_coords'][0],single,
                                    debug=False,mesh_meta=query['mesh_meta'])
                if not posed:raise RuntimeError('Official skinning returned no posed Gaussians')
                gs=posed[0]
                for k in ['xyz','rotation','scaling','opacity','shs']:
                    if not torch.isfinite(getattr(gs,k)).all():raise RuntimeError('Nonfinite animated '+k)
                if first_xyz is None:first_xyz=gs.xyz.detach().clone()
                else:xyz_change=max(xyz_change,float((gs.xyz-first_xyz).abs().max()))
                K=motion['render_intrs'][0,0,:3,:3].to('cuda').float().clone()
                source_w,source_h=motion['smplx_params']['img_size_wh'][0,0].tolist()
                K[0]*=width/source_w;K[1]*=height/source_h
                colors=gs.shs.squeeze(1).float()
                rgb,alpha,_=rasterization(means=gs.xyz.float().contiguous(),quats=gs.rotation.float().contiguous(),
                    scales=gs.scaling.float().contiguous(),opacities=gs.opacity.squeeze(-1).float().contiguous(),
                    colors=colors.contiguous(),viewmats=torch.eye(4,device='cuda')[None],Ks=K[None],
                    width=width,height=height,near_plane=.01,far_plane=100.,eps2d=.3,render_mode='RGB',
                    backgrounds=torch.ones(1,3,device='cuda'),camera_model='pinhole')
                if not torch.isfinite(rgb).all():raise RuntimeError('Nonfinite rendered image')
                frame=(rgb[0].clamp(0,1)*255).byte().cpu().numpy()
                count=int((alpha[0,...,0]>.1).sum());alpha_counts.append(count)
                if count<100:raise RuntimeError('Animation frame is blank')
                if prev is not None:frame_diffs.append(float(np.abs(frame.astype(float)-prev).mean()))
                prev=frame.astype(float)
                writer.append_data(frame)
                if i==0:imageio.imwrite(out/'animation_poster.png',frame)
                if i==len(paths)//2:imageio.imwrite(out/'animation_middle.png',frame)
                if i==len(paths)-1:imageio.imwrite(out/'animation_last.png',frame)
                if i%12==0 or i==len(paths)-1:print(f'STAGE: rendered {i+1}/{len(paths)} animation frames',flush=True)
    finally:writer.close()
    if len(paths)<2 or xyz_change<.001:raise RuntimeError('No measurable skeletal animation')
    meta={'motion':preset,'motion_label':cfg['label'],'frames':len(paths),'fps':cfg['fps'],
          'duration_seconds':round(len(paths)/cfg['fps'],2),'width':width,'height':height,
          'renderer':'official SMPL-X Gaussian skinning + gsplat RGB (no neural refinement)',
          'seconds_animation':round(time.time()-render_started,2),'seconds_pipeline':round(time.time()-started,2),
          'gpu_peak_gib':round(torch.cuda.max_memory_allocated()/2**30,3),
          'max_gaussian_displacement':xyz_change,'mean_frame_difference':float(np.mean(frame_diffs)),
          'min_foreground_pixels':min(alpha_counts),'finite':True,'source_frames':[p.name for p in paths]}
    (out/'animation.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ANIMATION DONE: '+json.dumps(meta,ensure_ascii=False),flush=True)
