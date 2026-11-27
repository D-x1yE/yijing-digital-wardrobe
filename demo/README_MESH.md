# 照片生成完整 3D 模型

双击 `启动照片生成3D.cmd`，浏览器打开 http://127.0.0.1:8766/mesh.html 。
上传单人全身 JPG / PNG / WebP，点击生成，完成后拖动模型或点击背面；可下载彩色 GLB 和 PLY，导入 Blender 等软件。
照片与推理均在本机处理，不上传到在线模型服务。普通照片自动抠图；透明背景 PNG 更稳定。首次模型已下载到 vendor/TripoSR-weights。

这条流程真实使用 VAST-AI-Research 的 TripoSR 开源预训练模型，完整单图三维生成，不是程序化假人，也不是 GPS-Gaussian 的双视角可见表面重建。不是课程三篇论文的完整复现。单图不可见背面由模型推测，不保证符合真人；人脸、手指、遮挡和复杂衣服仍可能失真。多张照片融合尚未实现。

本机适配：GPU 神经场推理，scikit-image CPU marching cubes 提取网格，DINO 配置离线读取，保留最大连通几何组件，顶点颜色导出。不进行人工拉伸或补造背面。

验证样例：`sample_person.png` 来自 GPS-Gaussian 官方示例数据及其前景 mask。生成产物在 mesh_results/sample，mesh.json 记录实际统计。上传结果按随机任务 ID 保存在 mesh_results，可手动删除不需要的照片和模型。

官方代码：https://github.com/VAST-AI-Research/TripoSR
权重：stabilityai/TripoSR，下载采用 AI-ModelScope/TripoSR 镜像；model.ckpt SHA256 为 429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee 。
