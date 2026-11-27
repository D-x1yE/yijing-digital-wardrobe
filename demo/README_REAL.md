# 真实人体三维重建与新视角生成演示

## 启动

双击 `启动真实人体演示.cmd`，保留终端窗口，浏览器进入 http://127.0.0.1:8765/real.html 。服务仅监听本机。若已有服务在运行，直接打开地址即可。

页面提供三个真实序列样例，每个样例两个已标定照片输入，17 个模型渲染视角、可旋转三维点预览和完整高斯模型导出。样例 0001—0003 来自同一公开序列，不代表三个不同人物。点击“重新运行 GPU 模型”会真正重新执行预处理、神经网络推理和 CUDA 渲染，状态与日志会更新。拖动视角滑块则展示本机已经生成的 PNG，不触发网络推理。

## 实现与来源

- 网络：GPS-Gaussian 官方代码 https://github.com/aipixel/GPS-Gaussian ，276 个参数项严格匹配，5,144,408 个参数。并非从零训练，也非三篇论文融合。
- 权重：gqy2468/GPS-Gaussian Hugging Face 镜像的 GPS-GS_stage2_final.pth。镜像文件与作者目录公布的文件名和长度相同，但尚未逐字节对比作者权重。不宣称完成来源哈希认证。
- 原始 checkpoint 包含优化器与调度器，较新 PyTorch 的 weights_only 无法解析其中的旧格式。已用限制全局对象类型的 unpickler 提取网络张量为 `vendor/gps_network_only.pth`，实际运行只以 weights_only=True 读取此文件。
- 数据：作者 README 链接的公共 OneDrive，使用 HTTP Range 从 real_data.zip 提取前 3 个样例的 0/1 视角图片、掩码与相机参数。抓取脚本 `fetch_minimal_data.py`。无需下载整个数据集。
- 光栅器：GauHuman 随附的修改版 diff-gaussian-rasterization；Windows CUDA 12.6 / MSVC 2022 编译，生成 `runtime/diff_gaussian_rasterization/_C.cp39-win_amd64.pyd`。该版本额外返回深度与 alpha，推理只取颜色输出。依赖原许可证，源码在 vendor/GauHuman-main/submodules/diff-gaussian-rasterization。
- RAFT 使用官方支持的 reg 相关性实现，不使用可选的 reg_cuda 扩展。
- 输出设为 1024×1024，关闭额外 2 倍输出尺寸，网络输入仍为官方 1024 设置。

## 重跑命令

```powershell
& 'E:/555/envs/sd_torch/python.exe' -u 'demo/run_real_model.py' --samples 3 --views 17
```

在项目根目录运行。脚本已定位项目路径，运行时会使用同目录 vendor 源码、权重、minimal_data 和 demo/runtime。当前 Python 环境在 E 盘，换电脑需要重新安装依赖、编译扩展并修改启动脚本 Python 路径。

## 实验记录

GPU：RTX 4060 Laptop 8GB；PyTorch 2.7.1+cu126。首轮验证的样例 0001 预测约 266,251 个高斯。网络推理及点元整理的冷启动约 2 秒，后续样例约数百毫秒。具体数值以每次运行的 metrics.json 为准。

计时使用 torch.cuda.synchronize；网络计时包含点元整理，不包含模型加载和预处理；渲染计时不包含 PNG 编码、磁盘写入和浏览器显示。不同样例存在冷启动、缓存差异，当前为功能验证记录，不是严谨重复测速。显存为 PyTorch max_memory_allocated，不是显卡总占用。

## 输出文件

`real_results/0001` 等目录包含输入照片、17 个 PNG、metrics.json、gaussians.npz、human_gaussians.ply 和 points.bin。

- NPZ：真实预测的 xyz、rgb、rot、scale、opacity。
- PLY：高斯中心、零阶球谐颜色、log scale、logit opacity、四元数，非闭合三角网格。仅存储零阶颜色，没有高阶 SH 字段；部分外部查看器需要适配其字段要求。
- points.bin：带 RGB 的抽样三维点预览，不是完整高斯渲染；用来观察真实空间结构。

## 演示范围

已实现：两张已标定人体照片 → 双目深度估计 → 三维高斯参数回归 → 新相机渲染 → 三维参数导出。

未实现：任意未标定手机照片上传、完整背面生成、360°高质量人体补全、闭合网格提取、SMPL 骨骼动画、服装生成与语言交互。没有目标新视角真值，未报告 PSNR/SSIM。

答辩时可表述：“我们跑通了 GPS-Gaussian 预训练基线，在本机 RTX 4060 上从公开双目人体照片估计三维高斯，生成相机之间的新视角；下一步扩展数据、评价与交互。”不能说已经复现 GauHuman、T3HG-Editor 或 Feature4X 全部算法。
