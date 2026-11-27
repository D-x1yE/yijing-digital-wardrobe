# LHM++ 本机验证记录

## 多图扩展

工作台照片入口现支持同时选择 1–4 张照片，TripoSR 仍仅单图。后端逐张本地抠图并裁剪，按参考视角维度堆叠为同一次推理输入，不是分别生成模型后拼接。两张官方同人物不同视角已实测：160000 高斯、所有属性有限，总计 70.44 秒，推理导出 5.49 秒，峰值显存 3.786GB。此记录验证运行可行性，不代表已证明画质优于单图。四图效果和显存尚需独立验证。

目标：从单张公开全身样例照片预测标准姿态的 Gaussian PLY，确认完整输出后才视为跑通。

环境：WSL Ubuntu 22.04，独立 Python 3.10 环境 `/opt/lhm-venv`，CUDA 12.1 编译工具 `/opt/lhm-cuda`，RTX 4060 Laptop 8GB。现有 Windows TripoSR 环境没有替换。

代码：`vendor/LHM-plusplus-main`，官方仓库 https://github.com/aigc3d/LHM-plusplus 。模型为 `Damo_XR_Lab/LHMPP-700M-SMPLX-FREE`，下载资产列表见 `tmp/LHMPP-700M-SMPLX-FREE-files.json` 和 `tmp/LHMPP-Prior-files.json`。下载脚本使用 ModelScope 文件 API 并校验 SHA256。

已验证依赖和 CUDA 运算见 `lhm_environment_verified.json`；精确安装包列表见 `lhm_wsl_requirements.txt`。pointops 与高斯光栅化 CUDA 编译记录在 `lhm_extensions.log`。

运行命令（在项目根目录的 PowerShell）：

```
wsl -d Ubuntu-22.04 -- /opt/lhm-venv/bin/python -u demo/run_lhm_verify.py vendor/LHM-plusplus-main/assets/example_multi_images/00000_yuliang_000.png demo/lhm_verification
```

适配：只复用官方导出脚本中的纯函数，避开不需要的 GUI、视频与姿态估计导入；保留官方 synthetic camera、人体参数与 canonical export 逻辑。DINO 初始化不再次下载通用权重，随后通过完整模型 checkpoint 严格加载。ArcFace 使用独立官方先验权重。Point Transformer 的 varlen attention 由 xFormers block-diagonal memory-efficient attention 执行，不改变学习参数。

注意：`LHMPP-700M-SMPLX-FREE` 指无需从输入照片拟合 SMPL-X 形状先验，不代表完全不需要 SMPL-X/FLAME 文件；生成与标准姿态变换仍使用这些资产。组员 PLY 的确切生成来源未确认，本实验验证的是候选开源路线。

真实推理日志：`lhm_verify.log`。成功时应出现 `DONE`，并生成 `lhm_verification/gaussian.ply` 与 `verification.json`。仅权重加载或 GPU 测试通过不能算生成成功。

## 已完成的实际验证

- 官方单张 `00000_yuliang_000.png`：160000 高斯，全部 PLY 属性有限；总耗时 65.72 秒，推理与导出 3.17 秒，PyTorch 峰值显存 3.741GB。
- 另一人物 `sample_person.png` 经前景裁剪与白底预处理：160000 高斯，全部属性有限；总耗时 65.54 秒，推理与导出 3.18 秒，峰值显存 3.741GB。
- 第一份有效输出已导入本地工作台并实际渲染查看；对应结果不是组员的原文件。
- 全精度尝试因接近整卡显存且耗时过长而停止；FP16 / 全局 BF16 尝试导出了 NaN 文件，判为失败，不能用于展示。
- 最终适配为 BF16 神经网络、FP32 稀疏卷积，以及强制关闭高斯属性预测和人体姿态变换的 autocast。修复稀疏卷积输入类型后成功。严格匹配了主权重，训练用 ArcFace 使用单独权重。

工作台已新增 LHM++ 后端选项。网页上传的照片由 Windows 环境执行本地去背景和 504×840 白底裁剪，再由 WSL 进行高斯预测。每次任务重新加载模型，所以端到端约一分钟，不能把约三秒的纯推理/导出时间宣传成网页总耗时。输出为标准姿态，单图不可见背面由模型推测，不保证衣服、面部、手或姿态完全还原。

网页按钮端到端验证：任务 `7a2daba964eb4269b26940ae7d3822d4` 完成，模型启动至导出 63.94 秒、推理导出 2.95 秒、3.741GB。已在独立查看器实际检查其正面和背面并保存 PNG。嵌入查看器在窄屏/后台状态下可能显示空白，可点击“打开独立三维查看器”查看。截图为 `LHM本机单图生成正面.png` / `LHM本机单图生成背面.png`。
