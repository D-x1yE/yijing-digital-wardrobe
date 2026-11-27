# 小组部署说明

## 已验证开发环境

Windows + WSL Ubuntu 22.04，RTX 4060 Laptop 8GB。Windows 的服务/抠图 Python 与 WSL 的 GPU Python 分开。

1. 克隆仓库，使用 `python -m pip install -r requirements-windows.txt` 配置网页服务与抠图依赖。
2. 在 WSL 为 LHM++ 配置 `/opt/lhm-venv`（Python 3.10、Torch 2.3.0+cu121、xformers 0.0.26.post1）。精确环境见 `demo/lhm_environment_verified.json`、`demo/lhm_wsl_requirements.txt`；这些是现有机器记录，不是通用无条件安装脚本。
3. 按 `vendor/LHM-plusplus-main/INSTALL_CN.md` 安装 pointops、spconv、高斯渲染、pytorch3d 等 CUDA 依赖。参考本项目 `demo/lhm_build_extensions.sh`，安装前检查路径与 CUDA 版本。
4. 获取 LHM++ 的 `LHMPP-700M-SMPLX-FREE` 和必要人体先验到 `vendor/LHM-plusplus-main/pretrained_models/`。下载脚本 `demo/setup_lhm_assets.py`，运行前阅读上游权重与人体资产许可。动作素材运行 `python demo/download_lhm_motions.py`。
5. 官方 FASHN 源码单独克隆到 `D:\AvatarPipeline\fashn-vton-1.5`：https://github.com/fashn-AI/fashn-vton-1.5 。在 `/opt/fashn-venv` 安装，保留 LHM++ 的原环境；CPU 版 onnxruntime 1.20.1 用于姿态检测，安装 fashn-human-parser 0.1.1。
6. `python demo/download_avatar_models.py` 下载并检查 FASHN、DWPose、Qwen GGUF 权重到 `D:\AvatarPipeline\weights`。人体解析器还需从 `fashn-ai/fashn-human-parser` 获取 `config.json` 与 `model.safetensors` 到 `weights/parser`；来源可使用 Hugging Face 或 ModelScope，同名文件应核对 SHA-256。
7. `/opt/fashn-venv` 安装 `llama-cpp-python==0.3.38`（CPU 构建），用于 Qwen3.5-4B Q4_K_M。视觉投影权重已在下载列表中，但当前未启用视觉评估。
8. Windows 安装 u2net/rembg 模型。参考 `demo/prepare_lhm_photo.py`。示例人物来自原实验素材；使用自己的已授权照片验证亦可。
9. 修改 `.cmd` 里的 `E:\555\envs\sd_torch\python.exe` 为组员机器的 Python。D 盘模型路径及 WSL 解释器路径目前写在源码中，需要按下表核对。

| 路径 | 使用位置 |
|---|---|
| D:/AvatarPipeline/weights | download_avatar_models.py、avatar_backend.py |
| /mnt/d/AvatarPipeline/weights | run_fashn.py、serve_qwen.py |
| /opt/fashn-venv/bin/python | avatar_backend.py、start_avatar_demo.py |
| /opt/lhm-venv/bin/python | serve_studio.py |
| vendor/LHM-plusplus-main | run_lhm_verify.py、lhm_animation.py |

## 启动

双击 `demo/start_avatar_demo.cmd`；手动启动 Windows 服务可用 `python demo/serve_studio.py`。Qwen 启动命令为：

```
wsl -d Ubuntu-22.04 -- /opt/fashn-venv/bin/python /mnt/<盘符>/<克隆目录>/demo/serve_qwen.py
```

浏览器入口 http://127.0.0.1:8770/avatar.html ，Qwen 服务 127.0.0.1:8791。首次生成依次加载各模型，需等待。旧 DiGS 实验位于开发者本机 D 盘独立项目，并非此仓库可复现的一部分。

## 协作

组员先 fork 或由仓库所有者在 Settings → Collaborators 添加，再各自创建分支提交 Pull Request。不要提交上传照片、模型文件、会话或生成目录；`.gitignore` 已配置。仓库为私有时，应由所有者邀请组员获得读取权限。
