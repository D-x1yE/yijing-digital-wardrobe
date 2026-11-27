# 衣境：Qwen + FASHN + LHM++ 对话数字衣橱

入口：http://127.0.0.1:8770/avatar.html 。双击 `start_avatar_demo.cmd` 启动网页与本机 Qwen CPU 服务，首次加载对话模型需要等待。

## 操作流程

1. 上传1–4张同一人物、同一套衣服的照片。可以先在对话框填写建模需求，再点击“创建原始三维人物”：Qwen选择reconstruct，LHM++生成V0。
2. 上传衣服图片，选择上装/下装/连衣裙，以及平铺商品图或真人穿着图。点击生成候选，或发送“给当前版本穿上这件上衣”；对话模型会选择tryon和衣物类别。
3. 对照原图、服装和候选图，逐项检查身份/体型、衣服细节和视角一致性。勾选三项后确认，才启动LHM++生成V1。拒绝不会覆盖V0，可以重新试衣。
4. 选择版本后生成舞蹈、太极或跳跃，或对话“让V1跳舞”；每个版本保存自己的动画，已有同一动作直接播放。
5. 对话“查看V0背面”“比较V0和V1”“撤销”；也可用按钮操作。撤销只改变当前版本，保留V0/V1及其资产。

## 实现与边界

- Qwen3.5-4B使用Unsloth Q4_K_M量化权重，llama-cpp-python 0.3.38在CPU执行工具分类，限制JSON操作集合，后端验证版本、动作和任务状态。当前聊天不运行视觉编码器，不应把文字调度或人工确认描述成自动视觉质量评估。视觉投影权重已保存，尚未接入。
- FASHN VTON 1.5执行真实二维试衣；DWPose和人体解析器运行在CPU，扩散模型在GPU运行。采用20步、单张采样、segmentation_free=False以限制无关区域改动。图片类型需要选对。FASHN此接口不接收自由文字改变衣服外观；颜色、图案来自商品参考图。
- LHM++使用现有LHMPP-700M-SMPLX-FREE权重，生成标准姿态高斯。动作使用官方SMPL-X驱动与gsplat渲染，输出MP4。视频与静态三维预览分别展示。
- V1是重新重建的三维人物，不是对V0只替换材质，面部和体型可能变化。没有衣服背面图时不能保证忠实恢复背面。逐视角二维试衣不保证多视角一致，本次模型验收使用单视角；多视角必须人工检查。
- 质量确认是人工流程。Qwen没有批准候选的工具权限。示例中的确认仅用于演示样例验收，不代表所有照片都能得到同等效果。
- 同一服务串行执行FASHN/LHM++任务；不要同时在另一个DiGS服务执行GPU推理。

## 存储与环境

- 网站、后端：本目录 `avatar.html`、`avatar_backend.py`、`serve_studio.py`。
- 项目状态与不可覆盖的版本参考图：`avatar_sessions/<项目ID>/session.json`、original及candidates。
- 三维模型、输入、视频与执行日志：`studio_results/<任务ID>/`。
- 模型及源码：`D:\AvatarPipeline`，避免占用C盘。
- WSL环境：`/opt/fashn-venv`；复用原`/opt/lhm-venv`的基础包，通过独立site-packages安装附加依赖。LHM运行环境与权重保持独立。
- `download_avatar_models.py`提供断点分块下载与SHA-256校验。人体解析器另存`D:\AvatarPipeline\weights\parser`，包含官方config.json与model.safetensors。
- 本机Qwen接口为127.0.0.1:8791，网页API为127.0.0.1:8770，无需云端API密钥，不向外部发送人物照片。

模型来源：https://huggingface.co/Qwen/Qwen3.5-4B 、https://huggingface.co/unsloth/Qwen3.5-4B-GGUF 、https://github.com/fashn-AI/fashn-vton-1.5 、https://lingtengqiu.github.io/LHM++/ 。遵守各模型、人体先验与样例的许可证。

## 2026-11-26验收

样例项目b3a13144828444baa26412f8937a4756：原图真实重建V0，FASHN生成候选，经样例人工检查后真实重建V1，再用对话生成V1舞蹈。V0/V1均为160000个有效高斯，V0文件SHA-256保持一致。V1动画72帧、15fps、4.8秒，峰值已分配显存3.741GiB。浏览器播放成功，下载MP4与服务端原文件SHA-256一致。

FASHN单图测试耗时63.9秒（含加载），峰值已分配显存3.64GiB；与其他GPU模型串行。Qwen真实验证了建模规划、连衣裙类别与试衣工具选择、背面查看、比较、撤销和V1动作选择；并非任意自然语言指令准确率评测。详见avatar_workflow_acceptance.json。
