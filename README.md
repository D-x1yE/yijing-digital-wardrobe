# 衣境 · 对话数字衣橱

本地运行的三维人物试衣工作台：人物照片 → Qwen 工具调度 → LHM++ 原始版本 V0 → FASHN 二维试衣 → 人工确认 → LHM++ 新版本 V1 → 动作视频。

## 功能

- 1–4 张同人照片联合重建，三维高斯自由视角预览。
- 商品图试衣，检查身份、衣服细节与视角后再保存新版本。
- V0/V1 版本保留、对比、撤销；预设舞蹈、太极、跳跃及 MP4 下载。
- 本机 Qwen3.5-4B 量化模型负责文字理解与受限工具调度。
- 简约三栏工作台、拖拽上传、聊天和移动端布局。

## 目录

- `demo/avatar.html`、`avatar.css`、`avatar.js`：主界面。
- `demo/avatar_backend.py`、`serve_studio.py`：工作流与任务服务。
- `demo/run_fashn.py`：真实二维试衣。
- `demo/run_lhm_verify.py`、`lhm_animation.py`：三维重建与动作。
- `demo/serve_qwen.py`：本机 CPU 对话服务。
- `vendor/LHM-plusplus-main`：当前使用的上游源码快照（不含权重与人体资产）。
- `docs/SETUP.md`：组员部署步骤及现有机器路径约定。

## 快速使用

已配置环境的开发机：双击 `demo/start_avatar_demo.cmd`，打开 http://127.0.0.1:8770/avatar.html 。

**新电脑不能只克隆后双击运行。** 需要安装 Python、WSL、CUDA 扩展并单独下载模型，具体见 [部署说明](docs/SETUP.md) 与 [详细工作流](demo/README_AVATAR_WORKFLOW.md)。

已在 RTX 4060 Laptop 8GB 上验证单图试衣与 V0/V1 重建、动画。Qwen 使用 CPU，FASHN 与 LHM++ 串行使用 GPU。多视角试衣一致性需要单独检查。

## 仓库范围

本仓库保存项目源码、网页依赖、开发说明与下载脚本。模型权重、虚拟环境、人体先验、上传照片、对话记录、历史生成结果、课程论文和汇报文件不进 Git；本地原文件保留。组员按说明下载模型即可共享代码开发。

质量检查目前为人工确认，尚未接入自动视觉质检。衣物背面、身份和服装细节可能失真，V1 是重新重建而非仅替换 V0 材质。动作以视频呈现，网页中的自由旋转预览为静态人物。

## 第三方许可

第三方源码与渲染库保留各自 LICENSE。LHM++ 的权重、SMPL-X/FLAME 等人体资产另有授权要求，不随源码仓库重新分发。详见 [第三方来源](docs/THIRD_PARTY.md)。项目自有代码暂未授予对外开源许可证，供团队内部协作。
