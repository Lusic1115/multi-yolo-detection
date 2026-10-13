# 无人机检测可复现实验

面向 Lusic1115/multi-yolo-detection 的单类别目标检测实验，对应 SSSteven0629 的模型协作任务。本包保留数据划分、训练配置、实测结果及可加载权重。模型由本次从官方 YOLOv8n 通用预训练权重微调得到。

## 实验设计

- 数据：MUKILAN S 发布的 YOLO Drone Detection Dataset，第 1 版，原始 1,012 张训练图片、347 张验证图片。
- 去除解码像素完全相同的图片，剩 1,339 张。相同文件名前缀且数字编号落在同一 20 张区间的图片，以及 64 位 dHash 距离不超过 4 的图片，按传递闭包归为同组。
- 固定种子 20261013，按组重新划分：训练 937 张、验证 201 张、测试 201 张。同组和检出的近重复图片不跨集合。具体图片及 SHA256 见 split_manifest.json。
- 文件编号分组是降低泄漏风险的启发式方法，原始数据没有完整来源视频标识，不能保证消除了所有语义相似或相邻帧泄漏。
- 原始训练图片 pic_722.jpg 缺少 JPEG 结束标记，首次训练时被 Ultralytics 自动修复。prepare.py 现在显式执行相同修复，并校验结果；image_repairs.json 保留原始与修复后 SHA256，split_manifest.json 保留原始文件摘要。
- 两组均使用 YOLOv8n、416 像素、batch 16、AdamW、学习率 0.001、30 epochs、固定种子、关闭 AMP。基础组关闭列出的颜色/几何/拼接增强；增强组启用 HSV、平移、缩放、水平翻转、Mosaic，最后 5 轮关闭 Mosaic。
- 按验证集 mAP50-95 选择组别及最佳 checkpoint；只对胜出模型进行最终测试集评估。没有使用测试指标调参。
- mAP 评估使用 conf=0.001、NMS IoU=0.7；演示推理使用 conf=0.25。Ultralytics 汇总的 Precision/Recall 对应其评估曲线所选工作点，不应解读为固定 0.25 阈值的 Precision/Recall。

## 平台加载

权重由 SSSteven0629 单独上传到项目根目录的 `models/drone_yolov8n_v1.pt`，上传后拉取该分支即可使用；本次脚本提交不包含权重二进制文件。选择“无人机检测”，选中该模型并加载。模型类别为 0: drone，输出为检测框，文件名含 drone，符合当前后端场景匹配规则。

推荐先用图片检测验证，再尝试视频和摄像头。该模型仅识别 drone，不负责火焰、烟雾、病虫害或无人机航拍画面中的通用目标。

如需复现本次旧版 Ultralytics 接口检查，可另建一个 Python 3.12 虚拟环境，安装 requirements-platform-smoke.txt，在数据准备完成后执行 `python verify_platform.py --repo 你的平台仓库路径`。此脚本通过 Flask 测试客户端调用真实模型，不需要 MySQL；不要把这份测试环境依赖直接覆盖到现有平台 requirements.txt。

## 重现实验

需要 Python 3.12、支持 CUDA 12.4 的 NVIDIA 驱动、约 8 GB 显存，以及用于依赖和数据的磁盘空间。实际环境和完整依赖版本分别记录在 environment.json 与 requirements-lock.txt。

在全新克隆的仓库中进入 `experiments/drone` 后运行（训练将更新仓库根目录的交付权重）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe prepare.py
.\.venv\Scripts\python.exe run.py --device 0 --epochs 30
.\.venv\Scripts\python.exe verify.py
.\.venv\Scripts\python.exe analyze_errors.py
```

prepare.py 下载固定版本数据、验证源压缩包 SHA256，重新生成绝对路径 data.yaml，并校验生成清单与随包冻结清单完全一致。run.py 会自动下载官方初始 yolov8n.pt 并校验 SHA256，来源与摘要见 initial_weights.json。若已有 runs 目录，请另建实验目录，脚本不会覆盖旧实验。相同配置不意味着跨硬件、驱动或软件版本逐位一致。

## 文件说明

- ../../models/drone_yolov8n_v1.pt：平台交付权重。
- run.py、prepare.py、verify.py：训练评估、数据准备和完整性/推理验证脚本。
- comparison.json：两组验证结果及训练耗时。
- final_metrics.json：选中模型的独立测试结果、推理阶段耗时及权重 SHA256。
- error_review.json：固定置信度 0.25、匹配 IoU 0.5 下的 TP/FP/FN 和逐图失败案例，按错误数量排序。
- export_verification.json：移除训练专用对象后，导出前后 8 张验证图片的输出一致性检查。
- platform_verification.json：对原仓库场景列表、模型加载、图片帧推理接口的真实调用结果。验证环境使用 Ultralytics 8.0.196 和 PyTorch 2.5.1，不等于对旧 requirements 中全部依赖组合的验证。
- evidence：本次运行的训练曲线、逐轮 CSV、参数和评估图。
- 本目录中的 evidence/baseline 与 evidence/augmented 对应运行记录里的 runs/baseline 与 runs/augmented，便于在实验目录中重新训练而不覆盖证据。
- split_manifest.json、data_summary.json：数据清单和去重分组统计。

## 数据与软件来源

- 数据集：https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset （发布者注明原始贡献者 https://www.kaggle.com/dasmehdixtr）。Kaggle 标注许可为 “Other (specified in description)”，说明没有明确列出再分发许可，本仓库不打包原始图片库；复现时从原地址获取。正式公开发布数据或商业使用前需确认上游授权。
- 初始权重：https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
- Ultralytics：https://github.com/ultralytics/ultralytics ，遵循其 AGPL-3.0/商业许可规则。本交付不变更上游软件及模型许可。
- 训练参数参考：https://docs.ultralytics.com/modes/train/

这是课程实验与演示模型。独立测试集只有 201 张，结果不能代表真实环境的全天候可靠性。抽查可见产品图、广告/标志图和视频帧，部分标注框对机臂/起落架的覆盖口径不一致；后续需人工清理标注，并增加鸟类、飞机、空天空等负样本及实际摄像头视频，评估误报和小目标漏检。

公开记录中的本机绝对路径已改为相对路径；实验指标和参数数值保持不变。evidence 仅包含曲线、混淆矩阵及数值记录，原始图片和图片拼图由复现时生成。
