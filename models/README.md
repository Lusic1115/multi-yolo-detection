# 无人机检测权重

将本次实验交付的 `drone_yolov8n_v1.pt` 上传到本目录，保留文件名。此脚本提交未包含权重二进制文件，由 SSSteven0629 单独上传。

- 文件大小：6,226,471 字节。
- SHA256：`901158e866867e9c97f9ad29eb3171d9fd6f54fdb752810d799432f0de7a715a`。
- 单类别目标检测：`0: drone`。
- 在平台选择“无人机检测”，再选择此模型并加载。

GitHub 网页上传时，请新建 `feat/drone-reproducible-experiment` 分支（如果已存在则选择它），进入本目录，使用 **Add file → Upload files**，上传后提交到同一分支，再创建或更新指向 main 的 PR。

也可按 [复现说明](../experiments/drone/README.md) 重新训练；新训练的权重会输出到本目录，跨设备重训的文件摘要和指标不一定逐位相同。
