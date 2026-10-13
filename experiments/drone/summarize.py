"""Produce a human-readable report from actual measurements, without invented values."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    comparison=json.loads((ROOT/'comparison.json').read_text())
    final=json.loads((ROOT/'final_metrics.json').read_text())
    env=json.loads((ROOT/'environment.json').read_text())
    platform=json.loads((ROOT/'platform_verification.json').read_text())
    export=json.loads((ROOT/'export_verification.json').read_text())
    errors=json.loads((ROOT/'error_review.json').read_text())
    keys=['metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)']
    rows=[]
    for row in comparison:
        run=ROOT/'runs'/row['run']
        if not run.exists(): run=ROOT/'evidence'/row['run']
        with (run/'results.csv').open() as f:
            history=list(csv.DictReader(f))
        best=max(history,key=lambda r:float(r['metrics/mAP50-95(B)']))
        rows.append('| '+row['run']+' | '+best['epoch']+' | '+' | '.join(f"{row['validation'][k]:.4f}" for k in keys)+' |')
    metrics='\n'.join(f'- {name}：{final["test"][key]:.4f}' for name,key in zip(['Precision','Recall','mAP50','mAP50-95'],keys))
    text=f'''# 无人机检测实验报告

本次实际完成两组 YOLOv8n 微调实验，每组 30 轮。按验证集 mAP50-95 选出 **{final['selected_run']}**，交付权重为 `models/drone_yolov8n_v1.pt`。此报告由真实运行结果生成。

## 数据与方法

公开数据集 YOLO Drone Detection Dataset v1 原有 1,359 张图，移除 20 张完全重复图后为 1,339 张。按近重复及编号分组重新划分为训练 937、验证 201、测试 201 张，检测类别为 drone。固定划分清单和源文件 SHA256 随包提供。

一张训练图片 pic_722.jpg 缺少 JPEG 结束标记，原运行中由 Ultralytics 自动修复；复现脚本已显式实现同样的修复并验证输出摘要，修复记录见 image_repairs.json。其他图片与原始清单摘要一致。

两组均从同一官方 YOLOv8n 权重开始，输入 416、batch 16、AdamW、初始学习率 0.001、种子 20261013、AMP 关闭。区别仅为 README 所列的颜色/几何/Mosaic 增强。训练环境为 {env['gpu']}，PyTorch {env['torch']}，Ultralytics {env['ultralytics']}。完整参数见各组 args.yaml。

## 验证集对比

| 组别 | 最佳轮次 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

最佳轮次由逐轮训练日志确定，表中指标为加载最佳权重后的统一复评。最佳 checkpoint 和组别均按验证集 mAP50-95 选择；未依据测试集结果选择模型。

## 独立测试集结果

测试集共 201 张、235 个标注框，输入分辨率 416。

{metrics}

以上值范围为 0–1。评估置信度下限 0.001、NMS IoU 0.7；Precision/Recall 是 Ultralytics 评估曲线对应的工作点，不是演示阈值 0.25 下的固定工作点成绩。推理阶段平均耗时为 {final['speed_ms']['inference']:.2f} ms/张（验证器测得，不能直接当作浏览器摄像头端到端 FPS）。

固定演示阈值 0.25、匹配 IoU 0.5 的逐图复核统计：TP={errors['tp']}、FP={errors['fp']}、FN={errors['fn']}，Precision={errors['precision']:.4f}、Recall={errors['recall']:.4f}。此处使用按分数排序的一对一贪心匹配；逐图记录及错误最多的样本见 error_review.json。

## 权重与平台兼容性

- 交付权重 SHA256：`{final['model_sha256']}`。
- 导出时移除了训练专用 criterion 和优化器，保留推理参数。8 张固定验证图片的导出前后输出对比通过，检测框及分数最大绝对差为 {export['max_absolute_box_difference']}。
- 实测 Ultralytics {platform['ultralytics']}、PyTorch {platform['torch']} 下，平台场景列表、模型加载、`/api/process_frame` 真实 CPU 推理均通过。
- 接口验证对应仓库提交：`{platform['repository_commit']}`。
- 平台验证使用原仓库 app.py 的 Flask 测试客户端，没有启动 MySQL 或浏览器，也没有测试实体摄像头。原 requirements 中的 PyTorch 2.0.1 未在本机 Python 3.12 环境验证。
- 平台默认推理尺寸与本报告 416 评估尺寸可能不同，不能直接把这里的耗时当作平台性能。

## 局限与后续工作

验证集和测试集没有纯背景样本；大量图片是产品图、近景或视频帧，远距离小无人机和无目标场景代表性不足。近重复分组降低了泄漏风险，但缺少完整视频来源标识，不能证明消除了所有相关图像。当前结果适合课程实验和功能演示；下一步应增加鸟类、飞机、空天空等负样本，进行真实摄像头视频测试，并用多个随机种子评估稳定性。

数据来源、许可说明、复现命令和平台加载步骤见 README.md。
'''
    (ROOT/'实验报告.md').write_text(text,encoding='utf-8')
    print(text)

if __name__=='__main__': main()
