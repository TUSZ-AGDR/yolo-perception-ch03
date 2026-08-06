# YOLO 视觉感知实验（第三章）

围绕 YOLOv8 目标检测的实践教程，覆盖 COCO10 数据集上的全流程实验：原理学习、环境搭建、数据准备、模型训练、推理测试与性能分析。

## 目录结构

```
.
├── docs/                        # 教程正文（第3章 Markdown）
│   └── chapter03_yolo_perception.md
├── src/yolo_coco/               # 实验源码（唯一的工作目录）
│   ├── environment_coco.yml     # Conda 环境配置
│   ├── yolov8n.pt               # YOLOv8n 预训练权重
│   ├── config/                  # 数据集配置文件
│   ├── scripts/                 # 训练/评估/推理/视频流脚本
│   └── assets/                  # 示例图片、训练曲线、街景视频
└── resources/                   # 附加资源
    └── models/
        └── yolo_coco_best.pt    # 已训练好的 COCO10 模型（可跳过训练直接推理）
```

## 环境要求

- **操作系统**：Windows 10+ / macOS / Linux
- **Python**：3.9+（Conda 环境）
- **GPU**（可选）：NVIDIA + CUDA ≥ 11.8，纯 CPU 可运行

## 快速开始

```bash
# 1. 创建环境
cd src/yolo_coco
conda env create -f environment_coco.yml
conda activate yolo_coco

# 2. 准备数据集
python scripts/prepare_coco10.py

# 3. 训练（约 35 分钟 / GPU）
python scripts/train.py --model yolov8n.pt --data config/coco10.yaml --epochs 50 --batch 16

# 4. 评估
python scripts/evaluate.py --model outputs/train/<时间戳>/weights/best.pt --data config/coco10.yaml

# 5. 推理
python scripts/detect.py --model outputs/train/<时间戳>/weights/best.pt --source data/coco10/images/val/000000000001.jpg --save

# 6. 视频流检测
python scripts/detect_video.py --model outputs/train/<时间戳>/weights/best.pt --source assets/street_scene.mp4 --show
```

详细步骤请阅读 `docs/chapter03_yolo_perception.md`。

## 实验链路

```
3.1 目的 → 3.2 原理 → 3.3 环境 → 3.4 数据 → 3.5 训练 → 3.6 推理（含视频流） → 3.7 分析 → 3.8 总结
```

## 核心指标（实测，RTX 2060）

| 指标 | 数值 |
|------|------|
| mAP@0.5 | 0.552 |
| mAP@0.5:0.95 | 0.356 |
| 批量推理 | 64 FPS（15.7 ms/张） |
| 视频流推理 | 42 FPS（24 ms/帧） |

## 演示视频

GitHub 中最稳定的播放方式是点击下方链接进入视频文件页播放：

[点击播放 yolo_coco.mp4](resources/%E5%AE%9E%E9%AA%8C%E6%BC%94%E7%A4%BA%E8%A7%86%E9%A2%91/yolo_coco.mp4)

如果当前 GitHub 页面支持 HTML 标签，也可直接在 README 内播放：

<video src="resources/%E5%AE%9E%E9%AA%8C%E6%BC%94%E7%A4%BA%E8%A7%86%E9%A2%91/yolo_coco.mp4" controls width="100%"></video>
