# YOLO 视觉感知实验（第三章）

围绕 YOLOv8 目标检测的实践实验，覆盖标准数据集训练与 Gazebo 仿真部署。

## 实验概览

| # | 内容 | 核心目录 |
|---|------|----------|
| 1 | COCO 子集上完成 YOLOv8 训练、评估与推理 | `src/yolo_coco/` |
| 2 | Gazebo 仿真中构建 ROS2 实时检测节点，域随机化场景生成、自动标注与模型训练 | `src/yolo_vision/` + `src/scripts/` |

## 目录结构

```
.
├── docs/              # 教程正文（Markdown + PDF）
├── src/
│   ├── scripts/       # 工具脚本（自动标注、数据生成、训练启动等）
│   ├── yolo_coco/     # 实验一：COCO 检测
│   └── yolo_vision/   # 实验二：ROS2 视觉节点
└── resources/
    ├── datasets/      # block_dataset 训练数据
    ├── environments/  # Conda 环境配置
    └── models/        # 预训练权重
```

## 环境要求

- **OS**: Ubuntu 24.04
- **ROS2**: Jazzy Jalisco
- **Gazebo**: Harmonic
- **Python**: 3.12 (Conda)
- **GPU**（可选）: NVIDIA + CUDA ≥ 11.8

## 快速开始

### 1. 配置环境

```bash
# 实验一（COCO 检测）
conda env create -f resources/environments/environment_coco.yml

# 实验二（ROS2 视觉节点）
conda env create -f resources/environments/environment_vision.yml
```

### 2. 展开工作目录

教程命令假定源码和资源位于同一扁平目录：

```bash
mkdir -p ~/yolo_workspace
cp -r src/yolo_coco              ~/yolo_workspace/
cp -r src/yolo_vision            ~/yolo_workspace/
cp -r src/scripts                ~/yolo_workspace/
cp -r resources/datasets/block_dataset ~/yolo_workspace/
cp    resources/models/*.pt      ~/yolo_workspace/
```

### 3. 运行实验

详见 `docs/chapter03_yolo_perception.md` 中各实验小节的步骤说明。

## 模型权重

| 文件 | 用途 | 大小 |
|------|------|------|
| `yolov8n.pt` | YOLOv8n 官方预训练权重 | 6 MB |
| `yolo_coco_best.pt` | 实验一：COCO10 子集微调最优权重 | 6 MB |
| `block_best.pt` | 实验二：block_dataset 训练最优权重 | 6 MB |

## 许可证

本项目仅用于教学目的。
