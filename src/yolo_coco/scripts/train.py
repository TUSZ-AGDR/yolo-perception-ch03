#!/usr/bin/env python3
"""
YOLOv8 训练脚本 — 使用 COCO 数据集

用法：
    # GPU 训练（默认自动检测）
    python scripts/train.py --model yolov8n.pt --data config/coco128.yaml --epochs 10

    # CPU 训练（无 GPU 环境）
    python scripts/train.py --model yolov8n.pt --data config/coco128.yaml --epochs 10 --device cpu

    # 指定 GPU 编号
    python scripts/train.py --model yolov8n.pt --data config/coco128.yaml --epochs 10 --device 0

    # 使用完整 COCO 2017 训练
    python scripts/train.py --model yolov8n.pt --data config/coco.yaml --epochs 100

    # 使用更大的模型
    python scripts/train.py --model yolov8s.pt --data config/coco.yaml --epochs 100 --batch 16

    # 从零开始训练（不使用预训练权重）
    python scripts/train.py --model yolov8n.yaml --data config/coco.yaml --epochs 300

模型选择：
    n=nano(最快), s=small, m=medium, l=large, x=xlarge
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 COCO 训练")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="模型文件：yolov8n.pt（预训练）或 yolov8n.yaml（从头训练）",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="coco128",
        help="数据集配置：coco128 / config/coco.yaml / 自定义 yaml",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="训练轮数（默认 100）"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="批次大小（默认 16，按 GPU 显存调整）"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="输入图像尺寸（默认 640）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备：0/1（GPU 编号）/ cpu（默认自动）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="数据加载线程数（默认 4，内存不足时减小）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从上次中断处恢复训练",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="实验名称（默认自动生成含时间戳的名称）",
    )
    args = parser.parse_args()

    # 实验名称
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(args.model).stem
        args.name = f"{model_name}_coco_{timestamp}"

    # 确保输出目录
    output_dir = PROJECT_ROOT / "outputs" / "train"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"YOLOv8 COCO 训练")
    print(f"  模型: {args.model}")
    print(f"  数据: {args.data}")
    print(f"  轮数: {args.epochs}")
    print(f"  批次: {args.batch}")
    print(f"  尺寸: {args.imgsz}")
    print(f"  实验: {args.name}")
    print(f"{'='*60}")

    # 加载模型
    model = YOLO(args.model)

    # 训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        resume=args.resume,
        name=args.name,
        workers=args.workers,
        project=str(output_dir),
        exist_ok=True,
        # 常用优化参数
        patience=50,          # 早停：50 轮无提升则停止
        save=True,
        save_period=10,       # 每 10 轮保存一次
        val=True,             # 每轮验证
        plots=True,           # 生成训练曲线图
    )

    # 保存最佳模型到 models/ 目录
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        import shutil
        dest = PROJECT_ROOT / "models" / f"{args.name}_best.pt"
        shutil.copy(best_pt, dest)
        print(f"[OK] 最佳模型已保存到: {dest}")

    print(f"[OK] 训练完成！结果保存在: {results.save_dir}")


if __name__ == "__main__":
    main()
