#!/usr/bin/env python3
"""
YOLOv8 评估/测试脚本 — 在 COCO 验证集上评估模型

用法：
    # 评估训练好的模型
    python scripts/evaluate.py --model models/yolo_best.pt --data config/coco.yaml

    # 评估预训练模型（基准测试）
    python scripts/evaluate.py --model yolov8n.pt --data coco128

    # 评估并保存预测结果
    python scripts/evaluate.py --model models/yolo_best.pt --data config/coco.yaml --save-json
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 模型评估")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型权重路径（.pt）",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="config/coco.yaml",
        help="数据集配置",
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="批次大小"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="图像尺寸"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="设备"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="保存 COCO 格式的评估 JSON",
    )
    parser.add_argument(
        "--conf", type=float, default=0.001, help="置信度阈值"
    )
    parser.add_argument(
        "--iou", type=float, default=0.6, help="NMS IoU 阈值"
    )
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"YOLOv8 模型评估")
    print(f"  模型: {args.model}")
    print(f"  数据: {args.data}")
    print(f"  批次: {args.batch}")
    print(f"{'='*60}")

    # 加载模型
    model = YOLO(args.model)

    # 验证（评估）
    metrics = model.val(
        data=args.data,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        save_json=args.save_json,
        plots=True,
    )

    # 打印核心指标
    print(f"\n{'='*60}")
    print(f"评估结果")
    print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"  精度 (P)     : {metrics.box.mp:.4f}")
    print(f"  召回 (R)     : {metrics.box.mr:.4f}")
    print(f"{'='*60}")

    return metrics


if __name__ == "__main__":
    main()
