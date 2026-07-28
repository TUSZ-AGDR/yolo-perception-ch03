#!/usr/bin/env python3
"""
YOLOv8 推理/检测脚本 — 对图像/视频/摄像头进行目标检测

用法：
    # 单张图片
    python scripts/detect.py --model yolov8n.pt --source test.jpg

    # 图片目录（批量）
    python scripts/detect.py --model models/best.pt --source data/images/

    # 视频文件
    python scripts/detect.py --model yolov8n.pt --source video.mp4

    # 摄像头（实时）
    python scripts/detect.py --model yolov8n.pt --source 0

    # 保存结果并显示
    python scripts/detect.py --model models/best.pt --source test.jpg --save --show
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 目标检测")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型权重路径（.pt）",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="输入源：图片路径 / 目录 / 视频 / 摄像头编号",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="置信度阈值（默认 0.25）"
    )
    parser.add_argument(
        "--iou", type=float, default=0.7, help="NMS IoU 阈值（默认 0.7）"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="推理图像尺寸"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="设备"
    )
    parser.add_argument(
        "--save", action="store_true", help="保存检测结果"
    )
    parser.add_argument(
        "--show", action="store_true", help="显示检测结果窗口"
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="过滤类别，如 --classes 0 2（只检测 person 和 car）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录（默认 outputs/detect/）",
    )
    args = parser.parse_args()

    # 输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "outputs" / "detect"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"YOLOv8 目标检测")
    print(f"  模型  : {args.model}")
    print(f"  输入  : {args.source}")
    print(f"  置信度: {args.conf}")
    print(f"  输出  : {output_dir}")
    print(f"{'='*60}")

    # 加载模型
    model = YOLO(args.model)

    # 推理
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
        show=args.show,
        classes=args.classes,
        project=str(output_dir),
        name="run",
        exist_ok=True,
    )

    # 统计
    total_objects = sum(len(r.boxes) for r in results)
    print(f"\n[OK] 检测完成: {len(results)} 张图像, 共检测到 {total_objects} 个目标")

    if args.save:
        print(f"[OK] 结果已保存到: {output_dir / 'run'}")


if __name__ == "__main__":
    main()
