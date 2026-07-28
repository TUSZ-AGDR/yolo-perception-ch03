#!/usr/bin/env python3
"""
下载 COCO 数据集脚本

用法：
    # 下载 COCO128（128 张图，快速验证，约 7MB）
    python scripts/download_coco.py --variant coco128

    # 下载完整 COCO 2017（约 19GB，需要较长时间）
    python scripts/download_coco.py --variant coco

说明：
    ultralytics 内置了 COCO 数据集自动下载功能。
    首次训练时会自动下载到 ~/.cache/ultralytics/datasets/ 或指定路径。
    本脚本可在训练前预先下载，避免训练时等待。
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_coco128(data_dir: Path):
    """下载 COCO128 数据集（128 张图，用于快速测试）"""
    from ultralytics.utils.downloads import download

    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
    target = data_dir / "coco128.zip"

    print(f"[INFO] 下载 COCO128 到 {data_dir} ...")
    download(url, dir=str(data_dir), unzip=True)
    print(f"[OK] COCO128 下载完成，位于 {data_dir / 'coco128'}")


def download_coco_full(data_dir: Path):
    """
    下载完整 COCO 2017 数据集。
    使用 ultralytics 内置方式：直接调用 YOLO 训练时自动下载。
    这里通过触发 ultralytics 的 dataset check 来实现。
    """
    from ultralytics.data.utils import check_det_dataset

    yaml_path = PROJECT_ROOT / "config" / "coco.yaml"

    print(f"[INFO] 检查并下载 COCO 2017 数据集（这可能需要较长时间，约 19GB）...")
    print(f"[INFO] 数据集将下载到 {data_dir}")

    # ultralytics 会自动检测并下载缺失的数据集
    data_info = check_det_dataset(str(yaml_path))
    print(f"[OK] COCO 数据集就绪：{data_info}")


def main():
    parser = argparse.ArgumentParser(description="下载 COCO 数据集")
    parser.add_argument(
        "--variant",
        choices=["coco128", "coco"],
        default="coco128",
        help="选择数据集版本：coco128（快速测试）或 coco（完整训练）",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="数据集存放目录（默认 data/coco）",
    )
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        data_dir = (PROJECT_ROOT / "data" / "coco").resolve()

    data_dir.mkdir(parents=True, exist_ok=True)

    if args.variant == "coco128":
        download_coco128(data_dir)
    else:
        download_coco_full(data_dir)


if __name__ == "__main__":
    main()
