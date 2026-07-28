#!/usr/bin/env python3
"""
从 COCO train2017 中均衡采样 10 个常见类别，生成教程数据集。

10 类：person, bicycle, car, motorcycle, bus, truck, cat, dog, chair, bottle
策略：从 train2017 前 20000 张中，确保每类至少 200 个训练实例，总计约 3000-4000 张。
"""

import json
import os
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 目标 10 类（COCO 原始 ID → 新 ID 0-9）
TARGET_CLASSES = {
    1: 0,   # person
    2: 1,   # bicycle
    3: 2,   # car
    4: 3,   # motorcycle
    6: 4,   # bus
    8: 5,   # truck
    17: 6,  # cat
    18: 7,  # dog
    62: 8,  # chair
    44: 9,  # bottle
}

TARGET_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "cat", "dog", "chair", "bottle"]

MIN_INSTANCES_PER_CLASS = 200   # 训练集中每类最少实例数
MAX_IMAGES = 4000               # 图片上限


def download_labels(data_dir: Path):
    """下载 COCO 2017 标注文件"""
    from ultralytics.utils.downloads import download

    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
    labels_dir = data_dir / "_raw"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 仅下载标注 JSON（不下载图片）
    print("[下载] COCO 标注文件...")
    download(url, dir=str(labels_dir))

    # 解压到 data/coco10/_raw/
    import zipfile
    zip_path = labels_dir / "coco2017labels.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            # 只提取需要的文件
            for member in zf.namelist():
                if "instances_train2017" in member or "instances_val2017" in member:
                    zf.extract(member, str(labels_dir))
        print(f"[OK] 标注文件已解压到 {labels_dir}")
    else:
        # 可能直接下载到目录中
        print("[INFO] 标注文件已就绪")

    return labels_dir


def download_images(image_ids: list, images_dir: Path):
    """下载指定的 COCO 图片"""
    from ultralytics.utils.downloads import download

    images_dir.mkdir(parents=True, exist_ok=True)
    base_url = "http://images.cocodataset.org/train2017/"

    # 过滤尚未下载的
    todo = [f"{base_url}{img_id:012d}.jpg" for img_id in image_ids
            if not (images_dir / f"{img_id:012d}.jpg").exists()]

    if not todo:
        print("[INFO] 所有图片已下载")
        return

    print(f"[下载] {len(todo)} 张图片 -> {images_dir} ...")
    download(todo, dir=str(images_dir), threads=8)


def build_dataset(data_dir: Path):
    """主流程：下载标注 → 筛选图片 → 均衡采样 → 生成 YOLO 标注"""
    raw_dir = data_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 确保有标注文件
    ann_file = raw_dir / "annotations" / "instances_train2017.json"
    if not ann_file.exists():
        ann_file = raw_dir / "instances_train2017.json"
    if not ann_file.exists():
        download_labels(data_dir)
        # 查找解压后的位置
        for root, dirs, files in os.walk(str(raw_dir)):
            for f in files:
                if f == "instances_train2017.json":
                    ann_file = Path(root) / f
                    break

    if not ann_file.exists():
        print("[ERROR] 未找到 instances_train2017.json，请检查下载")
        sys.exit(1)

    # Step 2: 解析标注，找出包含目标类别的图片
    print(f"[解析] {ann_file} ...")
    with open(ann_file) as f:
        coco = json.load(f)

    # 建立 image_id → annotations 映射（只保留目标类别）
    img_anns = {}  # image_id → [annotations with target classes]
    for ann in coco["annotations"]:
        if ann["category_id"] in TARGET_CLASSES:
            img_id = ann["image_id"]
            if img_id not in img_anns:
                img_anns[img_id] = []
            img_anns[img_id].append(ann)

    # 建立 image_id → image_info 映射
    img_info = {img["id"]: img for img in coco["images"]}

    print(f"[统计] 包含目标类别的图片: {len(img_anns)} 张")

    # Step 3: 均衡采样
    # 对每张图片，统计它包含哪些目标类
    img_classes = {}  # image_id → set of new_class_ids
    for img_id, anns in img_anns.items():
        classes = set()
        for ann in anns:
            new_cls = TARGET_CLASSES[ann["category_id"]]
            classes.add(new_cls)
        img_classes[img_id] = classes

    # 贪心采样：优先选包含稀有类别的图片
    selected = []
    class_counts = {i: 0 for i in range(10)}
    # 统计每个 img_id 中各标注类别的实例数
    img_class_instance_counts = {}
    for img_id, anns in img_anns.items():
        counts = {i: 0 for i in range(10)}
        for ann in anns:
            new_cls = TARGET_CLASSES[ann["category_id"]]
            counts[new_cls] += 1
        img_class_instance_counts[img_id] = counts

    # 候选图片按"含稀有类别"优先级排序
    candidates = list(img_anns.keys())
    random.shuffle(candidates)

    def score_image(img_id):
        """分数越低越优先：包含的类别中，当前累计数最少的那个类的权重"""
        counts = img_class_instance_counts[img_id]
        min_count = min(class_counts.values())
        # 给含稀有类的图片高分
        rare_bonus = sum(1.0 / max(class_counts[cls] + counts.get(cls, 0) + 1, 1)
                        for cls in img_classes[img_id])
        return -rare_bonus  # 负分 = 高优先级

    # 多轮采样
    for round_num in range(5):
        candidates.sort(key=score_image)
        for img_id in candidates:
            if img_id in selected:
                continue
            if len(selected) >= MAX_IMAGES:
                break
            # 检查是否有助于填充稀有类
            help_rare = any(class_counts[cls] < MIN_INSTANCES_PER_CLASS
                           for cls in img_classes[img_id])
            if help_rare or len(selected) < MAX_IMAGES * 0.7:
                selected.append(img_id)
                for cls in img_classes[img_id]:
                    class_counts[cls] += img_class_instance_counts[img_id].get(cls, 1)
        if all(c >= MIN_INSTANCES_PER_CLASS for c in class_counts.values()):
            break

    print(f"\n[采样结果] 选中 {len(selected)} 张图片")
    print(f"[类别分布]")
    for i in range(10):
        print(f"  {TARGET_NAMES[i]:12s}: {class_counts[i]:5d} 实例")

    # Step 4: 下载图片
    images_dir = data_dir / "images" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    download_images(selected, images_dir)

    # Step 5: 80/20 划分 train/val
    random.shuffle(selected)
    split = int(len(selected) * 0.8)
    train_ids = set(selected[:split])
    val_ids = set(selected[split:])

    # Step 6: 生成 YOLO 格式标注
    for subset, ids, subdir in [("train", train_ids, "train"), ("val", val_ids, "val")]:
        labels_dir = data_dir / "labels" / subdir
        labels_dir.mkdir(parents=True, exist_ok=True)
        img_subdir = data_dir / "images" / subdir
        img_subdir.mkdir(parents=True, exist_ok=True)

        for img_id in ids:
            info = img_info[img_id]
            w, h = info["width"], info["height"]

            # YOLO 标注
            lines = []
            for ann in img_anns.get(img_id, []):
                cat = ann["category_id"]
                if cat not in TARGET_CLASSES:
                    continue
                new_cls = TARGET_CLASSES[cat]
                bbox = ann["bbox"]  # COCO: [x, y, w, h]
                x_center = (bbox[0] + bbox[2] / 2) / w
                y_center = (bbox[1] + bbox[3] / 2) / h
                bw = bbox[2] / w
                bh = bbox[3] / h
                lines.append(f"{new_cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

            if not lines:
                continue

            label_path = labels_dir / f"{img_id:012d}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(lines))

            # 移动/复制图片到子目录
            src_img = data_dir / "images" / "train" / f"{img_id:012d}.jpg"
            dst_img = img_subdir / f"{img_id:012d}.jpg"
            if src_img.exists() and not dst_img.exists():
                shutil.move(str(src_img), str(dst_img))

        print(f"[{subset}] {len(ids)} 张图片, 标注已生成")

    # 清理残留
    leftover = data_dir / "images" / "train"
    if leftover.exists():
        remaining = list(leftover.glob("*.jpg"))
        if not remaining:
            leftover.rmdir()

    print(f"\n[OK] COCO10 数据集就绪！")
    print(f"  训练集: {len(train_ids)} 张")
    print(f"  验证集: {len(val_ids)} 张")
    print(f"  位置: {data_dir}")


if __name__ == "__main__":
    data_dir = PROJECT_ROOT / "data" / "coco10"
    build_dataset(data_dir)
