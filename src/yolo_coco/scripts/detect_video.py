#!/usr/bin/env python3
"""YOLO 视频流实时检测脚本

用法:
    # 对视频文件进行检测
    python detect_video.py --model best.pt --source video.mp4 --show

    # 对摄像头实时检测
    python detect_video.py --model best.pt --source 0 --show

    # 仅保存结果不显示
    python detect_video.py --model best.pt --source video.mp4
"""
import argparse
import cv2
import time
from pathlib import Path
from ultralytics import YOLO


def detect_video(model_path: str, video_path: str, conf: float = 0.25,
                 save: bool = True, show: bool = False):
    """对视频文件或摄像头逐帧进行 YOLO 推理并实时渲染"""
    # 加载模型
    model = YOLO(model_path)
    class_names = model.names  # {0: 'person', 1: 'bicycle', ...}

    # 打开视频源（文件路径或摄像头索引）
    source = int(video_path) if video_path.isdigit() else video_path
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频源: {video_path}")

    # 读取视频属性
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    is_camera = isinstance(source, int)
    if is_camera:
        print(f"摄像头: {width}×{height}")
    else:
        print(f"视频信息: {width}×{height}, {fps_input:.1f} fps, {total_frames} 帧")

    # 输出视频写入器
    out = None
    if save:
        output_dir = Path("outputs/detect_video")
        output_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = fps_input if fps_input > 0 else 30.0  # 摄像头可能返回 0
        out = cv2.VideoWriter(
            str(output_dir / "result.mp4"), fourcc, out_fps, (width, height)
        )

    # 逐帧推理
    frame_idx = 0
    inference_times = []
    detections_per_frame = []

    print("开始推理...（按 Q 键可提前退出）")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 推理
        t_start = time.time()
        results = model(frame, conf=conf, iou=0.7, verbose=False)
        t_infer = (time.time() - t_start) * 1000  # 毫秒
        inference_times.append(t_infer)

        # 统计检出目标数
        n_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        detections_per_frame.append(n_detections)

        # 绘制检测框
        annotated = results[0].plot()

        # 叠加帧号和推理耗时
        cv2.putText(annotated, f"Frame: {frame_idx}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Inference: {t_infer:.1f}ms",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Detections: {n_detections}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if out is not None:
            out.write(annotated)

        if show:
            cv2.imshow("YOLO Video Detection", annotated)

        frame_idx += 1

        # 按 Q 键退出
        if show and cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out is not None:
        out.release()
    if show:
        cv2.destroyAllWindows()

    # 统计输出
    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        effective_fps = 1000.0 / avg_time
        avg_detections = sum(detections_per_frame) / len(detections_per_frame)
        print(f"\n===== 推理统计 =====")
        print(f"总帧数: {frame_idx}")
        print(f"平均推理耗时: {avg_time:.1f} ms/帧")
        print(f"有效推理帧率: {effective_fps:.1f} FPS")
        print(f"平均每帧检出目标数: {avg_detections:.1f}")
        if save:
            print(f"结果保存至: outputs/detect_video/result.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 视频流检测")
    parser.add_argument("--model", required=True, help="模型权重路径 (.pt)")
    parser.add_argument("--source", required=True,
                        help="视频文件路径 或 摄像头索引 (如 0)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--show", action="store_true", help="实时显示检测画面")
    parser.add_argument("--no-save", action="store_true", help="不保存结果视频")
    args = parser.parse_args()

    detect_video(args.model, args.source, args.conf,
                 save=not args.no_save, show=args.show)
