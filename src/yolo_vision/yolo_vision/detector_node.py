#!/usr/bin/env python3
"""
yolo_vision.detector_node — YOLO 检测 ROS2 节点
===============================================
通信架构（发布/订阅）：
  Sub: /camera/color/image_raw  (sensor_msgs/Image)
  Pub: /detections              (vision_msgs/Detection2DArray)
  Pub: /detections_image        (sensor_msgs/Image, 标注可视化)

模型从包内 models/best.pt 加载，支持参数覆盖。
"""

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

import numpy as np
from ament_index_python.packages import get_package_share_directory


class DetectorNode(Node):

    def __init__(self):
        super().__init__('yolo_detector')

        # ---- 参数 ----
        self.declare_parameter('model', 'best.pt')
        self.declare_parameter('confidence', 0.5)
        self.declare_parameter('device', 'cuda:0')

        model_name = self.get_parameter('model').value
        self.conf = self.get_parameter('confidence').value
        device = self.get_parameter('device').value

        # 模型路径：包内 models/ > 绝对路径 > 自动下载默认模型
        if os.path.isabs(model_name) and os.path.exists(model_name):
            model_path = model_name
        else:
            share_dir = get_package_share_directory('yolo_vision')
            model_path = os.path.join(share_dir, 'models', model_name)
            if not os.path.exists(model_path):
                self.get_logger().warn(f'Model not found: {model_path}, using yolov8n.pt')
                model_path = 'yolov8n.pt'

        self.get_logger().info(f'Loading model: {model_path} on {device}')
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.model.to(device)

        self.bridge = CvBridge()

        # ---- 发布者 ----
        qos = QoSProfile(depth=10)
        self.det_pub = self.create_publisher(Detection2DArray, '/detections', qos)
        self.img_pub = self.create_publisher(Image, '/detections_image', qos)

        # ---- 订阅者（BEST_EFFORT 兼容 Gazebo） ----
        cam_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.callback, cam_qos
        )

        self.get_logger().info('YOLO Detector ready.')

    def callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        results = self.model(cv_img, conf=self.conf, verbose=False)
        if not results or results[0].boxes is None:
            return

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss  = r.boxes.cls.cpu().numpy()

        # 构建 Detection2DArray
        det_msg = Detection2DArray()
        det_msg.header.stamp = msg.header.stamp
        det_msg.header.frame_id = msg.header.frame_id or 'camera_link'

        for box, conf, cls_id in zip(boxes, confs, clss):
            x1, y1, x2, y2 = box
            w_img = msg.width if msg.width > 0 else 640
            h_img = msg.height if msg.height > 0 else 480

            d = Detection2D()
            d.header = det_msg.header
            d.bbox.center.position.x = float((x1 + x2) / 2.0 / w_img)
            d.bbox.center.position.y = float((y1 + y2) / 2.0 / h_img)
            d.bbox.size_x = float((x2 - x1) / w_img)
            d.bbox.size_y = float((y2 - y1) / h_img)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(int(cls_id))
            hyp.hypothesis.score = float(conf)
            d.results.append(hyp)

            det_msg.detections.append(d)

        self.det_pub.publish(det_msg)

        # 标注图像
        annotated = r.plot()
        img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        img_msg.header.stamp = msg.header.stamp
        img_msg.header.frame_id = msg.header.frame_id or 'camera_link'
        self.img_pub.publish(img_msg)


def main():
    rclpy.init()
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
