#!/usr/bin/env python3
"""Test Camera - Salva una foto dalla camera frontale."""

import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class TestCamera(Node):
    def __init__(self):
        super().__init__('test_camera')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Camera frontale RGB
        self.topic = '/aima/hal/sensor/rgbd_head_front/rgb_image'
        self.saved = False
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, self.topic, self.image_cb, qos)
        self.get_logger().info(f'In attesa di immagine da {self.topic}...')

    def image_cb(self, msg: Image):
        if self.saved:
            return

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        enc = msg.encoding.lower()
        if enc == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        ts = int(time.time() * 1000)
        path = f'foto_{ts}.png'
        cv2.imwrite(path, img)
        self.get_logger().info(
            f'Foto salvata: {path} ({img.shape[1]}x{img.shape[0]})')
        self.saved = True


def main():
    rclpy.init()
    node = TestCamera()

    start = time.time()
    while not node.saved and time.time() - start < 10.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    if not node.saved:
        print("Timeout: nessuna immagine ricevuta")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
