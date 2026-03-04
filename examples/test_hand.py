#!/usr/bin/env python3
"""Test Gripper - Apri/chiudi le mani del robot."""

import rclpy
from rclpy.node import Node
from aimdk_msgs.msg import HandCommandArray, HandCommand, HandType, MessageHeader


class TestHand(Node):
    def __init__(self):
        super().__init__('test_hand')
        self.pub = self.create_publisher(
            HandCommandArray, '/aima/hal/joint/hand/command', 10)
        self.timer = self.create_timer(0.02, self.publish)
        self.left_pos = 1.0
        self.right_pos = 1.0
        self.get_logger().info('Hand control pronto')

    def set_positions(self, left: float, right: float):
        self.left_pos = left
        self.right_pos = right
        self.get_logger().info(f'Mani: left={left:.1f} right={right:.1f}')

    def publish(self):
        msg = HandCommandArray()
        msg.header = MessageHeader()

        left = HandCommand()
        left.name = "left_hand"
        left.position = self.left_pos
        left.velocity = 1.0
        left.acceleration = 1.0
        left.deceleration = 1.0
        left.effort = 1.0

        right = HandCommand()
        right.name = "right_hand"
        right.position = self.right_pos
        right.velocity = 1.0
        right.acceleration = 1.0
        right.deceleration = 1.0
        right.effort = 1.0

        msg.left_hand_type = HandType(value=2)
        msg.right_hand_type = HandType(value=2)
        msg.left_hands = [left]
        msg.right_hands = [right]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = TestHand()

    import time

    print("Apro le mani...")
    node.set_positions(1.0, 1.0)
    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.02)

    time.sleep(2.0)

    print("Chiudo le mani...")
    node.set_positions(0.0, 0.0)
    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.02)

    time.sleep(2.0)

    print("Apro a meta...")
    node.set_positions(0.5, 0.5)
    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.02)

    time.sleep(2.0)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
