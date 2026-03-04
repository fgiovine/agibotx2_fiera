#!/usr/bin/env python3
"""Test Locomotion - Fa camminare il robot."""

import sys
import time
import signal
import rclpy
from rclpy.node import Node

from aimdk_msgs.msg import McLocomotionVelocity, MessageHeader
from aimdk_msgs.srv import SetMcInputSource, SetMcAction
from aimdk_msgs.msg import RequestHeader, McActionCommand, CommonState


class TestWalk(Node):
    def __init__(self):
        super().__init__('test_walk')

        self.vel_pub = self.create_publisher(
            McLocomotionVelocity, '/aima/mc/locomotion/velocity', 10)
        self.input_client = self.create_client(
            SetMcInputSource, '/aimdk_5Fmsgs/srv/SetMcInputSource')
        self.mode_client = self.create_client(
            SetMcAction, '/aimdk_5Fmsgs/srv/SetMcAction')

        self.forward = 0.0
        self.lateral = 0.0
        self.angular = 0.0
        self.timer = None

        self.get_logger().info('Walk controller pronto')

    def register_input(self):
        if not self.input_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('SetMcInputSource non disponibile')
            return False

        req = SetMcInputSource.Request()
        req.action.value = 1001
        req.input_source.name = 'ciruzzo'
        req.input_source.priority = 40
        req.input_source.timeout = 1000

        for i in range(8):
            req.request.header.stamp = self.get_clock().now().to_msg()
            future = self.input_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.25)
            if future.done():
                break

        if future.done():
            self.get_logger().info('Input source registrato')
            return True
        return False

    def set_locomotion_mode(self):
        if not self.mode_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('SetMcAction non disponibile')
            return False

        req = SetMcAction.Request()
        req.header = RequestHeader()
        cmd = McActionCommand()
        cmd.action_desc = 'LOCOMOTION_DEFAULT'
        req.command = cmd

        for i in range(8):
            req.header.stamp = self.get_clock().now().to_msg()
            future = self.mode_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.25)
            if future.done():
                break

        if future.done():
            resp = future.result()
            if resp.response.status.value == CommonState.SUCCESS:
                self.get_logger().info('Modalita LOCOMOTION attivata')
                return True
        return False

    def start_publishing(self):
        self.timer = self.create_timer(0.02, self._publish)

    def _publish(self):
        msg = McLocomotionVelocity()
        msg.header = MessageHeader()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = 'ciruzzo'
        msg.forward_velocity = self.forward
        msg.lateral_velocity = self.lateral
        msg.angular_velocity = self.angular
        self.vel_pub.publish(msg)

    def stop(self):
        self.forward = 0.0
        self.lateral = 0.0
        self.angular = 0.0


def main():
    rclpy.init()
    node = TestWalk()

    # Registra input
    if not node.register_input():
        print("Registrazione input fallita!")
        node.destroy_node()
        rclpy.shutdown()
        return

    # Imposta locomotion mode
    if not node.set_locomotion_mode():
        print("Cambio modalita fallito!")
        node.destroy_node()
        rclpy.shutdown()
        return

    # Cammina avanti lentamente per 3 secondi
    node.forward = 0.2  # 0.2 m/s (molto lento, sicuro)
    node.start_publishing()
    print("Cammino avanti a 0.2 m/s per 3 secondi...")

    start = time.time()
    while time.time() - start < 3.0:
        rclpy.spin_once(node, timeout_sec=0.02)

    node.stop()
    print("Stop!")

    # Pubblica zero per un po' per fermarsi
    for _ in range(50):
        rclpy.spin_once(node, timeout_sec=0.02)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
