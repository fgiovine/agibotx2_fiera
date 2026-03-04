#!/usr/bin/env python3
"""Test Input Source - Registra come sorgente input per controllare il robot."""

import rclpy
import rclpy.logging
from rclpy.node import Node

from aimdk_msgs.srv import SetMcInputSource
from aimdk_msgs.msg import RequestHeader, McInputAction


class TestInputSource(Node):
    def __init__(self):
        super().__init__('test_input_source')
        self.client = self.create_client(
            SetMcInputSource, '/aimdk_5Fmsgs/srv/SetMcInputSource')
        self.get_logger().info('Input source client pronto')

    def register(self):
        if not self.client.wait_for_service(timeout_sec=8.0):
            self.get_logger().error('Servizio non disponibile')
            return False

        req = SetMcInputSource.Request()
        req.action = McInputAction()
        req.action.value = 1001  # register
        req.input_source.name = 'ciruzzo'
        req.input_source.priority = 40
        req.input_source.timeout = 1000

        for i in range(8):
            req.request.header.stamp = self.get_clock().now().to_msg()
            future = self.client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.25)
            if future.done():
                break
            self.get_logger().info(f'Tentativo {i+1}...')

        if not future.done():
            self.get_logger().error('Timeout registrazione')
            return False

        resp = future.result()
        code = resp.response.header.code
        if code == 0:
            self.get_logger().info(
                f'Registrato! task_id={resp.response.task_id}')
            return True
        else:
            self.get_logger().error(f'Errore registrazione: code={code}')
            return False


def main():
    rclpy.init()
    node = TestInputSource()
    ok = node.register()
    print(f"Registrazione: {'OK' if ok else 'FALLITA'}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
