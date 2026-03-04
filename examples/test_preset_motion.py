#!/usr/bin/env python3
"""Test Preset Motion - Fa eseguire gesti al robot (saluto, onda, bacio)."""

import sys
import rclpy
from rclpy.node import Node

from aimdk_msgs.srv import SetMcPresetMotion
from aimdk_msgs.msg import McPresetMotion, McControlArea, RequestHeader, CommonState


MOTIONS = {
    'raise':     (1001, 'Alza il braccio'),
    'wave':      (1002, 'Saluta con la mano'),
    'handshake': (1003, 'Stretta di mano'),
    'airkiss':   (1004, 'Manda un bacio'),
}

AREAS = {
    'left':  1,
    'right': 2,
}


class TestPresetMotion(Node):
    def __init__(self):
        super().__init__('test_preset_motion')
        self.client = self.create_client(
            SetMcPresetMotion, '/aimdk_5Fmsgs/srv/SetMcPresetMotion')
        self.get_logger().info('Preset motion client pronto')

    def execute(self, area_id: int, motion_id: int):
        if not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Servizio non disponibile')
            return False

        req = SetMcPresetMotion.Request()
        req.header = RequestHeader()
        req.motion = McPresetMotion()
        req.area = McControlArea()
        req.motion.value = motion_id
        req.area.value = area_id
        req.interrupt = False

        for i in range(8):
            req.header.stamp = self.get_clock().now().to_msg()
            future = self.client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.25)
            if future.done():
                break

        if future.done():
            resp = future.result()
            if resp.response.header.code == 0:
                self.get_logger().info('Movimento avviato!')
                return True
            elif resp.response.state.value == CommonState.RUNNING:
                self.get_logger().info('Movimento in esecuzione...')
                return True

        self.get_logger().error('Errore esecuzione movimento')
        return False


def main():
    if len(sys.argv) < 3:
        print('Uso: python3 test_preset_motion.py <left|right> <raise|wave|handshake|airkiss>')
        print()
        for name, (mid, desc) in MOTIONS.items():
            print(f'  {name:12s} ({mid}) - {desc}')
        return

    side = sys.argv[1].lower()
    motion = sys.argv[2].lower()

    if side not in AREAS:
        print(f'Braccio non valido: {side} (usa left o right)')
        return
    if motion not in MOTIONS:
        print(f'Movimento non valido: {motion}')
        return

    rclpy.init()
    node = TestPresetMotion()
    node.execute(AREAS[side], MOTIONS[motion][0])
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
