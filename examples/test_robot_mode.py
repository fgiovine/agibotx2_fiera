#!/usr/bin/env python3
"""Test Get/Set Robot Mode - Verifica lo stato del robot."""

import sys
import rclpy
import rclpy.logging
from rclpy.node import Node

from aimdk_msgs.srv import GetMcAction, SetMcAction
from aimdk_msgs.msg import CommonRequest, RequestHeader, McActionCommand, CommonState


class RobotModeClient(Node):
    def __init__(self):
        super().__init__('robot_mode_client')
        self.get_client = self.create_client(
            GetMcAction, '/aimdk_5Fmsgs/srv/GetMcAction')
        self.set_client = self.create_client(
            SetMcAction, '/aimdk_5Fmsgs/srv/SetMcAction')
        self.get_logger().info('Robot mode client pronto')

    def _call_service(self, client, request, timeout=0.25, retries=8):
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Servizio non disponibile')
            return None

        for i in range(retries):
            request.request.header.stamp = self.get_clock().now().to_msg() \
                if hasattr(request, 'request') else None
            if hasattr(request, 'header'):
                request.header.stamp = self.get_clock().now().to_msg()

            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
            if future.done():
                return future.result()
            self.get_logger().info(f'Tentativo {i+1}...')
        return None

    def get_mode(self):
        req = GetMcAction.Request()
        req.request = CommonRequest()
        resp = self._call_service(self.get_client, req)
        if resp:
            self.get_logger().info(f'Modalita: {resp.info.action_desc}')
            self.get_logger().info(f'Stato: {resp.info.status.value}')
        return resp

    def set_mode(self, action_name: str):
        req = SetMcAction.Request()
        req.header = RequestHeader()
        cmd = McActionCommand()
        cmd.action_desc = action_name
        req.command = cmd

        self.get_logger().info(f'Imposto modalita: {action_name}')
        resp = self._call_service(self.set_client, req)
        if resp and resp.response.status.value == CommonState.SUCCESS:
            self.get_logger().info('Modalita impostata!')
        elif resp:
            self.get_logger().error(f'Errore: {resp.response.message}')
        return resp


MODES = {
    'PD': ('PASSIVE_DEFAULT', 'giunti senza coppia'),
    'DD': ('DAMPING_DEFAULT', 'giunti in damping'),
    'JD': ('JOINT_DEFAULT', 'posizione bloccata'),
    'SD': ('STAND_DEFAULT', 'in piedi stabile'),
    'LD': ('LOCOMOTION_DEFAULT', 'camminata'),
}


def main():
    rclpy.init()
    node = RobotModeClient()

    if len(sys.argv) > 1:
        cmd = sys.argv[1].upper()
        if cmd == 'GET':
            node.get_mode()
        elif cmd in MODES:
            node.set_mode(MODES[cmd][0])
        else:
            print(f'Comando sconosciuto: {cmd}')
    else:
        print('Uso: python3 test_robot_mode.py [GET|PD|DD|JD|SD|LD]')
        print()
        for k, (name, desc) in MODES.items():
            print(f'  {k:3s} - {name:25s} ({desc})')
        print()
        node.get_mode()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
