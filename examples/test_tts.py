#!/usr/bin/env python3
"""TTS Italiano per Ciruzzo - usa il servizio SDK con domain='it'."""

import rclpy
from rclpy.node import Node
from aimdk_msgs.srv import PlayTts
from aimdk_msgs.msg import PlayTtsRequest, CommonRequest


class CiruzzoTTS(Node):
    def __init__(self):
        super().__init__('ciruzzo_tts')
        self.client = self.create_client(
            PlayTts, '/aimdk_5Fmsgs/srv/PlayTts')
        self.get_logger().info('TTS italiano pronto')

    def speak(self, text: str, wait=True):
        if not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Servizio TTS non disponibile')
            return False

        req = PlayTts.Request()
        req.header = CommonRequest()
        req.tts_req = PlayTtsRequest()
        req.tts_req.text = text
        req.tts_req.domain = 'it'
        req.tts_req.trace_id = 'ciruzzo'

        self.get_logger().info(f'Dico: "{text}"')

        for i in range(8):
            req.header.header.stamp = self.get_clock().now().to_msg()
            future = self.client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)
            if future.done():
                return True
            self.get_logger().info(f'Tentativo {i+1}...')

        self.get_logger().error('TTS timeout')
        return False


def main():
    import time

    rclpy.init()
    node = CiruzzoTTS()

    node.speak("Uè! Sono Ciruzzo! Piacere di conoscerti!")
    time.sleep(4.0)

    node.speak("Vuoi che ti preparo un bel pacco di cialde?")
    time.sleep(4.0)

    node.speak("Jamme jà! Si parte!")
    time.sleep(3.0)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
