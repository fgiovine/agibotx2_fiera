"""Controllo LED strip colorati del robot.

Modalita LED:
- steady: colore fisso
- breathing: pulsazione lenta
- blinking: lampeggio veloce
- flowing: effetto scorrimento
"""

from typing import Tuple

from rclpy.node import Node

from agt_sdk_msgs.msg import LEDCommand

from src.utils.config_loader import ConfigLoader
from src.utils.ros_helpers import reliable_qos


class LEDMode:
    STEADY = 0
    BREATHING = 1
    BLINKING = 2
    FLOWING = 3


class LEDColor:
    """Colori predefiniti (R, G, B)."""
    GREEN = (0, 255, 0)
    BLUE = (0, 100, 255)
    YELLOW = (255, 255, 0)
    RED = (255, 0, 0)
    ORANGE = (255, 165, 0)
    PURPLE = (128, 0, 255)
    WHITE = (255, 255, 255)
    OFF = (0, 0, 0)


class LEDClient:
    """Controlla i LED strip del robot."""

    TOPIC = "/robot/led"

    def __init__(self, node: Node, config: ConfigLoader):
        self._node = node
        self._config = config
        self._logger = node.get_logger()

        self._brightness = config.get('interaction.led.brightness', 0.7)
        self._breathing_period = config.get(
            'interaction.led.breathing_period_s', 2.0
        )
        self._blinking_period = config.get(
            'interaction.led.blinking_period_s', 0.5
        )

        self._pub = node.create_publisher(
            LEDCommand, self.TOPIC, reliable_qos()
        )

    def set_led(self, color: Tuple[int, int, int],
                mode: int = LEDMode.STEADY,
                brightness: float = None,
                period: float = None):
        """Imposta i LED.

        Args:
            color: (R, G, B) 0-255
            mode: LEDMode.*
            brightness: 0.0-1.0
            period: periodo per breathing/blinking
        """
        if brightness is None:
            brightness = self._brightness

        if period is None:
            if mode == LEDMode.BREATHING:
                period = self._breathing_period
            elif mode == LEDMode.BLINKING:
                period = self._blinking_period
            else:
                period = 1.0

        msg = LEDCommand()
        msg.r = int(color[0] * brightness)
        msg.g = int(color[1] * brightness)
        msg.b = int(color[2] * brightness)
        msg.mode = mode
        msg.period = period

        self._pub.publish(msg)

    def green_breathing(self):
        """Verde pulsante - stato idle."""
        self.set_led(LEDColor.GREEN, LEDMode.BREATHING)

    def blue_steady(self):
        """Blu fisso - detection in corso."""
        self.set_led(LEDColor.BLUE, LEDMode.STEADY)

    def yellow_steady(self):
        """Giallo fisso - pick in corso."""
        self.set_led(LEDColor.YELLOW, LEDMode.STEADY)

    def green_steady(self):
        """Verde fisso - place riuscito."""
        self.set_led(LEDColor.GREEN, LEDMode.STEADY)

    def orange_blinking(self):
        """Arancio lampeggiante - scatola piena."""
        self.set_led(LEDColor.ORANGE, LEDMode.BLINKING)

    def purple_flowing(self):
        """Viola scorrimento - trasporto scatola."""
        self.set_led(LEDColor.PURPLE, LEDMode.FLOWING)

    def red_blinking(self):
        """Rosso lampeggiante - errore."""
        self.set_led(LEDColor.RED, LEDMode.BLINKING)

    def off(self):
        """Spegni LED."""
        self.set_led(LEDColor.OFF, LEDMode.STEADY)
