"""Client espressioni facciali (emoji) sul display del robot.

Codici emoji dal SDK Agibot X2:
- 90: Happy
- 100: Extra happy
- 101: Ecstatic
- 110: Sad
- 120: Angry
- 130: Surprised
- 140: Confused
- 150: Sleepy
- 160: Neutral
- 170: Thinking
"""

from rclpy.node import Node

from agt_sdk_msgs.msg import EmojiCommand

from src.utils.config_loader import ConfigLoader
from src.utils.ros_helpers import reliable_qos


class EmojiCode:
    """Codici espressioni facciali."""
    HAPPY = 90
    EXTRA_HAPPY = 100
    ECSTATIC = 101
    SAD = 110
    ANGRY = 120
    SURPRISED = 130
    CONFUSED = 140
    SLEEPY = 150
    NEUTRAL = 160
    THINKING = 170


EMOJI_NAMES = {
    EmojiCode.HAPPY: "Happy",
    EmojiCode.EXTRA_HAPPY: "Extra Happy",
    EmojiCode.ECSTATIC: "Ecstatic",
    EmojiCode.SAD: "Sad",
    EmojiCode.ANGRY: "Angry",
    EmojiCode.SURPRISED: "Surprised",
    EmojiCode.CONFUSED: "Confused",
    EmojiCode.SLEEPY: "Sleepy",
    EmojiCode.NEUTRAL: "Neutral",
    EmojiCode.THINKING: "Thinking",
}


class EmojiClient:
    """Controlla le espressioni facciali del robot."""

    TOPIC = "/robot/emoji"

    def __init__(self, node: Node, config: ConfigLoader):
        self._node = node
        self._config = config
        self._logger = node.get_logger()

        self._duration = config.get('interaction.emoji.duration_s', 3.0)
        self._current_emoji: int = EmojiCode.NEUTRAL

        self._pub = node.create_publisher(
            EmojiCommand, self.TOPIC, reliable_qos()
        )

    def set_emoji(self, code: int, duration: float = None):
        """Imposta un'espressione facciale.

        Args:
            code: codice emoji (EmojiCode.*)
            duration: durata in secondi (None = usa default)
        """
        if duration is None:
            duration = self._duration

        msg = EmojiCommand()
        msg.emoji_id = code
        msg.duration = duration

        self._pub.publish(msg)
        self._current_emoji = code

        name = EMOJI_NAMES.get(code, f"Unknown({code})")
        self._logger.debug(f"Emoji: {name} per {duration:.1f}s")

    def happy(self):
        self.set_emoji(EmojiCode.HAPPY)

    def extra_happy(self):
        self.set_emoji(EmojiCode.EXTRA_HAPPY)

    def ecstatic(self):
        self.set_emoji(EmojiCode.ECSTATIC)

    def sad(self):
        self.set_emoji(EmojiCode.SAD)

    def thinking(self):
        self.set_emoji(EmojiCode.THINKING)

    def surprised(self):
        self.set_emoji(EmojiCode.SURPRISED)

    def neutral(self):
        self.set_emoji(EmojiCode.NEUTRAL)

    @property
    def current_emoji(self) -> int:
        return self._current_emoji
