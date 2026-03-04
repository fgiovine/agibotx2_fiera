"""Orchestratore interazione vocale Ciruzzo.

Gestisce il flusso completo:
1. Ascolta wake word "Ciruzzo"
2. Chiede "Vuoi che ti preparo un pacco o vuoi parlare?"
3. Classifica l'intento (pacco vs parlare)
4. Se parlare -> modalita conversazione con ChatGPT
5. Se pacco -> ritorna alla routine packaging
"""

import re
from typing import Optional
from enum import Enum

from rclpy.node import Node

from src.utils.config_loader import ConfigLoader
from src.interaction.speech_recognizer import SpeechRecognizer
from src.interaction.chat_client import ChatClient
from src.interaction.tts_client import TTSClient
from src.interaction.emoji_client import EmojiClient, EmojiCode
from src.interaction.led_client import LEDClient


class UserIntent(Enum):
    PACCO = "pacco"
    PARLARE = "parlare"
    UNKNOWN = "unknown"


# Parole chiave per classificare l'intento
PACCO_KEYWORDS = [
    'pacco', 'pacch', 'scatola', 'scatol', 'prepara', 'preparami',
    'lavora', 'lavorare', 'cialde', 'cialda', 'caffe', 'caffè',
    'vai', 'inizia', 'parti', 'comincia', 'fai', 'metti',
    'impacchetta', 'confeziona', 'prendi', 'si', 'sì', 'ok',
]

PARLARE_KEYWORDS = [
    'parlare', 'parla', 'parliam', 'chiacchier', 'chiacchiera',
    'conversa', 'dimmi', 'racconta', 'raccontami', 'chat',
    'parli', 'parliamo', 'voglio parlare', 'fare due chiacchiere',
]


class VoiceInteraction:
    """Gestisce l'interazione vocale completa con Ciruzzo."""

    def __init__(self, node: Node, config: ConfigLoader,
                 tts: TTSClient, emoji: EmojiClient, led: LEDClient):
        self._node = node
        self._config = config
        self._logger = node.get_logger()
        self._tts = tts
        self._emoji = emoji
        self._led = led

        # Moduli voce
        self._speech = SpeechRecognizer(node, config)
        self._chat = ChatClient(node, config)

        # Parametri
        self._wake_word = config.get('interaction.voice.wake_word', 'ciruzzo')
        self._chat_timeout_s = config.get(
            'interaction.voice.chat_timeout_s', 60.0
        )
        self._max_chat_turns = config.get(
            'interaction.voice.max_chat_turns', 10
        )

        # Stato
        self._in_conversation = False

    def check_wake_word(self) -> bool:
        """Controlla se qualcuno ha detto "Ciruzzo" (non bloccante).

        Ascolta per un breve periodo e verifica.

        Returns:
            True se il wake word e stato rilevato
        """
        if not self._speech.is_available:
            return False

        text = self._speech.listen(timeout_s=2.0)
        if text and self._wake_word.lower() in text.lower():
            self._logger.info(f"Wake word rilevato in: \"{text}\"")
            return True
        return False

    def handle_activation(self) -> UserIntent:
        """Gestisce l'attivazione dopo il wake word.

        Chiede cosa vuole l'utente e classifica l'intento.

        Returns:
            UserIntent.PACCO, UserIntent.PARLARE, o UserIntent.UNKNOWN
        """
        # Ciruzzo risponde con entusiasmo
        self._emoji.set_emoji(EmojiCode.EXTRA_HAPPY)
        self._led.green_steady()

        self._tts.speak(
            "Eccomi! Sono Ciruzzo! "
            "Vuoi che ti preparo un pacco oppure vuoi parlare con me?",
            blocking=True,
        )

        self._emoji.set_emoji(EmojiCode.THINKING)
        self._led.blue_steady()

        # Ascolta la risposta
        response = self._speech.listen(timeout_s=8.0)
        if response is None:
            self._tts.speak("Non ho sentito bene, puoi ripetere?")
            response = self._speech.listen(timeout_s=8.0)

        if response is None:
            self._tts.speak(
                "Va bene, torno a lavorare! Chiamami quando vuoi!",
            )
            self._emoji.happy()
            return UserIntent.UNKNOWN

        intent = self._classify_intent(response)
        self._logger.info(f"Risposta: \"{response}\" -> intento: {intent.value}")

        return intent

    def _classify_intent(self, text: str) -> UserIntent:
        """Classifica l'intento dell'utente dal testo.

        Args:
            text: testo trascritto

        Returns:
            UserIntent classificato
        """
        text_lower = text.lower().strip()

        # Cerca keyword "parlare"
        parlare_score = sum(
            1 for kw in PARLARE_KEYWORDS if kw in text_lower
        )

        # Cerca keyword "pacco"
        pacco_score = sum(
            1 for kw in PACCO_KEYWORDS if kw in text_lower
        )

        if parlare_score > pacco_score:
            return UserIntent.PARLARE
        elif pacco_score > 0:
            return UserIntent.PACCO
        else:
            # Default: se la frase e corta e affermativa -> pacco
            if text_lower in ('si', 'sì', 'ok', 'va bene', 'dai', 'certo'):
                return UserIntent.PACCO
            return UserIntent.UNKNOWN

    def start_conversation(self) -> str:
        """Avvia la modalita conversazione con ChatGPT.

        Il robot chiacchiera con il visitatore fino a che:
        - Il visitatore dice "basta", "ciao", "vai a lavorare"
        - Timeout raggiunto
        - Numero massimo di turni raggiunto

        Returns:
            Ultimo stato da cui riprendere ("IDLE")
        """
        self._in_conversation = True
        self._chat.reset_conversation()

        self._emoji.set_emoji(EmojiCode.EXTRA_HAPPY)
        self._led.green_breathing()

        self._tts.speak(
            "Uè, bello! Parliamo un po'! "
            "Di che vuoi parlare? Chiedi quello che vuoi a Ciruzzo!",
            blocking=True,
        )

        import time
        start_time = time.time()
        turns = 0

        while self._in_conversation and turns < self._max_chat_turns:
            # Controlla timeout
            if (time.time() - start_time) > self._chat_timeout_s:
                self._tts.speak(
                    "E stato bello chiacchierare! "
                    "Ma adesso devo tornare a lavorare. A dopo!"
                )
                break

            # Ascolta il visitatore
            self._emoji.set_emoji(EmojiCode.THINKING)
            self._led.blue_steady()

            user_text = self._speech.listen(timeout_s=10.0)

            if user_text is None:
                if turns > 0:
                    self._tts.speak("Ci sei ancora? Se vuoi torno a lavorare!")
                    user_text = self._speech.listen(timeout_s=5.0)
                    if user_text is None:
                        self._tts.speak("Ok, torno al lavoro! Ciao!")
                        break
                else:
                    self._tts.speak("Non ho sentito, puoi ripetere?")
                    continue

            # Controlla se vuole terminare
            if self._wants_to_stop(user_text):
                self._tts.speak(
                    "E stato un piacere! Torno a lavorare con le cialde. "
                    "Chiamami quando vuoi, jamm ja!"
                )
                break

            # Controlla se vuole il pacco
            if self._wants_pacco(user_text):
                self._tts.speak(
                    "Subito! Torno a preparare i pacchi! Jamm ja!"
                )
                self._in_conversation = False
                return "IDLE"

            # Controlla se vuole un saluto per la fiera
            if self._wants_saluto_fiera(user_text):
                self._emoji.set_emoji(EmojiCode.EXTRA_HAPPY)
                self._tts.speak(
                    "Ciao! Sono Ciruzzo, se mi volete conoscere "
                    "venite alla fiera MECSPE!",
                    blocking=True,
                )
                turns += 1
                continue

            # Rispondi con ChatGPT
            self._emoji.set_emoji(EmojiCode.HAPPY)
            self._led.green_steady()

            reply = self._chat.chat(user_text)
            self._tts.speak(reply, blocking=True)
            turns += 1

        self._in_conversation = False
        self._emoji.set_emoji(EmojiCode.HAPPY)
        self._led.green_breathing()

        return "IDLE"

    def _wants_to_stop(self, text: str) -> bool:
        """Controlla se l'utente vuole terminare la conversazione."""
        stop_words = [
            'basta', 'stop', 'ciao', 'arrivederci', 'addio',
            'vai a lavorare', 'torna a lavorare', 'lavora',
            'ho finito', 'fine', 'grazie basta',
        ]
        text_lower = text.lower()
        return any(w in text_lower for w in stop_words)

    def _wants_pacco(self, text: str) -> bool:
        """Controlla se l'utente vuole passare alla modalita pacco."""
        pacco_words = [
            'prepara il pacco', 'fai il pacco', 'preparami un pacco',
            'le cialde', 'metti le cialde', 'torna a lavorare',
            'fai le scatole', 'impacchetta',
        ]
        text_lower = text.lower()
        return any(w in text_lower for w in pacco_words)

    def _wants_saluto_fiera(self, text: str) -> bool:
        """Controlla se l'utente vuole un saluto per la fiera/amici."""
        saluto_words = [
            'saluta gli amici', 'saluto per la fiera', 'manda un saluto',
            'fai un saluto', 'saluta tutti', 'saluta il pubblico',
            'saluta la fiera', 'saluta per la fiera', 'un saluto',
            'salutali', 'saluta amici',
        ]
        text_lower = text.lower()
        return any(w in text_lower for w in saluto_words)

    @property
    def in_conversation(self) -> bool:
        return self._in_conversation

    @property
    def is_available(self) -> bool:
        return self._speech.is_available
