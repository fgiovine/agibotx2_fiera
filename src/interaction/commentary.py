"""Script frasi/emoji/LED per ogni stato della demo.

Ciruzzo - il robot napoletano del caffe!
Per ogni stato della FSM definisce cosa dire, che espressione mostrare
e che colore LED impostare. Tutte le frasi sono in italiano con tocco napoletano.
"""

from src.interaction.tts_client import TTSClient
from src.interaction.emoji_client import EmojiClient, EmojiCode
from src.interaction.led_client import LEDClient


# Script per ogni stato - personalita Ciruzzo
STATE_SCRIPTS = {
    "IDLE": {
        "phrases": [
            "Ciao! Sono Ciruzzo! Mettete le cialde sul tavolo e ci penso io!",
            "Uè, sono pronto! Appoggiate le cialde quando volete.",
            "Ciruzzo e pronto a lavorare! Chi mette le cialde sul tavolo?",
            "Eccomi qua! Mettete le cialde e Ciruzzo fa il resto!",
        ],
        "emoji": EmojiCode.HAPPY,
        "led": "green_breathing",
    },
    "SCAN_TABLES": {
        "phrases": [
            "Un momento, sto guardando dove sono i tavoli...",
            "Aspettate che Ciruzzo si guarda intorno!",
        ],
        "emoji": EmojiCode.THINKING,
        "led": "blue_steady",
    },
    "DETECT_PODS": {
        "phrases": [
            "Uè, vedo delle cialde!",
            "Ecco, ne ho trovata una! Bell stu caffe!",
            "Ne vedo una! Adesso la prendo, jamm ja!",
        ],
        "emoji": EmojiCode.THINKING,
        "led": "blue_steady",
    },
    "NAV_TO_POD_TABLE": {
        "phrases": [
            "Vado a prendere la cialda, un attimo!",
            "Mi avvicino, Ciruzzo arriva!",
        ],
        "emoji": EmojiCode.HAPPY,
        "led": "blue_steady",
    },
    "PICK_POD": {
        "phrases": [
            "Adesso prendo questa cialda, piano piano!",
            "Ecco, la afferro delicatamente. Ciruzzo ha mano buona!",
            "Presa! Che bella cialda!",
        ],
        "emoji": EmojiCode.HAPPY,
        "led": "yellow_steady",
    },
    "NAV_TO_BOX_TABLE": {
        "phrases": [
            "Porto la cialda alla scatola!",
            "Vado alla scatola, un momento!",
        ],
        "emoji": EmojiCode.HAPPY,
        "led": "yellow_steady",
    },
    "PLACE_POD_IN_BOX": {
        "phrases": [
            "Ecco, la metto nella scatola! Perfetto!",
            "Una in piu nella scatola! Ciruzzo e preciso!",
            "Cialda posizionata, bell e buon!",
        ],
        "emoji": EmojiCode.EXTRA_HAPPY,
        "led": "green_steady",
    },
    "CHECK_BOX": {
        "phrases": [],  # Transizione veloce
        "emoji": EmojiCode.HAPPY,
        "led": "green_steady",
    },
    "BOX_FULL": {
        "phrases": [
            "La scatola e piena! Venti cialde, Ciruzzo e un campione!",
            "Scatola completa! La chiudo, jamm ja!",
        ],
        "emoji": EmojiCode.ECSTATIC,
        "led": "orange_blinking",
    },
    "CLOSE_BOX": {
        "phrases": [
            "Metto il coperchio, un attimo!",
            "Ecco, chiudo la scatola. Ciruzzo fa tutto!",
        ],
        "emoji": EmojiCode.HAPPY,
        "led": "orange_blinking",
    },
    "CARRY_BOX": {
        "phrases": [
            "Porto la scatola piena al tavolo!",
            "Scatola pronta! La porto via, attenti che passo!",
        ],
        "emoji": EmojiCode.HAPPY,
        "led": "purple_flowing",
    },
    "PLACE_FULL_BOX": {
        "phrases": [
            "Ecco la scatola piena! Bel lavoro Ciruzzo!",
        ],
        "emoji": EmojiCode.EXTRA_HAPPY,
        "led": "purple_flowing",
    },
    "GET_EMPTY_BOX": {
        "phrases": [
            "Prendo una scatola vuota, si ricomincia!",
            "Nuova scatola! Ciruzzo non si ferma mai!",
        ],
        "emoji": EmojiCode.HAPPY,
        "led": "blue_steady",
    },
    "WAKE_WORD_DETECTED": {
        "phrases": [],  # Gestito da VoiceInteraction
        "emoji": EmojiCode.SURPRISED,
        "led": "green_steady",
    },
    "CONVERSATION": {
        "phrases": [],  # Gestito da VoiceInteraction/ChatClient
        "emoji": EmojiCode.EXTRA_HAPPY,
        "led": "green_breathing",
    },
    "ERROR": {
        "phrases": [
            "Ue, qualcosa non ha funzionato! Ma Ciruzzo ci riprova!",
            "Un piccolo problema, ma niente paura!",
        ],
        "emoji": EmojiCode.SAD,
        "led": "red_blinking",
    },
}


class Commentary:
    """Gestisce l'interazione di Ciruzzo con il pubblico per ogni stato."""

    def __init__(self, tts: TTSClient, emoji: EmojiClient, led: LEDClient):
        self._tts = tts
        self._emoji = emoji
        self._led = led
        self._phrase_counters: dict = {}  # Contatore per rotazione frasi

    def play_state(self, state_name: str, speak: bool = True):
        """Esegue lo script di interazione per uno stato.

        Args:
            state_name: nome dello stato (chiave in STATE_SCRIPTS)
            speak: se True, pronuncia la frase
        """
        script = STATE_SCRIPTS.get(state_name)
        if script is None:
            return

        # Emoji
        self._emoji.set_emoji(script["emoji"])

        # LED
        led_method = getattr(self._led, script["led"], None)
        if led_method:
            led_method()

        # TTS (rotazione frasi)
        if speak and script["phrases"]:
            idx = self._phrase_counters.get(state_name, 0)
            phrase = script["phrases"][idx % len(script["phrases"])]
            self._tts.speak(phrase)
            self._phrase_counters[state_name] = idx + 1

    def announce_pod_count(self, count: int, total: int):
        """Annuncia il conteggio cialde."""
        if count % 5 == 0 and count > 0:
            remaining = total - count
            self._tts.speak(
                f"Ciruzzo ha messo {count} cialde! Ne mancano {remaining}!"
            )

    def announce_error(self, error_msg: str):
        """Annuncia un errore."""
        self._emoji.sad()
        self._led.red_blinking()
        self._tts.speak(f"Ue! {error_msg} Ma Ciruzzo ci riprova!")
