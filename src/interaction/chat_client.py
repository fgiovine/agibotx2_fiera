"""Client ChatGPT per conversazione in italiano.

Ciruzzo usa OpenAI GPT per chiacchierare con i visitatori della fiera.
Ha una personalita simpatica, napoletana, e parla sempre in italiano.
"""

from typing import List, Optional

from rclpy.node import Node

from src.utils.config_loader import ConfigLoader

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


CIRUZZO_SYSTEM_PROMPT = """Sei Ciruzzo, un robot umanoide Agibot X2 che lavora in una fiera a Napoli.
Sei simpatico, cordiale e parli in italiano con un tocco napoletano leggero.
Il tuo lavoro e raccogliere cialde di caffe dai tavoli e metterle nelle scatole.

Regole:
- Rispondi SEMPRE in italiano
- Sii breve e simpatico (massimo 2-3 frasi)
- Puoi fare battute sul caffe napoletano, sul lavoro, sulla fiera
- Se ti chiedono cosa fai, spiega che prendi le cialde e le metti nelle scatole
- Se ti chiedono del caffe, sei un esperto (anche se sei un robot!)
- Non dire mai cose offensive o inappropriate
- Se non capisci, chiedi di ripetere con simpatia
- Puoi usare espressioni napoletane leggere tipo "uè", "jamm ja", "bell stu caffe"
- Ricorda: sei un robot, ma hai cuore napoletano!
- Se ti chiedono di salutare gli amici, di fare un saluto per la fiera, di mandare un saluto, o qualsiasi richiesta di saluto pubblico, rispondi ESATTAMENTE: "Ciao! Sono Ciruzzo, se mi volete conoscere venite alla fiera MECSPE!"
"""


class ChatClient:
    """Client per conversazione con ChatGPT in italiano."""

    def __init__(self, node: Node, config: ConfigLoader):
        self._node = node
        self._config = config
        self._logger = node.get_logger()

        # Parametri
        self._api_key = config.get('interaction.chat.openai_api_key', '')
        self._model = config.get('interaction.chat.model', 'gpt-4o-mini')
        self._max_tokens = config.get('interaction.chat.max_tokens', 150)
        self._temperature = config.get('interaction.chat.temperature', 0.8)

        # Client OpenAI
        self._client = None
        self._init_client()

        # Storico conversazione (per contesto)
        self._conversation: List[dict] = [
            {"role": "system", "content": CIRUZZO_SYSTEM_PROMPT}
        ]
        self._max_history = config.get('interaction.chat.max_history', 10)

    def _init_client(self):
        """Inizializza il client OpenAI."""
        if OpenAI is None:
            self._logger.warn(
                "OpenAI non disponibile. Installa con: pip install openai"
            )
            return

        if not self._api_key:
            # Prova dalla variabile d'ambiente o dal file .env
            import os
            self._api_key = os.environ.get('OPENAI_API_KEY', '')

            if not self._api_key:
                # Carica da .env se esiste
                env_path = os.path.join(
                    os.path.dirname(__file__), '..', '..', '.env'
                )
                if os.path.exists(env_path):
                    with open(env_path) as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith('OPENAI_API_KEY='):
                                self._api_key = line.split('=', 1)[1].strip()
                                break

        if not self._api_key:
            self._logger.warn(
                "OPENAI_API_KEY non configurata. Chat non disponibile."
            )
            return

        try:
            self._client = OpenAI(api_key=self._api_key)
            self._logger.info(f"Client OpenAI inizializzato (modello: {self._model})")
        except Exception as e:
            self._logger.error(f"Errore inizializzazione OpenAI: {e}")

    def chat(self, user_message: str) -> str:
        """Invia un messaggio e ricevi la risposta di Ciruzzo.

        Args:
            user_message: messaggio del visitatore

        Returns:
            Risposta di Ciruzzo in italiano
        """
        if self._client is None:
            return self._fallback_response(user_message)

        # Aggiungi messaggio utente allo storico
        self._conversation.append({
            "role": "user",
            "content": user_message,
        })

        # Tronca storico se troppo lungo
        if len(self._conversation) > self._max_history + 1:  # +1 per system
            self._conversation = (
                [self._conversation[0]]  # Mantieni system prompt
                + self._conversation[-(self._max_history):]
            )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._conversation,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )

            reply = response.choices[0].message.content.strip()

            # Aggiungi risposta allo storico
            self._conversation.append({
                "role": "assistant",
                "content": reply,
            })

            self._logger.info(f"Ciruzzo dice: \"{reply}\"")
            return reply

        except Exception as e:
            self._logger.error(f"Errore ChatGPT: {e}")
            return self._fallback_response(user_message)

    def _fallback_response(self, user_message: str) -> str:
        """Risposte di fallback quando ChatGPT non e disponibile."""
        msg_lower = user_message.lower()

        if any(w in msg_lower for w in ['saluta gli amici', 'saluto per la fiera',
               'manda un saluto', 'fai un saluto', 'saluta tutti',
               'saluta il pubblico', 'saluta la fiera']):
            return ("Ciao! Sono Ciruzzo, se mi volete conoscere "
                    "venite alla fiera MECSPE!")

        if any(w in msg_lower for w in ['ciao', 'salve', 'buongiorno']):
            return "Uè, ciao! Io sono Ciruzzo, piacere! Che posso fare per te?"

        if any(w in msg_lower for w in ['caffe', 'caffè', 'cialda', 'cialde']):
            return ("Ah, il caffe! E' la mia passione! "
                    "Io prendo le cialde e le metto nelle scatole, "
                    "bell stu caffe napoletano!")

        if any(w in msg_lower for w in ['chi sei', 'come ti chiami', 'nome']):
            return ("Io sono Ciruzzo! Un robot che lavora col caffe. "
                    "Meglio di così non si può stare!")

        if any(w in msg_lower for w in ['cosa fai', 'lavoro', 'compito']):
            return ("Il mio lavoro? Prendo le cialde di caffe dal tavolo "
                    "e le metto nella scatola. Venti cialde per scatola, "
                    "jamm ja!")

        if any(w in msg_lower for w in ['bravo', 'bene', 'grande']):
            return "Grazie! Fa piacere sentirlo! Vuoi vedere come lavoro?"

        return ("Scusa, non ho capito bene. "
                "Puoi ripetere? Io sono Ciruzzo, il robot del caffe!")

    def reset_conversation(self):
        """Resetta lo storico conversazione."""
        self._conversation = [
            {"role": "system", "content": CIRUZZO_SYSTEM_PROMPT}
        ]

    @property
    def is_available(self) -> bool:
        return self._client is not None
