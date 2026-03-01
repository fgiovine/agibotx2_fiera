"""Riconoscimento vocale con Whisper per interazione in italiano.

Usa OpenAI Whisper (locale) per convertire speech-to-text.
Ascolta dal microfono del robot e trascrive in italiano.
"""

import threading
import time
import queue
from typing import Optional

import numpy as np

from rclpy.node import Node

from src.utils.config_loader import ConfigLoader

try:
    import whisper
except ImportError:
    whisper = None

try:
    import sounddevice as sd
except ImportError:
    sd = None


class SpeechRecognizer:
    """Riconoscimento vocale in italiano con Whisper."""

    def __init__(self, node: Node, config: ConfigLoader):
        self._node = node
        self._config = config
        self._logger = node.get_logger()

        # Parametri
        self._model_size = config.get('interaction.speech.whisper_model', 'base')
        self._language = config.get('interaction.speech.language', 'it')
        self._sample_rate = config.get('interaction.speech.sample_rate', 16000)
        self._silence_threshold = config.get(
            'interaction.speech.silence_threshold', 0.01
        )
        self._silence_duration_s = config.get(
            'interaction.speech.silence_duration_s', 1.5
        )
        self._max_record_s = config.get(
            'interaction.speech.max_record_s', 15.0
        )
        self._mic_device = config.get('interaction.speech.mic_device', None)

        # Modello Whisper
        self._model = None
        self._init_whisper()

        # Stato
        self._is_listening = False
        self._audio_queue: queue.Queue = queue.Queue()

    def _init_whisper(self):
        """Carica il modello Whisper."""
        if whisper is None:
            self._logger.warn(
                "Whisper non disponibile. Installa con: pip install openai-whisper"
            )
            return

        try:
            self._logger.info(f"Caricamento modello Whisper '{self._model_size}'...")
            self._model = whisper.load_model(self._model_size)
            self._logger.info("Modello Whisper caricato")
        except Exception as e:
            self._logger.error(f"Errore caricamento Whisper: {e}")

    def listen(self, timeout_s: float = None) -> Optional[str]:
        """Ascolta dal microfono e trascrive.

        Registra fino a che l'utente smette di parlare (silenzio)
        o fino al timeout.

        Args:
            timeout_s: timeout massimo registrazione

        Returns:
            Testo trascritto in italiano, o None
        """
        if self._model is None:
            self._logger.error("Whisper non disponibile")
            return None

        if sd is None:
            self._logger.error("sounddevice non disponibile")
            return None

        if timeout_s is None:
            timeout_s = self._max_record_s

        self._logger.info("Ascolto in corso...")
        self._is_listening = True

        try:
            audio = self._record_until_silence(timeout_s)
        except Exception as e:
            self._logger.error(f"Errore registrazione: {e}")
            self._is_listening = False
            return None

        self._is_listening = False

        if audio is None or len(audio) < self._sample_rate * 0.5:
            self._logger.info("Audio troppo corto, ignorato")
            return None

        # Trascrivi con Whisper
        self._logger.info("Trascrizione in corso...")
        try:
            audio_float = audio.astype(np.float32) / 32768.0
            result = self._model.transcribe(
                audio_float,
                language=self._language,
                fp16=False,
            )
            text = result["text"].strip()
            self._logger.info(f"Trascritto: \"{text}\"")
            return text if text else None
        except Exception as e:
            self._logger.error(f"Errore trascrizione: {e}")
            return None

    def _record_until_silence(self, timeout_s: float) -> Optional[np.ndarray]:
        """Registra audio fino a silenzio prolungato o timeout."""
        chunks = []
        silence_start = None
        start_time = time.time()
        chunk_duration = 0.1  # 100ms per chunk
        chunk_size = int(self._sample_rate * chunk_duration)

        def audio_callback(indata, frames, time_info, status):
            self._audio_queue.put(indata.copy())

        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype='int16',
            blocksize=chunk_size,
            device=self._mic_device,
            callback=audio_callback,
        ):
            while (time.time() - start_time) < timeout_s:
                try:
                    chunk = self._audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                chunks.append(chunk)

                # Controlla se c'e silenzio
                rms = np.sqrt(np.mean(chunk.astype(float) ** 2))
                is_silent = rms < self._silence_threshold * 32768

                if is_silent:
                    if silence_start is None:
                        silence_start = time.time()
                    elif (time.time() - silence_start) > self._silence_duration_s:
                        # Silenzio prolungato, fine registrazione
                        if len(chunks) > int(1.0 / chunk_duration):
                            break
                else:
                    silence_start = None

        if not chunks:
            return None

        return np.concatenate(chunks, axis=0).flatten()

    def listen_for_wake_word(self, wake_word: str = "ciruzzo",
                              timeout_s: float = 30.0) -> bool:
        """Ascolta continuamente per il wake word.

        Args:
            wake_word: parola di attivazione (case-insensitive)
            timeout_s: timeout massimo attesa

        Returns:
            True se il wake word e stato rilevato
        """
        start = time.time()
        while (time.time() - start) < timeout_s:
            text = self.listen(timeout_s=3.0)
            if text and wake_word.lower() in text.lower():
                self._logger.info(f"Wake word '{wake_word}' rilevato!")
                return True
        return False

    @property
    def is_listening(self) -> bool:
        return self._is_listening

    @property
    def is_available(self) -> bool:
        return self._model is not None and sd is not None
