"""Text-to-Speech in italiano per Ciruzzo.

Sistema TTS a 3 livelli di fallback:
1. OpenAI TTS (voce naturale eccellente, richiede API key)
2. gTTS (Google TTS, buona qualita, richiede internet)
3. pyttsx3 (offline, usa voci di sistema)

L'audio viene riprodotto tramite gli speaker del robot.
"""

import os
import io
import time
import tempfile
import threading
from typing import Optional

from rclpy.node import Node

from src.utils.config_loader import ConfigLoader

# Import opzionali per i vari backend TTS
try:
    from openai import OpenAI as OpenAIClient
except ImportError:
    OpenAIClient = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sd = None
    sf = None

try:
    import pygame
except ImportError:
    pygame = None


class TTSBackend:
    OPENAI = "openai"
    GTTS = "gtts"
    PYTTSX3 = "pyttsx3"
    ROS_SERVICE = "ros_service"


class TTSClient:
    """Text-to-Speech in italiano con fallback multipli."""

    def __init__(self, node: Node, config: ConfigLoader):
        self._node = node
        self._config = config
        self._logger = node.get_logger()

        self._language = config.get('interaction.tts.language', 'it')
        self._volume = config.get('interaction.tts.volume', 0.8)
        self._is_speaking = False
        self._lock = threading.Lock()

        # Configurazione OpenAI TTS
        self._openai_voice = config.get('interaction.tts.openai_voice', 'nova')
        self._openai_model = config.get('interaction.tts.openai_model', 'tts-1')
        self._openai_speed = config.get('interaction.tts.openai_speed', 1.0)

        # Inizializza backend disponibili (in ordine di preferenza)
        self._openai_client: Optional[OpenAIClient] = None
        self._pyttsx3_engine = None
        self._active_backend: str = ""
        self._pygame_initialized = False

        self._init_backends()

    def _init_backends(self):
        """Inizializza i backend TTS disponibili."""
        # 1. Prova OpenAI TTS
        if OpenAIClient is not None:
            api_key = self._config.get('interaction.chat.openai_api_key', '')
            if not api_key:
                api_key = os.environ.get('OPENAI_API_KEY', '')
                if not api_key:
                    env_path = os.path.join(
                        os.path.dirname(__file__), '..', '..', '.env'
                    )
                    if os.path.exists(env_path):
                        with open(env_path) as f:
                            for line in f:
                                if line.strip().startswith('OPENAI_API_KEY='):
                                    api_key = line.strip().split('=', 1)[1]
                                    break

            if api_key:
                try:
                    self._openai_client = OpenAIClient(api_key=api_key)
                    self._active_backend = TTSBackend.OPENAI
                    self._logger.info(
                        "TTS: OpenAI attivo (voce italiana di alta qualita)"
                    )
                except Exception as e:
                    self._logger.warn(f"TTS OpenAI non disponibile: {e}")

        # 2. Prova gTTS
        if not self._active_backend and gTTS is not None:
            self._active_backend = TTSBackend.GTTS
            self._logger.info("TTS: Google TTS attivo (richiede internet)")

        # 3. Prova pyttsx3
        if not self._active_backend and pyttsx3 is not None:
            try:
                self._pyttsx3_engine = pyttsx3.init()
                # Cerca voce italiana
                for voice in self._pyttsx3_engine.getProperty('voices'):
                    if 'italian' in voice.name.lower() or 'it' in voice.id.lower():
                        self._pyttsx3_engine.setProperty('voice', voice.id)
                        self._logger.info(f"TTS pyttsx3: voce italiana trovata: {voice.name}")
                        break
                self._pyttsx3_engine.setProperty(
                    'rate', self._config.get('interaction.tts.rate', 150)
                )
                self._pyttsx3_engine.setProperty('volume', self._volume)
                self._active_backend = TTSBackend.PYTTSX3
                self._logger.info("TTS: pyttsx3 attivo (offline)")
            except Exception as e:
                self._logger.warn(f"TTS pyttsx3 non disponibile: {e}")

        # Inizializza pygame per riproduzione audio (OpenAI e gTTS)
        if self._active_backend in (TTSBackend.OPENAI, TTSBackend.GTTS):
            self._init_audio_player()

        if not self._active_backend:
            self._logger.error(
                "NESSUN backend TTS disponibile! "
                "Installa: pip install openai gtts pyttsx3"
            )

        self._logger.info(f"TTS backend attivo: {self._active_backend or 'NESSUNO'}")

    def _init_audio_player(self):
        """Inizializza il player audio."""
        if pygame is not None:
            try:
                pygame.mixer.init()
                self._pygame_initialized = True
                return
            except Exception:
                pass

        if sd is None or sf is None:
            self._logger.warn(
                "Ne pygame ne sounddevice disponibili per riproduzione audio. "
                "Installa: pip install pygame oppure pip install sounddevice soundfile"
            )

    def speak(self, text: str, blocking: bool = False) -> bool:
        """Pronuncia una frase in italiano.

        Args:
            text: testo da pronunciare
            blocking: se True, attende che finisca

        Returns:
            True se avviato con successo
        """
        if not text or not self._active_backend:
            return False

        self._logger.info(f"TTS [{self._active_backend}]: \"{text}\"")

        if blocking:
            return self._speak_sync(text)

        # Non-blocking: avvia in un thread
        thread = threading.Thread(target=self._speak_sync, args=(text,))
        thread.daemon = True
        thread.start()
        return True

    def _speak_sync(self, text: str) -> bool:
        """Pronuncia in modo sincrono."""
        with self._lock:
            self._is_speaking = True

        try:
            if self._active_backend == TTSBackend.OPENAI:
                return self._speak_openai(text)
            elif self._active_backend == TTSBackend.GTTS:
                return self._speak_gtts(text)
            elif self._active_backend == TTSBackend.PYTTSX3:
                return self._speak_pyttsx3(text)
            return False
        except Exception as e:
            self._logger.error(f"Errore TTS: {e}")
            # Prova fallback
            return self._speak_fallback(text)
        finally:
            self._is_speaking = False

    def _speak_openai(self, text: str) -> bool:
        """TTS con OpenAI - voce italiana naturale e di alta qualita."""
        try:
            response = self._openai_client.audio.speech.create(
                model=self._openai_model,
                voice=self._openai_voice,
                input=text,
                speed=self._openai_speed,
            )

            # Salva in file temporaneo e riproduci
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                f.write(response.content)
                tmp_path = f.name

            self._play_audio_file(tmp_path)
            os.unlink(tmp_path)
            return True

        except Exception as e:
            self._logger.warn(f"OpenAI TTS fallito: {e}, provo fallback")
            return self._speak_fallback(text)

    def _speak_gtts(self, text: str) -> bool:
        """TTS con Google (gTTS) - buona qualita italiano, serve internet."""
        try:
            tts = gTTS(text=text, lang=self._language, slow=False)

            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                tts.save(f.name)
                tmp_path = f.name

            self._play_audio_file(tmp_path)
            os.unlink(tmp_path)
            return True

        except Exception as e:
            self._logger.warn(f"gTTS fallito: {e}, provo fallback")
            return self._speak_fallback(text)

    def _speak_pyttsx3(self, text: str) -> bool:
        """TTS con pyttsx3 - offline, usa voci di sistema."""
        try:
            self._pyttsx3_engine.say(text)
            self._pyttsx3_engine.runAndWait()
            return True
        except Exception as e:
            self._logger.error(f"pyttsx3 fallito: {e}")
            return False

    def _speak_fallback(self, text: str) -> bool:
        """Tenta i backend di fallback in ordine."""
        if self._active_backend != TTSBackend.GTTS and gTTS is not None:
            try:
                return self._speak_gtts(text)
            except Exception:
                pass

        if self._active_backend != TTSBackend.PYTTSX3 and self._pyttsx3_engine is not None:
            try:
                return self._speak_pyttsx3(text)
            except Exception:
                pass

        self._logger.error(f"Tutti i TTS falliti per: \"{text}\"")
        return False

    def _play_audio_file(self, file_path: str):
        """Riproduce un file audio (mp3/wav)."""
        if self._pygame_initialized:
            try:
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.set_volume(self._volume)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                return
            except Exception as e:
                self._logger.warn(f"pygame playback fallito: {e}")

        if sd is not None and sf is not None:
            try:
                data, samplerate = sf.read(file_path)
                sd.play(data, samplerate)
                sd.wait()
                return
            except Exception as e:
                self._logger.warn(f"sounddevice playback fallito: {e}")

        self._logger.error("Nessun player audio disponibile!")

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def active_backend(self) -> str:
        return self._active_backend

    def destroy(self):
        if self._pygame_initialized:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
        if self._pyttsx3_engine is not None:
            try:
                self._pyttsx3_engine.stop()
            except Exception:
                pass
