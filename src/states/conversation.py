"""Stato CONVERSATION: Ciruzzo chiacchiera con i visitatori.

Attivato quando qualcuno chiama "Ciruzzo" e sceglie "parlare".
Usa ChatGPT per conversare in italiano con tocco napoletano.
"""

from src.states.base_state import BaseState, StateResult
from src.interaction.voice_interaction import UserIntent


class WakeWordDetectedState(BaseState):
    """Stato intermedio: Ciruzzo e stato chiamato, chiede cosa vuole l'utente."""

    NAME = "WAKE_WORD_DETECTED"

    def enter(self):
        self._logger.info("Qualcuno ha chiamato Ciruzzo!")

        # Ferma qualsiasi movimento
        self.locomotion.stop()

        # Guarda verso il pubblico
        from src.manipulation.predefined_poses import get_head_pose
        try:
            look_audience = get_head_pose("look_audience")
        except KeyError:
            pass

    def execute(self) -> StateResult:
        voice = self._context.voice_interaction

        # Chiedi cosa vuole l'utente
        intent = voice.handle_activation()

        if intent == UserIntent.PARLARE:
            self._logger.info("L'utente vuole parlare -> CONVERSATION")
            return StateResult(next_state="CONVERSATION")

        elif intent == UserIntent.PACCO:
            self._logger.info("L'utente vuole il pacco -> IDLE")
            self.commentary.play_state("IDLE")
            return StateResult(next_state="IDLE")

        else:
            # Non ha capito o l'utente se n'e andato
            self._logger.info("Intento non chiaro, torno a IDLE")
            return StateResult(next_state="IDLE")

    def exit(self):
        pass


class ConversationState(BaseState):
    """Stato di conversazione: Ciruzzo chiacchiera usando ChatGPT."""

    NAME = "CONVERSATION"

    def enter(self):
        self._logger.info("Ciruzzo entra in modalita conversazione!")
        self.commentary.play_state("CONVERSATION")

        # Posa amichevole
        from src.manipulation.predefined_poses import get_arm_pose
        try:
            present = get_arm_pose("present", self.arm_side)
            self.arm_controller.move_to_positions(
                present, self.arm_side, duration_s=2.0
            )
        except Exception:
            pass

    def execute(self) -> StateResult:
        voice = self._context.voice_interaction

        # Avvia la conversazione (blocca fino a fine)
        next_state = voice.start_conversation()

        self._logger.info(f"Conversazione terminata, ritorno a {next_state}")
        return StateResult(next_state=next_state)

    def exit(self):
        # Torna in posizione home
        from src.manipulation.predefined_poses import get_arm_pose
        try:
            home = get_arm_pose("home", self.arm_side)
            self.arm_controller.move_to_positions(
                home, self.arm_side, duration_s=2.0
            )
        except Exception:
            pass
