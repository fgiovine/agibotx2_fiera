"""Stato ERROR_RECOVERY: gestione errori e recupero.

Quando si verifica un errore:
1. Ferma il robot
2. Rilascia gripper
3. Torna in posizione sicura
4. Tenta di riprendere da uno stato sicuro
"""

import time

from src.states.base_state import BaseState, StateResult
from src.manipulation.predefined_poses import get_arm_pose


class ErrorRecoveryState(BaseState):
    """Gestione errori e tentativo di recupero."""

    NAME = "ERROR"
    MAX_CONSECUTIVE_ERRORS = 3
    RECOVERY_DELAY_S = 3.0

    def __init__(self, context):
        super().__init__(context)
        self._error_count = 0

    def enter(self):
        error_msg = self._context.state_data.get("error", "Errore sconosciuto")
        self._logger.error(f"ERRORE: {error_msg}")
        self.commentary.play_state("ERROR")
        self.commentary.announce_error(error_msg)

        self._error_count += 1

    def execute(self) -> StateResult:
        # Troppi errori consecutivi -> fermati
        if self._error_count >= self.MAX_CONSECUTIVE_ERRORS:
            self._logger.error(
                f"Troppi errori consecutivi ({self._error_count}). "
                f"Robot in attesa intervento operatore."
            )
            # Attendi intervento manuale
            time.sleep(10.0)
            self._error_count = 0
            return StateResult(next_state="SCAN_TABLES")

        # 1. Ferma movimento
        self.locomotion.stop()

        # 2. Rilascia gripper se tiene qualcosa
        side = self.arm_side
        self.gripper.open(side)
        time.sleep(0.5)

        # 3. Braccio in posizione sicura
        try:
            home = get_arm_pose("home", side)
            self.arm_controller.move_to_positions(home, side, duration_s=2.0)
        except Exception as e:
            self._logger.error(f"Impossibile tornare a home: {e}")

        # 4. Modalita stand
        self.mode_manager.ensure_stand()

        # 5. Pausa per recovery
        time.sleep(self.RECOVERY_DELAY_S)

        # 6. Tenta di riprendere
        self._logger.info("Tentativo di recupero...")

        # Verifica se i tavoli sono ancora visibili
        if not self.table_detector.all_tables_detected:
            return StateResult(next_state="SCAN_TABLES")

        # Se il gripper non tiene nulla, torna a IDLE
        return StateResult(next_state="IDLE")

    def exit(self):
        pass

    def reset_error_count(self):
        """Chiamato quando uno stato completa con successo."""
        self._error_count = 0
