"""Stato IDLE: ascolto wake word "Ciruzzo".

Per ora il robot fa SOLO conversazione (test fiera).
La parte pick & place (cialde) e commentata.
- Se sente "Ciruzzo" -> WAKE_WORD_DETECTED (chiede pacco o parlare)
"""

import time

from src.states.base_state import BaseState, StateResult


class IdleState(BaseState):
    """Stato di attesa: monitora cialde e ascolta per il wake word."""

    NAME = "IDLE"
    POLL_INTERVAL_S = 1.0

    def __init__(self, context):
        super().__init__(context)
        self._wake_word_counter = 0
        self._wake_word_check_every = 3  # Controlla wake word ogni N cicli

    def enter(self):
        self.commentary.play_state("IDLE")
        self._logger.info("Stato IDLE: Ciruzzo in attesa...")
        self._wake_word_counter = 0

        # Braccio in posizione sicura
        from src.manipulation.predefined_poses import get_arm_pose
        home = get_arm_pose("home", self.arm_side)
        self.arm_controller.move_to_positions(home, self.arm_side, duration_s=2.0)

        # Guarda il tavolo cialde
        from src.manipulation.predefined_poses import get_head_pose
        try:
            look_down = get_head_pose("look_down")
        except KeyError:
            pass

    def execute(self) -> StateResult:
        """Ascolta per 'Ciruzzo' (solo conversazione per ora)."""

        # --- Controlla wake word "Ciruzzo" ---
        self._wake_word_counter += 1
        if (self._wake_word_counter % self._wake_word_check_every == 0
                and self.voice_interaction.is_available):
            if self.voice_interaction.check_wake_word():
                self._logger.info("Qualcuno ha chiamato Ciruzzo!")
                return StateResult(next_state="WAKE_WORD_DETECTED")

        # --- Rileva cialde (COMMENTATO: solo conversazione per test fiera) ---
        # rgb = self.camera.get_rgb()
        # if rgb is None:
        #     time.sleep(self.POLL_INTERVAL_S)
        #     return StateResult(next_state=self.NAME)
        #
        # detections = self.pod_detector.detect(rgb)
        #
        # if detections:
        #     self._logger.info(f"Rilevate {len(detections)} cialde!")
        #
        #     # Verifica se siamo abbastanza vicini al tavolo
        #     pod_table = self.table_detector.get_table("pod_table")
        #     if pod_table is not None:
        #         from src.utils.transforms import distance_2d
        #         dist = distance_2d(
        #             self.position_tracker.position, pod_table.center
        #         )
        #         approach_dist = self.config.get('tables.approach_distance_m', 0.45)
        #
        #         if dist > approach_dist + 0.1:
        #             return StateResult(next_state="NAV_TO_POD_TABLE")
        #
        #     return StateResult(next_state="DETECT_PODS")

        # Nessuna azione cialde, solo attesa wake word
        time.sleep(self.POLL_INTERVAL_S)
        return StateResult(next_state=self.NAME)

    def exit(self):
        pass
