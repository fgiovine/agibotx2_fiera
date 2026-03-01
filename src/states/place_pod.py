"""Stato PLACE_POD_IN_BOX: deposita cialda nella griglia 4x5 della scatola.

Sequenza:
1. Braccio a pre_place_box (sopra scatola)
2. IK per slot griglia corrente
3. Scendi in scatola -> apri gripper -> risali
"""

import numpy as np

from src.states.base_state import BaseState, StateResult
from src.manipulation.predefined_poses import get_arm_pose
from src.utils.transforms import approach_vector_down


class PlacePodState(BaseState):
    """Deposita una cialda nella scatola."""

    NAME = "PLACE_POD_IN_BOX"

    def enter(self):
        self.commentary.play_state("PLACE_POD_IN_BOX")
        self._logger.info("Deposito cialda nella scatola...")

        # Modalita manipolazione
        self.mode_manager.ensure_manipulation()

    def execute(self) -> StateResult:
        side = self.arm_side

        # Verifica che abbiamo ancora la cialda
        if not self.gripper.has_object(side):
            self._logger.warn("Cialda persa durante il trasporto!")
            return StateResult(next_state="IDLE")

        # 1. Pre-place sopra la scatola
        pre_place = get_arm_pose("pre_place_box", side)
        if not self.arm_controller.move_to_positions(
            pre_place, side, duration_s=1.5
        ):
            return StateResult(next_state="ERROR", error="Move a pre_place fallito")

        # 2. Calcola posizione del prossimo slot
        slot_pos = self.box_tracker.get_next_place_position()
        if slot_pos is None:
            self._logger.warn("Scatola piena o posizione non disponibile")
            return StateResult(next_state="CHECK_BOX")

        # Aggiungi offset altezza per scendere nella scatola
        place_height = self.config.get('manipulation.place.release_height_m', 0.02)
        slot_above = slot_pos.copy()
        slot_above[2] += self.config.get('manipulation.place.pre_place_height_m', 0.10)

        R_down = approach_vector_down()

        # 3. IK per posizione sopra lo slot
        q_above = self.ik_solver.solve(slot_above, R_down, side=side)
        if q_above is None:
            self._logger.error("IK fallito per slot sopra scatola")
            return StateResult(
                next_state="ERROR",
                error="IK irraggiungibile per scatola"
            )

        # Muovi sopra lo slot
        trajectory = self.trajectory_planner.plan(pre_place, q_above)
        if not self.arm_controller.execute_trajectory(trajectory, side):
            return StateResult(next_state="ERROR", error="Traiettoria place fallita")

        # 4. Scendi allo slot
        slot_place = slot_pos.copy()
        slot_place[2] += place_height
        q_place = self.ik_solver.solve(slot_place, R_down, side=side)

        if q_place is not None:
            self.arm_controller.move_to_positions(
                q_place, side, duration_s=1.0
            )

        # 5. Rilascia cialda
        self.gripper.open(side)
        import time
        time.sleep(0.3)  # Breve pausa per stabilizzazione

        # 6. Risali
        self.arm_controller.move_to_positions(pre_place, side, duration_s=1.0)

        # 7. Aggiorna contatore
        self.box_tracker.add_pod()

        self._logger.info(
            f"Cialda depositata! "
            f"({self.box_tracker.pod_count}/{self.config.get('pods.max_per_box', 20)})"
        )

        # Annuncia progresso
        self.commentary.announce_pod_count(
            self.box_tracker.pod_count,
            self.config.get('pods.max_per_box', 20)
        )

        return StateResult(next_state="CHECK_BOX")

    def exit(self):
        pass


class CheckBoxState(BaseState):
    """Controlla se la scatola e piena."""

    NAME = "CHECK_BOX"

    def enter(self):
        pass

    def execute(self) -> StateResult:
        if self.box_tracker.is_full:
            self._logger.info("Scatola piena! Procedura di chiusura...")
            self.commentary.play_state("BOX_FULL")
            return StateResult(next_state="CLOSE_BOX")

        # Scatola non piena, torna a cercare cialde
        return StateResult(next_state="IDLE")

    def exit(self):
        pass
