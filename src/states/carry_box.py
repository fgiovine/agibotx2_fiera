"""Stato CARRY_BOX: prende la scatola piena e la trasporta al tavolo DX.

Sequenza:
1. Afferra scatola (effort forte)
2. Solleva
3. Transita a navigazione verso tavolo scatole piene
"""

import numpy as np

from src.states.base_state import BaseState, StateResult
from src.manipulation.predefined_poses import get_arm_pose
from src.utils.transforms import approach_vector_down


class CarryBoxState(BaseState):
    """Prende la scatola piena e prepara per il trasporto."""

    NAME = "CARRY_BOX"

    def enter(self):
        self.commentary.play_state("CARRY_BOX")
        self._logger.info("Prendo la scatola piena per trasportarla...")
        self.mode_manager.ensure_manipulation()

    def execute(self) -> StateResult:
        side = self.arm_side

        box_pos = self.box_tracker.box_position
        if box_pos is None:
            self._logger.error("Posizione scatola non nota")
            return StateResult(
                next_state="ERROR",
                error="Posizione scatola sconosciuta"
            )

        R_down = approach_vector_down()

        # 1. Posiziona sopra la scatola per la presa
        box_grasp = get_arm_pose("box_grasp", side)
        self.arm_controller.move_to_positions(box_grasp, side, duration_s=1.5)
        self.gripper.open(side, position=0.9)  # Apri di piu per la scatola

        # 2. Scendi sulla scatola
        # Presa laterale sulla scatola
        grasp_pos = box_pos.copy()
        box_height = self.config.get('box.height_mm', 60.0) / 1000.0
        grasp_pos[2] += box_height / 2  # Meta altezza scatola

        q_grasp = self.ik_solver.solve(grasp_pos, R_down, side=side)
        if q_grasp is not None:
            self.arm_controller.move_to_positions(q_grasp, side, duration_s=1.5)

        # 3. Chiudi gripper (effort forte per scatola)
        self.gripper.close_for_box(side)

        if not self.gripper.wait_for_grasp(side, timeout_s=3.0):
            self._logger.error("Presa scatola fallita!")
            self.gripper.open(side)
            return StateResult(
                next_state="ERROR",
                error="Impossibile afferrare la scatola"
            )

        # 4. Solleva
        carry_pose = get_arm_pose("box_carry", side)
        self.arm_controller.move_to_positions(carry_pose, side, duration_s=2.0)

        self._logger.info("Scatola afferrata, pronto per il trasporto")

        # Transita a navigazione verso tavolo scatole piene
        return StateResult(next_state="NAV_TO_FULL_TABLE")

    def exit(self):
        pass


class PlaceFullBoxState(BaseState):
    """Deposita la scatola piena sul tavolo DX."""

    NAME = "PLACE_FULL_BOX"

    def enter(self):
        self.commentary.play_state("PLACE_FULL_BOX")
        self._logger.info("Deposito scatola piena sul tavolo...")
        self.mode_manager.ensure_manipulation()

    def execute(self) -> StateResult:
        side = self.arm_side

        full_table = self.table_detector.get_table("full_table")
        if full_table is None:
            return StateResult(
                next_state="ERROR",
                error="Tavolo scatole piene non trovato"
            )

        R_down = approach_vector_down()

        # Posizione di deposito sul tavolo
        place_pos = full_table.center.copy()
        place_pos[2] += 0.10  # Sopra il tavolo

        pre_place = get_arm_pose("box_carry", side)

        # Scendi per depositare
        q_place = self.ik_solver.solve(place_pos, R_down, side=side)
        if q_place is not None:
            self.arm_controller.move_to_positions(q_place, side, duration_s=2.0)

        # Appoggia
        place_pos[2] -= 0.05
        q_down = self.ik_solver.solve(place_pos, R_down, side=side)
        if q_down is not None:
            self.arm_controller.move_to_positions(q_down, side, duration_s=1.0)

        # Rilascia
        self.gripper.open(side)

        import time
        time.sleep(0.5)

        # Risali
        self.arm_controller.move_to_positions(pre_place, side, duration_s=1.5)

        self._logger.info("Scatola piena depositata!")

        return StateResult(next_state="NAV_BACK_TO_BOX_TABLE")

    def exit(self):
        pass
