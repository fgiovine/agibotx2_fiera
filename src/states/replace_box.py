"""Stato REPLACE_BOX: prende una scatola vuota e la posiziona sul tavolo centrale.

Sostituisce la scatola piena appena rimossa con una scatola vuota.
"""

import numpy as np

from src.states.base_state import BaseState, StateResult
from src.manipulation.predefined_poses import get_arm_pose
from src.utils.transforms import approach_vector_down


class GetEmptyBoxState(BaseState):
    """Prende una scatola vuota dalla pila."""

    NAME = "GET_EMPTY_BOX"

    def enter(self):
        self.commentary.play_state("GET_EMPTY_BOX")
        self._logger.info("Prendo una scatola vuota...")
        self.mode_manager.ensure_manipulation()

    def execute(self) -> StateResult:
        side = self.arm_side

        box_table = self.table_detector.get_table("box_table")
        if box_table is None:
            return StateResult(
                next_state="ERROR",
                error="Tavolo scatola non trovato"
            )

        R_down = approach_vector_down()

        # La pila di scatole vuote e sul tavolo centrale
        # (in una posizione nota, leggermente spostata rispetto alla scatola attiva)
        pile_pos = box_table.center.copy()
        box_width = self.config.get('box.width_mm', 280.0) / 1000.0
        pile_pos[1] += box_width + 0.10  # A lato della posizione principale

        # 1. Posiziona sopra la pila
        pre_grasp = get_arm_pose("box_grasp", side)
        self.arm_controller.move_to_positions(pre_grasp, side, duration_s=1.5)
        self.gripper.open(side, position=0.9)

        # 2. Scendi sulla scatola vuota
        q_grasp = self.ik_solver.solve(pile_pos, R_down, side=side)
        if q_grasp is not None:
            self.arm_controller.move_to_positions(q_grasp, side, duration_s=1.5)

        # 3. Afferra
        self.gripper.close_for_box(side)

        if not self.gripper.wait_for_grasp(side, timeout_s=3.0):
            self._logger.warn("Nessuna scatola vuota disponibile o presa fallita")
            self.gripper.open(side)
            self.arm_controller.move_to_positions(pre_grasp, side, duration_s=1.0)
            # Continua comunque, potrebbe non esserci una scatola vuota
            return StateResult(next_state="PLACE_EMPTY_BOX")

        # 4. Solleva
        carry = get_arm_pose("box_carry", side)
        self.arm_controller.move_to_positions(carry, side, duration_s=1.5)

        self._logger.info("Scatola vuota afferrata")
        return StateResult(next_state="PLACE_EMPTY_BOX")

    def exit(self):
        pass


class PlaceEmptyBoxState(BaseState):
    """Posiziona la scatola vuota al centro del tavolo."""

    NAME = "PLACE_EMPTY_BOX"

    def enter(self):
        self._logger.info("Posiziono la scatola vuota...")

    def execute(self) -> StateResult:
        side = self.arm_side

        box_table = self.table_detector.get_table("box_table")
        if box_table is None:
            return StateResult(next_state="ERROR", error="Tavolo non trovato")

        R_down = approach_vector_down()

        # Posizione centrale del tavolo per la nuova scatola
        place_pos = box_table.center.copy()
        place_pos[2] += 0.10

        carry = get_arm_pose("box_carry", side)

        # Scendi per depositare
        q_place = self.ik_solver.solve(place_pos, R_down, side=side)
        if q_place is not None:
            self.arm_controller.move_to_positions(q_place, side, duration_s=2.0)

        place_pos[2] -= 0.05
        q_down = self.ik_solver.solve(place_pos, R_down, side=side)
        if q_down is not None:
            self.arm_controller.move_to_positions(q_down, side, duration_s=1.0)

        # Rilascia
        self.gripper.open(side)

        import time
        time.sleep(0.3)

        # Risali
        self.arm_controller.move_to_positions(carry, side, duration_s=1.0)

        # Reset box tracker per nuova scatola
        self.box_tracker.reset()
        self.box_tracker.set_box_position(place_pos)

        self._logger.info("Scatola vuota posizionata! Torno a IDLE.")

        return StateResult(next_state="IDLE")

    def exit(self):
        pass
