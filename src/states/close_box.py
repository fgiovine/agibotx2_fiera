"""Stato CLOSE_BOX: chiude la scatola piena con il coperchio.

Usa scatola con coperchio separato (raccomandato).
Operazione: pick coperchio -> place sopra scatola.
"""

import numpy as np

from src.states.base_state import BaseState, StateResult
from src.manipulation.predefined_poses import get_arm_pose
from src.utils.transforms import approach_vector_down


class CloseBoxState(BaseState):
    """Chiude la scatola piena mettendo il coperchio."""

    NAME = "CLOSE_BOX"

    def enter(self):
        self.commentary.play_state("CLOSE_BOX")
        self._logger.info("Chiusura scatola con coperchio...")
        self.mode_manager.ensure_manipulation()

    def execute(self) -> StateResult:
        side = self.arm_side

        # Il coperchio e posizionato accanto alla scatola sul tavolo
        box_pos = self.box_tracker.box_position
        if box_pos is None:
            self._logger.warn("Posizione scatola non nota")
            return StateResult(next_state="CARRY_BOX")

        # Stima posizione coperchio (accanto alla scatola)
        box_width = self.config.get('box.width_mm', 280.0) / 1000.0
        lid_pos = box_pos.copy()
        lid_pos[1] += box_width + 0.05  # Coperchio a lato

        R_down = approach_vector_down()

        # 1. Prendi il coperchio
        lid_above = lid_pos.copy()
        lid_above[2] += 0.10

        pre_grasp = get_arm_pose("lid_place", side)
        self.arm_controller.move_to_positions(pre_grasp, side, duration_s=1.5)

        self.gripper.open(side)

        q_above = self.ik_solver.solve(lid_above, R_down, side=side)
        if q_above is not None:
            self.arm_controller.move_to_positions(q_above, side, duration_s=1.5)

        q_lid = self.ik_solver.solve(lid_pos, R_down, side=side)
        if q_lid is not None:
            self.arm_controller.move_to_positions(q_lid, side, duration_s=1.0)

        self.gripper.close_for_pod(side)  # Presa gentile sul coperchio

        if not self.gripper.wait_for_grasp(side, timeout_s=2.0):
            self._logger.warn("Presa coperchio fallita, proseguo senza chiudere")
            self.gripper.open(side)
            self.arm_controller.move_to_positions(pre_grasp, side, duration_s=1.0)
            return StateResult(next_state="CARRY_BOX")

        # 2. Solleva coperchio
        if q_above is not None:
            self.arm_controller.move_to_positions(q_above, side, duration_s=1.0)

        # 3. Posiziona sopra la scatola
        box_top = box_pos.copy()
        box_height = self.config.get('box.height_mm', 60.0) / 1000.0
        box_top[2] += box_height + 0.05  # Sopra la scatola

        q_box_above = self.ik_solver.solve(box_top, R_down, side=side)
        if q_box_above is not None:
            self.arm_controller.move_to_positions(q_box_above, side, duration_s=1.5)

        # 4. Appoggia coperchio
        box_top[2] -= 0.03  # Scendi per appoggiare
        q_place = self.ik_solver.solve(box_top, R_down, side=side)
        if q_place is not None:
            self.arm_controller.move_to_positions(q_place, side, duration_s=1.0)

        self.gripper.open(side)

        # 5. Risali
        self.arm_controller.move_to_positions(pre_grasp, side, duration_s=1.0)

        self._logger.info("Scatola chiusa con coperchio!")

        return StateResult(next_state="CARRY_BOX")

    def exit(self):
        pass
