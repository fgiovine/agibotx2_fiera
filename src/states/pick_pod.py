"""Stato PICK_POD: afferra una cialda dal tavolo.

Sequenza:
1. Braccio a pre_grasp (sopra tavolo)
2. Gripper aperto a 0.7
3. IK per posizione 3D cialda (approccio dall'alto)
4. Muovi a 5cm sopra cialda -> scendi a contatto
5. Chiudi gripper (effort 0.4, gentile)
6. Verifica presa (stato gripper: 2=stalled -> presa OK)
7. Solleva a pre_grasp
"""

import numpy as np

from src.states.base_state import BaseState, StateResult
from src.manipulation.predefined_poses import get_arm_pose
from src.utils.transforms import approach_vector_down


class PickPodState(BaseState):
    """Afferra una cialda dal tavolo."""

    NAME = "PICK_POD"
    MAX_RETRIES = 2

    def enter(self):
        self.commentary.play_state("PICK_POD")
        self._logger.info("Inizio sequenza pick cialda...")

        # Assicura modalita manipolazione
        self.mode_manager.ensure_manipulation()

    def execute(self) -> StateResult:
        side = self.arm_side
        data = self._context.state_data

        pod_position = data.get("pod_position")
        if pod_position is None:
            return StateResult(
                next_state="DETECT_PODS",
                error="Posizione cialda non disponibile"
            )

        # 1. Pre-grasp
        pre_grasp = get_arm_pose("pre_grasp", side)
        if not self.arm_controller.move_to_positions(
            pre_grasp, side, duration_s=1.5
        ):
            return StateResult(next_state="ERROR", error="Move a pre_grasp fallito")

        # 2. Apri gripper
        self.gripper.open(side)

        # 3. Calcola IK per posizione sopra la cialda
        approach_height = self.config.get('manipulation.pick.approach_height_m', 0.05)
        above_pos = pod_position.copy()
        above_pos[2] += approach_height

        R_down = approach_vector_down()
        q_above = self.ik_solver.solve(above_pos, R_down, side=side)

        if q_above is None:
            self._logger.error("IK fallito per posizione sopra cialda")
            return StateResult(
                next_state="ERROR",
                error="IK irraggiungibile per cialda"
            )

        # 4. Muovi sopra la cialda
        trajectory = self.trajectory_planner.plan(pre_grasp, q_above)
        if not self.arm_controller.execute_trajectory(trajectory, side):
            return StateResult(next_state="ERROR", error="Traiettoria fallita")

        # 5. Scendi a contatto
        q_contact = self.ik_solver.solve(pod_position, R_down, side=side)
        if q_contact is None:
            # Prova con un offset minimo
            pod_position[2] += 0.01
            q_contact = self.ik_solver.solve(pod_position, R_down, side=side)

        if q_contact is not None:
            self.arm_controller.move_to_positions(
                q_contact, side, duration_s=1.0
            )

        # 6. Chiudi gripper (gentile per cialda)
        self.gripper.close_for_pod(side)

        # 7. Verifica presa
        grasp_ok = self.gripper.wait_for_grasp(side, timeout_s=2.0)

        if not grasp_ok:
            self._logger.warn("Presa fallita! Riprovo detection...")
            self.gripper.open(side)
            # Torna a pre_grasp
            self.arm_controller.move_to_positions(
                pre_grasp, side, duration_s=1.0
            )
            return StateResult(next_state="DETECT_PODS")

        # 8. Solleva a pre_grasp
        self.arm_controller.move_to_positions(
            pre_grasp, side, duration_s=1.5
        )

        self._logger.info("Cialda afferrata con successo!")

        return StateResult(next_state="NAV_TO_BOX_TABLE")

    def exit(self):
        pass
