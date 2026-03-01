"""Stato NAVIGATE: cammina tra i tavoli con visual servoing.

Gestisce la navigazione verso un tavolo target usando:
- Fase grossolana (0.3 m/s) fino a 1m
- Fase fine (0.15 m/s) con visual servoing
- Allineamento angolare finale
- Stop a 0.45m dal bordo
"""

import time

import rclpy

from src.states.base_state import BaseState, StateResult
from src.navigation.approach_planner import ApproachPhase
from src.manipulation.predefined_poses import get_arm_pose


class NavigateState(BaseState):
    """Navigazione verso un tavolo target."""

    NAME = "NAVIGATE"
    UPDATE_RATE_HZ = 10
    TIMEOUT_S = 60.0

    def __init__(self, context, target_table_name: str, next_state: str):
        """
        Args:
            context: contesto demo
            target_table_name: nome del tavolo target
            next_state: stato successivo dopo navigazione riuscita
        """
        super().__init__(context)
        self._target_table_name = target_table_name
        self._next_state = next_state
        self._start_time = 0.0

    def enter(self):
        # Assicura modalita locomotion
        self.mode_manager.ensure_locomotion()

        # Braccio in posizione sicura per camminata
        walk_safe = get_arm_pose("walk_safe", self.arm_side)
        self.arm_controller.move_to_positions(
            walk_safe, self.arm_side, duration_s=1.5
        )

        # Inizia approccio
        table = self.table_detector.get_table(self._target_table_name)
        if table is None:
            self._logger.error(
                f"Tavolo {self._target_table_name} non trovato!"
            )
            return

        self.approach_planner.start_approach(table)
        self._start_time = time.time()

        self._logger.info(
            f"Navigazione verso {self._target_table_name} iniziata"
        )

    def execute(self) -> StateResult:
        # Timeout
        if (time.time() - self._start_time) > self.TIMEOUT_S:
            self.locomotion.stop()
            return StateResult(
                next_state="ERROR",
                error=f"Timeout navigazione verso {self._target_table_name}"
            )

        # Aggiorna posizione
        robot_pos = self.position_tracker.position
        robot_yaw = self.position_tracker.yaw

        # Aggiorna visual servoing con nuova osservazione del tavolo
        table = self.table_detector.get_table(self._target_table_name)
        table_pos = table.center if table is not None else None

        phase = self.approach_planner.update(robot_pos, robot_yaw, table_pos)

        if phase == ApproachPhase.DONE:
            self._logger.info(
                f"Arrivato a {self._target_table_name}!"
            )
            self.locomotion.stop()
            return StateResult(next_state=self._next_state)

        # Controlla spostamento tavoli
        if self.check_tables_moved():
            self.locomotion.stop()
            return StateResult(next_state="SCAN_TABLES")

        # Heartbeat sicurezza
        self.safety.heartbeat()

        time.sleep(1.0 / self.UPDATE_RATE_HZ)
        return StateResult(next_state=self.NAME)

    def exit(self):
        self.locomotion.stop()


class NavToPodTableState(NavigateState):
    """Navigazione verso il tavolo delle cialde."""
    NAME = "NAV_TO_POD_TABLE"

    def __init__(self, context):
        super().__init__(context, "pod_table", "DETECT_PODS")

    def enter(self):
        self.commentary.play_state("NAV_TO_POD_TABLE")
        super().enter()


class NavToBoxTableState(NavigateState):
    """Navigazione verso il tavolo della scatola."""
    NAME = "NAV_TO_BOX_TABLE"

    def __init__(self, context):
        super().__init__(context, "box_table", "PLACE_POD_IN_BOX")

    def enter(self):
        self.commentary.play_state("NAV_TO_BOX_TABLE")
        super().enter()


class NavToFullTableState(NavigateState):
    """Navigazione verso il tavolo scatole piene."""
    NAME = "NAV_TO_FULL_TABLE"

    def __init__(self, context):
        super().__init__(context, "full_table", "PLACE_FULL_BOX")

    def enter(self):
        self.commentary.play_state("CARRY_BOX")
        super().enter()


class NavBackToBoxTableState(NavigateState):
    """Torna al tavolo centrale dopo aver depositato la scatola piena."""
    NAME = "NAV_BACK_TO_BOX_TABLE"

    def __init__(self, context):
        super().__init__(context, "box_table", "GET_EMPTY_BOX")

    def enter(self):
        super().enter()
