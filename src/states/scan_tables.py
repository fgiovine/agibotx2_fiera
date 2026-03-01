"""Stato SCAN_TABLES: rileva posizione dei 3 tavoli.

Usa RGB-D + RANSAC per trovare i 3 piani orizzontali (tavoli).
Li identifica come pod_table (SX), box_table (CENTRO), full_table (DX).
"""

import time

import numpy as np

from src.states.base_state import BaseState, StateResult


class ScanTablesState(BaseState):
    """Rileva e identifica i 3 tavoli."""

    NAME = "SCAN_TABLES"
    MAX_ATTEMPTS = 5
    SCAN_DELAY_S = 1.0

    def enter(self):
        self.commentary.play_state("SCAN_TABLES")
        self._logger.info("Scansione tavoli in corso...")
        self._attempts = 0

    def execute(self) -> StateResult:
        self._attempts += 1

        if self._attempts > self.MAX_ATTEMPTS:
            return StateResult(
                next_state="ERROR",
                error="Impossibile rilevare tutti i tavoli"
            )

        # Ottieni frame RGB-D
        frame = self.camera.get_frame()
        if frame is None:
            self._logger.warn("Frame non disponibile, riprovo...")
            time.sleep(self.SCAN_DELAY_S)
            return StateResult(next_state=self.NAME)

        rgb, depth = frame

        if not self.camera.has_intrinsics:
            self._logger.warn("Intrinseci camera non disponibili")
            time.sleep(self.SCAN_DELAY_S)
            return StateResult(next_state=self.NAME)

        fx, fy, cx, cy = self.camera.intrinsics

        # Trasformazione camera -> base (usa la corrente dalla testa)
        T_cam_base = np.eye(4)
        if hasattr(self.pose_estimator, '_T_camera_to_base'):
            T = self.pose_estimator._T_camera_to_base
            if T is not None:
                T_cam_base = T

        # Rileva tavoli
        tables = self.table_detector.detect_from_depth(
            depth, fx, fy, cx, cy, T_cam_base
        )

        if len(tables) < 3:
            self._logger.warn(
                f"Solo {len(tables)} tavoli rilevati (servono 3). "
                f"Tentativo {self._attempts}/{self.MAX_ATTEMPTS}"
            )
            # Potrebbe ruotare la testa per vedere meglio
            time.sleep(self.SCAN_DELAY_S)
            return StateResult(next_state=self.NAME)

        # Aggiorna il position tracker con le posizioni dei tavoli
        self.position_tracker.set_reference_tables(
            self.table_detector.tables
        )

        self._logger.info(
            f"Tutti e 3 i tavoli rilevati! "
            + ", ".join(f"{t.name}: ({t.center[0]:.2f}, {t.center[1]:.2f})"
                        for t in tables)
        )

        return StateResult(next_state="IDLE")

    def exit(self):
        pass
