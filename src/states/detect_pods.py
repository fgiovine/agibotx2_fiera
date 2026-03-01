"""Stato DETECT_PODS: rileva cialde sul tavolo con YOLO + depth -> 3D.

Localizza le cialde, stima le posizioni 3D, e seleziona la piu vicina.
"""

from src.states.base_state import BaseState, StateResult


class DetectPodsState(BaseState):
    """Rileva e localizza cialde in 3D."""

    NAME = "DETECT_PODS"

    def enter(self):
        self.commentary.play_state("DETECT_PODS")
        self._logger.info("Rilevamento cialde in corso...")

    def execute(self) -> StateResult:
        # Ottieni frame
        frame = self.camera.get_frame()
        if frame is None:
            return StateResult(
                next_state="ERROR",
                error="Camera non disponibile per detection"
            )

        rgb, depth = frame
        fx, fy, cx, cy = self.camera.intrinsics

        # Rileva cialde 2D
        detections = self.pod_detector.detect(rgb)

        if not detections:
            self._logger.info("Nessuna cialda rilevata, torno a IDLE")
            return StateResult(next_state="IDLE")

        self._logger.info(f"Rilevate {len(detections)} cialde")

        # Stima posizioni 3D
        pods_3d = self.pose_estimator.estimate_multiple(
            detections, depth, fx, fy, cx, cy
        )

        if not pods_3d:
            self._logger.warn("Nessuna posizione 3D valida")
            return StateResult(next_state="IDLE")

        # Seleziona la cialda piu vicina
        robot_pos = self.position_tracker.position
        nearest = self.pose_estimator.select_nearest_pod(pods_3d, robot_pos)

        if nearest is None:
            return StateResult(next_state="IDLE")

        det, pos_3d = nearest
        self._logger.info(
            f"Cialda selezionata a ({pos_3d[0]:.3f}, {pos_3d[1]:.3f}, {pos_3d[2]:.3f})"
        )

        return StateResult(
            next_state="PICK_POD",
            data={"pod_position": pos_3d, "detection": det}
        )

    def exit(self):
        pass
