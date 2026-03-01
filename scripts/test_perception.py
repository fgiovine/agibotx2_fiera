#!/usr/bin/env python3
"""Test standalone per il modulo di percezione.

Testa camera, detection cialde, detection tavoli.

Uso:
    python scripts/test_perception.py [--test camera|pods|tables|all]
"""

import argparse
import time
import sys

import numpy as np

import rclpy
from rclpy.node import Node

from src.utils.config_loader import ConfigLoader
from src.perception.camera_manager import CameraManager
from src.perception.pod_detector import PodDetector
from src.perception.table_detector import TableDetector
from src.perception.pose_estimator import PoseEstimator


class PerceptionTester(Node):
    """Nodo per test percezione."""

    def __init__(self):
        super().__init__('perception_tester')
        self._config = ConfigLoader()
        self._camera = CameraManager(self, self._config)
        self._pod_detector = PodDetector(self, self._config)
        self._table_detector = TableDetector(self, self._config)
        self._pose_estimator = PoseEstimator(self, self._config)

    def test_camera(self) -> bool:
        """Testa la connessione alla camera."""
        print("\n=== Test Camera ===")
        print("Attendo frame dalla camera...")

        for i in range(50):  # 5 secondi
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._camera.has_data:
                break

        if not self._camera.has_data:
            print("FAIL: Nessun frame ricevuto dalla camera")
            return False

        frame = self._camera.get_frame()
        rgb, depth = frame
        print(f"OK: Frame ricevuto")
        print(f"  RGB: {rgb.shape}, dtype={rgb.dtype}")
        print(f"  Depth: {depth.shape}, dtype={depth.dtype}")
        print(f"  Depth range: {depth[depth > 0].min():.3f} - {depth.max():.3f} m")

        if self._camera.has_intrinsics:
            fx, fy, cx, cy = self._camera.intrinsics
            print(f"  Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
        else:
            print("  WARN: Intrinseci non ancora ricevuti")

        return True

    def test_pod_detection(self) -> bool:
        """Testa la detection delle cialde."""
        print("\n=== Test Pod Detection ===")

        # Attendi un frame
        for i in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._camera.has_data:
                break

        rgb = self._camera.get_rgb()
        if rgb is None:
            print("FAIL: Nessuna immagine RGB")
            return False

        detections = self._pod_detector.detect(rgb)
        print(f"Rilevate {len(detections)} cialde:")
        for i, det in enumerate(detections):
            print(
                f"  {i + 1}. center={det.center_px}, "
                f"radius={det.radius_px:.1f}px, "
                f"conf={det.confidence:.2f}, "
                f"method={det.method}"
            )

        # Visualizza
        try:
            import cv2
            vis = self._pod_detector.draw_detections(rgb, detections)
            cv2.imshow("Pod Detection", vis)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
        except Exception:
            print("(Visualizzazione non disponibile)")

        return len(detections) > 0

    def test_table_detection(self) -> bool:
        """Testa la detection dei tavoli."""
        print("\n=== Test Table Detection ===")

        for i in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._camera.has_data:
                break

        frame = self._camera.get_frame()
        if frame is None:
            print("FAIL: Nessun frame")
            return False

        rgb, depth = frame
        if not self._camera.has_intrinsics:
            print("FAIL: Intrinseci non disponibili")
            return False

        fx, fy, cx, cy = self._camera.intrinsics
        T_identity = np.eye(4)  # Senza calibrazione, usa identita

        tables = self._table_detector.detect_from_depth(
            depth, fx, fy, cx, cy, T_identity
        )

        print(f"Rilevati {len(tables)} tavoli:")
        for t in tables:
            print(
                f"  {t.name}: center=({t.center[0]:.3f}, {t.center[1]:.3f}, "
                f"{t.center[2]:.3f}), size=({t.size[0]:.2f}x{t.size[1]:.2f}m)"
            )

        return len(tables) >= 1


def main():
    parser = argparse.ArgumentParser(description='Test percezione')
    parser.add_argument('--test', type=str, default='all',
                        choices=['camera', 'pods', 'tables', 'all'])

    args = parser.parse_args()

    rclpy.init()
    tester = PerceptionTester()

    results = {}

    if args.test in ('camera', 'all'):
        results['camera'] = tester.test_camera()

    if args.test in ('pods', 'all'):
        results['pods'] = tester.test_pod_detection()

    if args.test in ('tables', 'all'):
        results['tables'] = tester.test_table_detection()

    print("\n=== Risultati ===")
    all_ok = True
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
        if not result:
            all_ok = False

    tester.destroy_node()
    rclpy.shutdown()

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
