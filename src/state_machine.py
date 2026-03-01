"""FSM (Finite State Machine) per la demo fiera.

Gestisce le transizioni tra stati e il contesto condiviso.
Evento globale: table_moved -> ritorna a SCAN_TABLES da qualsiasi stato.
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass, field

from rclpy.node import Node

from src.utils.config_loader import ConfigLoader
from src.perception.camera_manager import CameraManager
from src.perception.table_detector import TableDetector
from src.perception.pod_detector import PodDetector
from src.perception.pose_estimator import PoseEstimator
from src.perception.box_tracker import BoxTracker
from src.manipulation.ik_solver import IKSolver
from src.manipulation.arm_controller import ArmController
from src.manipulation.gripper_controller import GripperController
from src.manipulation.trajectory_planner import TrajectoryPlanner
from src.navigation.locomotion_controller import LocomotionController
from src.navigation.approach_planner import ApproachPlanner
from src.navigation.position_tracker import PositionTracker
from src.interaction.tts_client import TTSClient
from src.interaction.emoji_client import EmojiClient
from src.interaction.led_client import LEDClient
from src.interaction.commentary import Commentary
from src.interaction.voice_interaction import VoiceInteraction
from src.robot_hal.mode_manager import ModeManager
from src.robot_hal.input_source_manager import InputSourceManager
from src.robot_hal.safety_monitor import SafetyMonitor
from src.states.base_state import BaseState, StateResult


class DemoContext:
    """Contesto condiviso tra tutti gli stati.

    Contiene riferimenti a tutti i moduli del robot.
    """

    def __init__(self, node: Node, config: ConfigLoader):
        self.node = node
        self.config = config
        self.state_data: dict = {}  # Dati passati tra stati

        # Inizializza moduli
        self.mode_manager = ModeManager(node)
        self.input_source = InputSourceManager(node)
        self.safety = SafetyMonitor(node, config)
        self.camera = CameraManager(node, config)
        self.table_detector = TableDetector(node, config)
        self.pod_detector = PodDetector(node, config)
        self.pose_estimator = PoseEstimator(node, config)
        self.box_tracker = BoxTracker(node, config)
        self.ik_solver = IKSolver(node, config)
        self.arm_controller = ArmController(node, config, self.safety)
        self.gripper = GripperController(node, config)
        self.trajectory_planner = TrajectoryPlanner(node, config)
        self.locomotion = LocomotionController(node, config, self.mode_manager)
        self.approach_planner = ApproachPlanner(node, config, self.locomotion)
        self.position_tracker = PositionTracker(node, config)

        # Interazione
        tts = TTSClient(node, config)
        emoji = EmojiClient(node, config)
        led = LEDClient(node, config)
        self.commentary = Commentary(tts, emoji, led)

        # Interazione vocale (Ciruzzo)
        self.voice_interaction = VoiceInteraction(
            node, config, tts, emoji, led
        )

    def destroy(self):
        """Cleanup di tutti i moduli."""
        self.input_source.destroy()
        self.safety.destroy()
        self.camera.destroy()
        self.arm_controller.destroy()
        self.gripper.destroy()


class StateMachine:
    """Macchina a stati finiti per la demo."""

    def __init__(self, node: Node, config: ConfigLoader):
        self._node = node
        self._config = config
        self._logger = node.get_logger()

        # Context condiviso
        self._context = DemoContext(node, config)

        # Registro stati
        self._states: Dict[str, BaseState] = {}
        self._current_state: Optional[BaseState] = None
        self._current_state_name: str = ""
        self._running = False

        # Statistiche
        self._state_transitions: int = 0
        self._total_pods_picked: int = 0
        self._total_boxes_filled: int = 0

        self._register_states()

    def _register_states(self):
        """Registra tutti gli stati disponibili."""
        from src.states.idle import IdleState
        from src.states.scan_tables import ScanTablesState
        from src.states.detect_pods import DetectPodsState
        from src.states.navigate import (
            NavToPodTableState, NavToBoxTableState,
            NavToFullTableState, NavBackToBoxTableState,
        )
        from src.states.pick_pod import PickPodState
        from src.states.place_pod import PlacePodState, CheckBoxState
        from src.states.close_box import CloseBoxState
        from src.states.carry_box import CarryBoxState, PlaceFullBoxState
        from src.states.replace_box import GetEmptyBoxState, PlaceEmptyBoxState
        from src.states.error_recovery import ErrorRecoveryState
        from src.states.conversation import (
            WakeWordDetectedState, ConversationState,
        )

        ctx = self._context
        self._states = {
            "IDLE": IdleState(ctx),
            "SCAN_TABLES": ScanTablesState(ctx),
            "DETECT_PODS": DetectPodsState(ctx),
            "NAV_TO_POD_TABLE": NavToPodTableState(ctx),
            "NAV_TO_BOX_TABLE": NavToBoxTableState(ctx),
            "NAV_TO_FULL_TABLE": NavToFullTableState(ctx),
            "NAV_BACK_TO_BOX_TABLE": NavBackToBoxTableState(ctx),
            "PICK_POD": PickPodState(ctx),
            "PLACE_POD_IN_BOX": PlacePodState(ctx),
            "CHECK_BOX": CheckBoxState(ctx),
            "CLOSE_BOX": CloseBoxState(ctx),
            "CARRY_BOX": CarryBoxState(ctx),
            "PLACE_FULL_BOX": PlaceFullBoxState(ctx),
            "GET_EMPTY_BOX": GetEmptyBoxState(ctx),
            "PLACE_EMPTY_BOX": PlaceEmptyBoxState(ctx),
            "WAKE_WORD_DETECTED": WakeWordDetectedState(ctx),
            "CONVERSATION": ConversationState(ctx),
            "ERROR": ErrorRecoveryState(ctx),
        }

    def initialize(self) -> bool:
        """Inizializza il robot e registra input source.

        Returns:
            True se inizializzazione riuscita
        """
        self._logger.info("Inizializzazione demo...")

        # Registra input source
        if not self._context.input_source.register():
            self._logger.error("Registrazione input source fallita!")
            return False

        # Modalita stand
        if not self._context.mode_manager.ensure_stand():
            self._logger.error("Impossibile passare a modalita STAND")
            return False

        self._logger.info("Inizializzazione completata")
        return True

    def start(self, initial_state: str = "SCAN_TABLES"):
        """Avvia la macchina a stati."""
        self._running = True
        self._transition_to(initial_state)
        self._logger.info(f"FSM avviata dallo stato: {initial_state}")

    def stop(self):
        """Ferma la macchina a stati."""
        self._running = False
        self._context.locomotion.stop()
        self._logger.info("FSM fermata")

    def step(self):
        """Esegue un singolo passo della FSM.

        Chiamato dal loop principale del nodo ROS2.
        """
        if not self._running or self._current_state is None:
            return

        # Heartbeat sicurezza
        self._context.safety.heartbeat()

        # Controlla emergency stop
        if self._context.safety.estop_active:
            self._logger.error("Emergency stop attivo!")
            self._context.locomotion.stop()
            return

        # Esegui stato corrente
        try:
            result = self._current_state.execute()
        except Exception as e:
            self._logger.error(
                f"Eccezione nello stato {self._current_state_name}: {e}"
            )
            result = StateResult(
                next_state="ERROR",
                error=f"Eccezione: {e}"
            )

        # Gestisci transizione
        if result.next_state != self._current_state_name:
            # Passa dati al prossimo stato
            if result.data:
                self._context.state_data.update(result.data)
            if result.error:
                self._context.state_data["error"] = result.error

            self._transition_to(result.next_state)

    def _transition_to(self, state_name: str):
        """Transita a un nuovo stato."""
        # Exit stato corrente
        if self._current_state is not None:
            try:
                self._current_state.exit()
            except Exception as e:
                self._logger.error(f"Errore in exit di {self._current_state_name}: {e}")

        # Trova nuovo stato
        if state_name not in self._states:
            self._logger.error(f"Stato {state_name} non trovato! Vado a ERROR.")
            state_name = "ERROR"

        self._current_state = self._states[state_name]
        old_name = self._current_state_name
        self._current_state_name = state_name
        self._state_transitions += 1

        self._logger.info(f"Transizione: {old_name} -> {state_name}")

        # Reset error count se usciti da ERROR con successo
        if old_name == "ERROR" and state_name != "ERROR":
            error_state = self._states.get("ERROR")
            if hasattr(error_state, 'reset_error_count'):
                error_state.reset_error_count()

        # Enter nuovo stato
        try:
            self._current_state.enter()
        except Exception as e:
            self._logger.error(f"Errore in enter di {state_name}: {e}")
            if state_name != "ERROR":
                self._context.state_data["error"] = f"Enter fallito: {e}"
                self._transition_to("ERROR")

    @property
    def current_state_name(self) -> str:
        return self._current_state_name

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def context(self) -> DemoContext:
        return self._context

    def destroy(self):
        """Cleanup."""
        self.stop()
        self._context.destroy()
