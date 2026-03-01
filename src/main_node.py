"""Nodo ROS2 principale - Orchestratore demo fiera.

Inizializza il robot, avvia la FSM, gestisce il ciclo principale.
"""

import os
import signal
import sys

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from src.utils.config_loader import ConfigLoader
from src.state_machine import StateMachine


class DemoNode(Node):
    """Nodo ROS2 principale per la demo pick & place cialde."""

    def __init__(self):
        super().__init__('agibot_x2_demo_fiera')

        self.get_logger().info("=" * 60)
        self.get_logger().info("  Agibot X2 - Demo Fiera Pick & Place Cialde")
        self.get_logger().info("=" * 60)

        # Carica configurazione
        config_path = self.declare_parameter(
            'config_path', ''
        ).get_parameter_value().string_value

        if not config_path:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', 'config', 'demo_config.yaml'
            )

        self.get_logger().info(f"Configurazione: {config_path}")
        self._config = ConfigLoader(config_path)

        # Crea la state machine
        self._fsm = StateMachine(self, self._config)

        # Timer per il loop principale
        self._loop_rate = 10.0  # Hz
        self._main_timer = self.create_timer(
            1.0 / self._loop_rate, self._main_loop
        )

        # Flag inizializzazione
        self._initialized = False

        # Gestione shutdown
        self._shutdown_requested = False

    def _main_loop(self):
        """Loop principale: inizializza al primo ciclo, poi esegui FSM."""
        if self._shutdown_requested:
            return

        if not self._initialized:
            self._initialize()
            return

        # Esegui un passo della FSM
        self._fsm.step()

    def _initialize(self):
        """Inizializzazione del robot e avvio FSM."""
        self.get_logger().info("Inizializzazione robot...")

        if self._fsm.initialize():
            self._fsm.start("SCAN_TABLES")
            self._initialized = True
            self.get_logger().info("Demo avviata!")
        else:
            self.get_logger().error(
                "Inizializzazione fallita. Riprovo al prossimo ciclo..."
            )

    def shutdown(self):
        """Shutdown pulito."""
        self.get_logger().info("Shutdown in corso...")
        self._shutdown_requested = True

        if self._fsm is not None:
            self._fsm.destroy()

        self.get_logger().info("Shutdown completato")


def main(args=None):
    rclpy.init(args=args)

    node = DemoNode()

    # Gestione SIGINT/SIGTERM
    def signal_handler(sig, frame):
        node.get_logger().info("Segnale di arresto ricevuto")
        node.shutdown()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Executor multi-threaded per callbacks concorrenti
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
