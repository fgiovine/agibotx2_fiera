"""Helper ROS2 per QoS, service call con retry, time sync."""

import time
from typing import Any, Optional, Type

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)


def sensor_qos() -> QoSProfile:
    """QoS profile per topic sensori (best effort, volatile)."""
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def reliable_qos(depth: int = 10) -> QoSProfile:
    """QoS profile reliable per comandi."""
    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def call_service_sync(node: Node, client, request,
                       timeout_sec: float = 5.0,
                       max_retries: int = 3) -> Optional[Any]:
    """Chiama un servizio ROS2 in modo sincrono con retry.

    Args:
        node: nodo ROS2
        client: service client
        request: richiesta
        timeout_sec: timeout per singola chiamata
        max_retries: numero massimo tentativi

    Returns:
        Response o None se fallito
    """
    for attempt in range(max_retries):
        if not client.wait_for_service(timeout_sec=timeout_sec):
            node.get_logger().warn(
                f"Servizio non disponibile (tentativo {attempt + 1}/{max_retries})"
            )
            continue

        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)

        if future.done():
            return future.result()

        node.get_logger().warn(
            f"Timeout servizio (tentativo {attempt + 1}/{max_retries})"
        )

    node.get_logger().error(f"Servizio fallito dopo {max_retries} tentativi")
    return None


def wait_for_topic(node: Node, topic_name: str, msg_type: Type,
                    timeout_sec: float = 10.0) -> Optional[Any]:
    """Attende un singolo messaggio da un topic.

    Args:
        node: nodo ROS2
        topic_name: nome del topic
        msg_type: tipo del messaggio
        timeout_sec: timeout

    Returns:
        Messaggio ricevuto o None
    """
    received = {'msg': None}

    def callback(msg):
        received['msg'] = msg

    sub = node.create_subscription(msg_type, topic_name, callback, sensor_qos())

    start = time.time()
    while received['msg'] is None and (time.time() - start) < timeout_sec:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_subscription(sub)
    return received['msg']


def stamp_to_sec(stamp) -> float:
    """Converte un ROS2 Time stamp in secondi float."""
    return stamp.sec + stamp.nanosec * 1e-9


def sec_to_stamp(sec: float):
    """Converte secondi float in ROS2 Time stamp."""
    from builtin_interfaces.msg import Time
    stamp = Time()
    stamp.sec = int(sec)
    stamp.nanosec = int((sec - int(sec)) * 1e9)
    return stamp
