"""Trasformazioni di coordinate camera -> base robot."""

import numpy as np
from scipy.spatial.transform import Rotation


def pixel_to_3d(u: float, v: float, depth: float,
                fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Converte pixel (u,v) + depth in punto 3D nel frame camera.

    Args:
        u, v: coordinate pixel
        depth: profondita in metri
        fx, fy: lunghezze focali
        cx, cy: centro ottico

    Returns:
        Punto 3D [x, y, z] nel frame camera
    """
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return np.array([x, y, z])


def transform_point(point: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Trasforma un punto 3D usando una matrice 4x4.

    Args:
        point: punto [x, y, z]
        T: matrice di trasformazione 4x4

    Returns:
        Punto trasformato [x, y, z]
    """
    p_hom = np.append(point, 1.0)
    p_out = T @ p_hom
    return p_out[:3]


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Crea matrice di trasformazione 4x4 da rotazione (3x3) e traslazione (3,)."""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def make_transform_from_euler(roll: float, pitch: float, yaw: float,
                               tx: float, ty: float, tz: float) -> np.ndarray:
    """Crea matrice 4x4 da angoli di Eulero (XYZ) e traslazione."""
    R = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    return make_transform(R, np.array([tx, ty, tz]))


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverte una matrice di trasformazione 4x4."""
    T_inv = np.eye(4)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def approach_vector_down() -> np.ndarray:
    """Matrice di rotazione per approccio dall'alto verso il basso.

    L'asse Z dell'end-effector punta verso il basso (-Z globale).
    """
    # End-effector Z punta giu, X in avanti, Y a sinistra
    R = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ])
    return R


def compute_box_slot_position(row: int, col: int,
                               box_origin: np.ndarray,
                               spacing_m: float = 0.05) -> np.ndarray:
    """Calcola posizione 3D di uno slot nella griglia scatola (4x5).

    Args:
        row: riga (0-3)
        col: colonna (0-4)
        box_origin: angolo in basso a sinistra della scatola [x, y, z]
        spacing_m: distanza tra centri cialde

    Returns:
        Posizione 3D del centro dello slot
    """
    offset = np.array([
        col * spacing_m + spacing_m / 2,
        row * spacing_m + spacing_m / 2,
        0.0
    ])
    return box_origin + offset


def distance_2d(p1: np.ndarray, p2: np.ndarray) -> float:
    """Distanza 2D (piano XY) tra due punti 3D."""
    diff = p1[:2] - p2[:2]
    return float(np.linalg.norm(diff))


def angle_to_target(robot_pos: np.ndarray, robot_yaw: float,
                     target_pos: np.ndarray) -> float:
    """Angolo da ruotare per puntare verso un target.

    Returns:
        Angolo in radianti (positivo = antiorario)
    """
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    target_yaw = np.arctan2(dy, dx)
    return _normalize_angle(target_yaw - robot_yaw)


def _normalize_angle(angle: float) -> float:
    """Normalizza angolo in [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
