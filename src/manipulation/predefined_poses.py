"""Pose predefinite per il braccio.

Pose chiave pre-registrate per le operazioni di demo.
Queste vanno calibrate sul robot reale con record_poses.py.
"""

import numpy as np


# Pose braccio sinistro (7 DOF, radianti)
LEFT_ARM_POSES = {
    # Posizione di riposo - braccio a lato del corpo
    "home": np.array([0.0, 0.5, 0.0, -1.2, 0.0, 0.0, 0.0]),

    # Pre-grasp - braccio disteso sopra il tavolo
    "pre_grasp": np.array([-0.8, 0.3, 0.0, -0.8, 0.0, 0.3, 0.0]),

    # Sopra la scatola - per piazzare cialde
    "pre_place_box": np.array([-0.6, 0.4, 0.0, -0.9, 0.0, 0.2, 0.0]),

    # Presa scatola - per sollevare la scatola
    "box_grasp": np.array([-0.7, 0.5, 0.0, -1.0, 0.0, 0.0, 0.0]),

    # Trasporto scatola - braccio davanti al corpo
    "box_carry": np.array([-0.5, 0.3, 0.0, -0.7, 0.0, 0.2, 0.0]),

    # Posa coperchio scatola
    "lid_place": np.array([-0.6, 0.4, 0.0, -0.8, 0.0, 0.3, 0.0]),

    # Braccio sollevato per camminata sicura
    "walk_safe": np.array([-0.3, 0.2, 0.0, -1.5, 0.0, 0.0, 0.0]),

    # Posizione di presentazione (per interazione con pubblico)
    "present": np.array([-0.4, 0.8, 0.0, -0.5, 0.0, 0.3, 0.0]),
}

# Pose braccio destro (specchiate)
RIGHT_ARM_POSES = {
    "home": np.array([0.0, -0.5, 0.0, -1.2, 0.0, 0.0, 0.0]),
    "pre_grasp": np.array([-0.8, -0.3, 0.0, -0.8, 0.0, 0.3, 0.0]),
    "pre_place_box": np.array([-0.6, -0.4, 0.0, -0.9, 0.0, 0.2, 0.0]),
    "box_grasp": np.array([-0.7, -0.5, 0.0, -1.0, 0.0, 0.0, 0.0]),
    "box_carry": np.array([-0.5, -0.3, 0.0, -0.7, 0.0, 0.2, 0.0]),
    "lid_place": np.array([-0.6, -0.4, 0.0, -0.8, 0.0, 0.3, 0.0]),
    "walk_safe": np.array([-0.3, -0.2, 0.0, -1.5, 0.0, 0.0, 0.0]),
    "present": np.array([-0.4, -0.8, 0.0, -0.5, 0.0, 0.3, 0.0]),
}

# Pose testa
HEAD_POSES = {
    "center": np.array([0.0, 0.0]),           # Dritto avanti
    "look_down": np.array([0.0, 0.4]),         # Guarda tavolo
    "look_left": np.array([0.5, 0.2]),         # Guarda tavolo SX
    "look_right": np.array([-0.5, 0.2]),       # Guarda tavolo DX
    "look_audience": np.array([0.0, -0.1]),    # Guarda il pubblico
}


def get_arm_pose(name: str, side: str = "left") -> np.ndarray:
    """Ritorna una posa predefinita per il braccio.

    Args:
        name: nome della posa
        side: "left" o "right"

    Returns:
        Array 7 DOF di posizioni in radianti

    Raises:
        KeyError: se la posa non esiste
    """
    poses = LEFT_ARM_POSES if side == "left" else RIGHT_ARM_POSES
    if name not in poses:
        raise KeyError(
            f"Posa '{name}' non trovata. Disponibili: {list(poses.keys())}"
        )
    return poses[name].copy()


def get_head_pose(name: str) -> np.ndarray:
    """Ritorna una posa predefinita per la testa."""
    if name not in HEAD_POSES:
        raise KeyError(
            f"Posa testa '{name}' non trovata. Disponibili: {list(HEAD_POSES.keys())}"
        )
    return HEAD_POSES[name].copy()


def update_pose(name: str, values: np.ndarray, side: str = "left"):
    """Aggiorna una posa predefinita (dopo calibrazione).

    Args:
        name: nome della posa
        values: nuovi valori
        side: "left" o "right"
    """
    poses = LEFT_ARM_POSES if side == "left" else RIGHT_ARM_POSES
    poses[name] = values.copy()
