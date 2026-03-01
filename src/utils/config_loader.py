"""Caricamento e validazione configurazione YAML."""

import os
from typing import Any

import yaml


class ConfigLoader:
    """Carica e fornisce accesso ai parametri di configurazione."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'config', 'demo_config.yaml'
            )
        self._config_path = os.path.abspath(config_path)
        self._config: dict = {}
        self.load()

    def load(self):
        """Carica il file YAML."""
        with open(self._config_path, 'r') as f:
            self._config = yaml.safe_load(f) or {}

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Accesso a parametri con notazione punto.

        Esempio: config.get('manipulation.gripper.open_position')
        """
        keys = dotted_key.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    @property
    def raw(self) -> dict:
        """Ritorna il dizionario completo."""
        return self._config


class JointLimitsLoader:
    """Carica i limiti dei giunti dal file YAML."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'config', 'joint_limits.yaml'
            )
        with open(os.path.abspath(config_path), 'r') as f:
            self._limits = yaml.safe_load(f) or {}

    def get_limits(self, group: str) -> dict:
        """Ritorna limiti per un gruppo (left_arm, right_arm, head, waist)."""
        return self._limits.get(group, {})

    def get_safe_limits(self, group: str) -> list:
        """Ritorna limiti con margine di sicurezza applicato."""
        limits = self.get_limits(group)
        margin = self._limits.get('safety_margin_deg', 5.0)
        pos_limits = limits.get('position_limits_deg', [])
        safe = []
        for lo, hi in pos_limits:
            safe.append([lo + margin, hi - margin])
        return safe

    @property
    def safety_margin_deg(self) -> float:
        return self._limits.get('safety_margin_deg', 5.0)
