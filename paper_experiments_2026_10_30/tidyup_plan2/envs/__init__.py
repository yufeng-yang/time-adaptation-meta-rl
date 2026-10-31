"""Hidden-gate time-aware Meta-RL environments.

The discrete and MuJoCo environments have separate optional dependencies, so
the public discrete classes are loaded only when requested.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ThreeRouteHiddenGateEnvV3",
    "TwoRouteHiddenGateEnvV3",
    "ThreeRouteHiddenGateEnvV4",
    "TwoRouteHiddenGateEnvV4",
]


def __getattr__(name: str) -> Any:
    if name in {"ThreeRouteHiddenGateEnvV3", "TwoRouteHiddenGateEnvV3"}:
        from .base_env_V3 import (
            ThreeRouteHiddenGateEnvV3,
            TwoRouteHiddenGateEnvV3,
        )

        return {
            "ThreeRouteHiddenGateEnvV3": ThreeRouteHiddenGateEnvV3,
            "TwoRouteHiddenGateEnvV3": TwoRouteHiddenGateEnvV3,
        }[name]
    if name in {"ThreeRouteHiddenGateEnvV4", "TwoRouteHiddenGateEnvV4"}:
        from .base_env_V4 import (
            ThreeRouteHiddenGateEnvV4,
            TwoRouteHiddenGateEnvV4,
        )

        return {
            "ThreeRouteHiddenGateEnvV4": ThreeRouteHiddenGateEnvV4,
            "TwoRouteHiddenGateEnvV4": TwoRouteHiddenGateEnvV4,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
