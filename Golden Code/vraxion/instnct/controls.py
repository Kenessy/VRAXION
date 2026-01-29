"""INSTNCT control loops (AGC, thermostat, inertia auto, cadence, panic).

This module is the **stable public facade** for INSTNCT controls.

The implementation is intentionally split into small, single-purpose modules
(:mod:`.thermo`, :mod:`.agc`, :mod:`.inertia_auto`, :mod:`.panic`, :mod:`.cadence`).
Keeping the public API concentrated here makes callsites and auditing easier
while allowing internal refactors with lower risk.

IMPORTANT:
- Behavior is locked by ``tests/verify_golden.py`` and must not change.
- All functions mutate the passed-in ``model`` object in-place.
"""

from __future__ import annotations

from .agc import AGCParams, apply_update_agc
from .cadence import CadenceGovernor
from .inertia_auto import InertiaAutoParams, apply_inertia_auto
from .panic import PanicReflex
from .thermo import ThermostatParams, apply_thermostat

__all__ = [
    "ThermostatParams",
    "AGCParams",
    "InertiaAutoParams",
    "apply_thermostat",
    "apply_update_agc",
    "apply_inertia_auto",
    "PanicReflex",
    "CadenceGovernor",
]


# Preserve the historical ``__module__`` for re-exported symbols.
#
# This keeps introspection and pickling stable for callers that persisted
# objects from the era where these definitions lived directly in this module.
ThermostatParams.__module__ = __name__
AGCParams.__module__ = __name__
InertiaAutoParams.__module__ = __name__
PanicReflex.__module__ = __name__
CadenceGovernor.__module__ = __name__
apply_thermostat.__module__ = __name__
apply_update_agc.__module__ = __name__
apply_inertia_auto.__module__ = __name__
