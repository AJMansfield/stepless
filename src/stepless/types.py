from dataclasses import dataclass
from abc import ABC
from typing import Union
import numpy as np

scalar_T = np.float64
"""Scalar type used by simulation."""

vector_T = np.ndarray[(2,), scalar_T]
"""Vector type used by simulation."""

vec_zero = np.zeros(2, dtype=scalar_T)
"""Zero vector constant, for convenience."""

massable_T = Union['Massive', scalar_T]

@dataclass
class Massive(ABC):
    """Any object from which a mass can be retrieved."""
    
    m: scalar_T
    """Mass of this object."""

    @classmethod
    def mass_of(cls, that: massable_T) -> scalar_T:
        """Retrieve this object's mass (or return directly if a scalar)."""
        return getattr(that, 'm', that)

