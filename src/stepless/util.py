from numpy.typing import NDArray
from stepless.types import scalar_T
import numpy as np

def dot(a, b):
    """Compute the dot product of two vectors, with fallback to alternative
    ways of computing it to allow e.g. sympy vectors to be used as well."""
    try:
        return a.dot(b)
    except AttributeError:
        return a.T @ b #sympy compatibility?

def next_time_after(roots: NDArray, t: scalar_T) -> scalar_T:
    """Finds the smallest *real* element of `roots` that's larger than `t`."""
    best = np.inf
    for r in roots:
        r = np.real_if_close(r)
        if np.isrealobj(r) and r > t and r < best:
            best = r
    return best