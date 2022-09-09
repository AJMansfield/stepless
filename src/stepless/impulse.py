from typing import Union
from dataclasses import dataclass, replace
import numpy as np
from stepless.types import scalar_T, vector_T, vec_zero, Massive, massable_T

@dataclass
class CollisionImpulse:
    t: scalar_T
    dx: vector_T = vec_zero
    dv: vector_T = vec_zero

    # _e: scalar_T = 0
    # @property
    # def e(self) -> scalar_T:
    #     """Set the restitution value of this impulse."""
    #     return self.e
    # @e.setter
    # def e(self, restitution: scalar_T):
    #     old_e = self._e
    #     self._e = restitution
    #     self.dv = self.dv / (1+old_e) * (1+self._e)
    
    def with_restitution(self, e: scalar_T) -> 'CollisionImpulse':
        return replace(self, dv=self.dv*(1+e))
    
    def split(self, m1: massable_T, m2: massable_T) -> 'tuple[CollisionImpulse, CollisionImpulse]':
        """Split this impulse into two pieces based on relative masses."""
        m1 = Massive.mass_of(m1)
        m2 = Massive.mass_of(m2)
        if np.isinf(m1) and not np.isinf(m2):
            return (
                replace(self, dx= vec_zero, dv= vec_zero),
                self,
            )
        elif not np.isinf(m1) and np.isinf(m2):
            return (
                -self,
                replace(self, dx= vec_zero, dv= vec_zero),
            )
        else: # if both infinite, the standard behavior already correctly returns NaNs
            denom = m1 + m2
            f1 = -m2 / denom
            f2 = m1 / denom
            return (
                replace(self, dx= self.dx*f1, dv= self.dv*f1),
                replace(self, dx= self.dx*f2, dv= self.dv*f2),
            )
    
    def __add__(self, other: 'CollisionImpulse') -> 'CollisionImpulse':
        if not np.isclose(self.t, other.t):
            return NotImplemented
        return replace(self, dx=self.dx + other.dx, dv=self.dv + other.dv)
    def __sub__(self, other: 'CollisionImpulse') -> 'CollisionImpulse':
        if not np.isclose(self.t, other.t):
            return NotImplemented
        return replace(self, dx=self.dx - other.dx, dv=self.dv - other.dv)
    def __mul__(self, other: scalar_T) -> 'CollisionImpulse':
        return replace(self, dx=self.dx*other, dv=self.dv*other)
    def __rmul__(self, other: scalar_T) -> 'CollisionImpulse':
        return replace(self, dx=self.dx*other, dv=self.dv*other)
    def __truediv__(self, other: scalar_T) -> 'CollisionImpulse':
        return replace(self, dx=self.dx/other, dv=self.dv/other)
    def __pos__(self) -> 'CollisionImpulse':
        return self
    def __neg__(self) -> 'CollisionImpulse':
        return replace(self, dx=-self.dx, dv=-self.dv)