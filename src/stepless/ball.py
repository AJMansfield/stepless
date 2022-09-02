from dataclasses import dataclass, field
import dataclasses
import numpy as np
import numpy.typing as npt
from warnings import warn
from typing import TypedDict, Union
from functools import singledispatchmethod

scalar_T = np.float64
vector_T = np.ndarray[(2,), scalar_T]

def dot(a, b):
    try:
        return a.dot(b)
    except AttributeError:
        return a.T @ b #sympy compatibility?

def next_root_after(roots: npt.NDArray, t: scalar_T) -> scalar_T:
    best = np.inf
    for r in roots:
        r = np.real_if_close(r)
        if np.isrealobj(r) and r > t and r < best:
            best = r
    return best

vec_zero = np.zeros(2)

@dataclass
class CollisionImpulse:
    t: scalar_T
    dx: vector_T
    dv: vector_T

    _e: scalar_T = 0
    @property
    def e(self) -> scalar_T:
        """Set the restitution value of this impulse."""
        return self.e
    @e.setter
    def e(self, restitution: scalar_T):
        old_e = self._e
        self._e = restitution
        self.dv = self.dv / (1+old_e) * (1+self._e)
    
    def with_restitution(self, e: scalar_T) -> 'CollisionImpulse':
        result = dataclasses.replace(self)
        result.e = e
        return result
    
    def split(self, m1: scalar_T, m2: scalar_T) -> 'tuple[CollisionImpulse, CollisionImpulse]':
        """Split this impulse into two pieces based on relative masses."""
        denom = m1 + m2
        f1 = -m1 / denom
        f2 = m2 / denom
        return (
            CollisionImpulse(t=self.t, dx=self.dx * f1, dv=self.dv * f1, _e=self._e),
            CollisionImpulse(t=self.t, dx=self.dx * f2, dv=self.dv * f2, _e=self._e)
        )


@dataclass
class Ball:
    x: vector_T = vec_zero
    """Virtual t=0 position of the center of this ball."""
    v: vector_T = vec_zero
    """Virtual t=0 velocity of the center of this ball."""
    a: vector_T = vec_zero
    """Acceleration of this ball."""
    r: scalar_T = 1.
    """Collision radius of this ball."""

    def x_at(self, t: scalar_T) -> vector_T:
        return (self.a / 2 * t + self.v) * t + self.x
    def v_at(self, t: scalar_T) -> vector_T:
        return self.a * t + self.v
    def a_at(self, t: scalar_T) -> vector_T:
        return self.a
    
    def apply_impulse(self, t: Union[scalar_T, CollisionImpulse],
            dx: vector_T = vec_zero,
            dv: vector_T = vec_zero,
            da: vector_T = vec_zero,
            ) -> 'Ball':
        if isinstance(t, CollisionImpulse):
            assert np.all(dx == vec_zero) and np.all(dv == vec_zero) and np.all(da == vec_zero)
            return self.apply_impulse(t=t.t, dx=t.dx, dv=t.dv)

        new_a = self.a + da
        new_v = self.v - da*t + dv
        new_x = self.x + (da/2*t - dv)*t + dx
        return self.__class__(x=new_x, v=new_v, a=new_a, r=self.r)
    
    
    def apply_state(self, t: scalar_T,
            x: vector_T = None,
            v: vector_T = None,
            a: vector_T = None,
            ):
        dx = vec_zero if x is None else self.x - x
        dv = vec_zero if v is None else self.v - v
        da = vec_zero if a is None else self.a - a
        return self.apply_impulse(t=t,dx=dx,dv=dv,da=da)

    def find_collision_ball(self, other: 'Ball') -> npt.NDArray:
        x = self.x - other.x
        v = self.v - other.v
        a = self.a - other.a
        r = self.r + other.r
        
        return np.poly1d((
            dot(a,a) / 4,
            dot(v,a),
            dot(x,a) + dot(v,v),
            dot(x,v) * 2,
            dot(x,x) - r*r,
        )).roots
    
    def get_collision_impulse(self, other: 'Ball', t: scalar_T) -> CollisionImpulse:
        x = self.x_at(t) - other.x_at(t)
        v = self.v_at(t) - other.v_at(t)
        a = self.a_at(t) - other.a_at(t)
        r = self.r + other.r

        dx = x * (1 - r / np.linalg.norm(x)) # displacement required for exact contact
        if not np.allclose(dx, vec_zero):
            warn(f"Collision displacement is nonzero: {dx}")
        dv = dot(v,x) / dot(x,x) * x

        return CollisionImpulse(t=t, dx=dx, dv=dv)
        

