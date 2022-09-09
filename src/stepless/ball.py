from dataclasses import dataclass, replace
import numpy as np
from numpy.typing import NDArray
from warnings import warn
from typing import Union

from stepless.types import scalar_T, vector_T, vec_zero, Massive
from stepless.util import dot, next_time_after
from stepless.impulse import CollisionImpulse

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
    m: scalar_T = 1.
    """Mass of this ball."""

    def x_at(self, t: scalar_T) -> vector_T:
        return (self.a / 2 * t + self.v) * t + self.x
    def v_at(self, t: scalar_T) -> vector_T:
        return self.a * t + self.v
    def a_at(self, t: scalar_T) -> vector_T:
        return self.a

    def m_at(self, t: scalar_T) -> vector_T:
        return self.m

    def P_at(self, t: scalar_T) -> vector_T:
        """Momentum."""
        return self.m * self.v_at(t)
    def F_at(self, t: scalar_T) -> vector_T:
        """Force."""
        return self.m * self.a_at(t)
    def U_at(self, t: scalar_T) -> scalar_T:
        """Potential energy (from acceleration vector)."""
        return -self.m * dot(self.a_at(t), self.x_at(t) - self.x)
    def E_at(self, t: scalar_T) -> scalar_T:
        """Kinetic energy (from velocity)."""
        v = self.v_at(t)
        return 0.5 * self.m * dot(v,v)

    
    def apply_impulse(self, t: Union[scalar_T, CollisionImpulse],
            dx: vector_T = vec_zero,
            dv: vector_T = vec_zero,
            da: vector_T = vec_zero,
            dP: vector_T = vec_zero,
            dF: vector_T = vec_zero,
            ) -> 'Ball':
        if isinstance(t, CollisionImpulse):
            assert all([
                np.all(dx == vec_zero),
                np.all(dv == vec_zero),
                np.all(da == vec_zero),
                np.all(dP == vec_zero),
                np.all(dF == vec_zero),
            ])
            return self.apply_impulse(t=t.t, dx=t.dx, dv=t.dv)

        da += dF / self.m
        dv += dP / self.m

        new_a = self.a + da
        new_v = self.v - da*t + dv
        new_x = self.x + (da/2*t - dv)*t + dx
        return replace(self, x=new_x, v=new_v, a=new_a)
    
    def apply_state(self, t: scalar_T,
            x: vector_T = None,
            v: vector_T = None,
            a: vector_T = None,
            P: vector_T = None,
            F: vector_T = None,
            ):
        dx = vec_zero if x is None else self.x_at(t) - x
        dv = vec_zero if v is None else self.v_at(t) - v
        da = vec_zero if a is None else self.a_at(t) - a
        dP = vec_zero if P is None else self.P_at(t) - P
        dF = vec_zero if F is None else self.F_at(t) - F
        return self.apply_impulse(t=t,dx=dx,dv=dv,da=da,dP=dP,dF=dF)

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
        
    def compute_collision_times(self, other: 'Ball') -> NDArray:
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

    def compute_next_collision_time(self, other: 'Ball', t: scalar_T) -> scalar_T:
        return next_time_after(self.compute_collision_times(other), t=t)
