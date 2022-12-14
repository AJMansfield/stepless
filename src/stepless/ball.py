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
    r"""Virtual t=0 position state variable. $$\vec{x}_0$$"""
    v: vector_T = vec_zero
    r"""Virtual t=0 velocity state variable. $$\vec{v}_0$$"""
    a: vector_T = vec_zero
    r"""Acceleration state variable. $$\vec{a}$$"""
    r: scalar_T = 1.
    r"""Collision radius state variable. $$r$$"""
    m: scalar_T = 1.
    r"""Mass state variable. $$m$$"""
    b: vector_T = vec_zero
    r"""Restitution vector. $$R = b_1 \cdot b_2$$"""

    def x_at(self, t: scalar_T) -> vector_T:
        r"""Position. $$\vec{x} = \frac{1}{2} \vec{a}_0 t^2 + \vec{v}_0 t + \vec{x}_0$$"""
        return (self.a / 2 * t + self.v) * t + self.x
    def v_at(self, t: scalar_T) -> vector_T:
        r"""Velocity. $$\vec{v} = \vec{a}_0 t + \vec{v}_0$$"""
        return self.a * t + self.v
    def a_at(self, t: scalar_T) -> vector_T:
        r"""Acceleration. $$\vec{a} = \vec{a}_0$$"""
        return self.a

    def r_at(self, t: scalar_T) -> vector_T:
        r"""Collision radius. $$r$$"""
        return self.r
    def m_at(self, t: scalar_T) -> vector_T:
        r"""Mass. $$m$$"""
        return self.m

    def P_at(self, t: scalar_T) -> vector_T:
        r"""Momentum. $$P=m\vec{v}$$"""
        return self.m * self.v_at(t)
    def F_at(self, t: scalar_T) -> vector_T:
        r"""Force aka Thrust, $$F=m\vec{a}$$"""
        return self.m * self.a_at(t)
    def U_at(self, t: scalar_T) -> scalar_T:
        r"""Potential energy (from acceleration vector). $$U = -m \vec{a}\cdot\vec{x}$$"""
        return -self.m * dot(self.a_at(t), self.x_at(t))
    def K_at(self, t: scalar_T) -> scalar_T:
        r"""Kinetic energy (from velocity). $$K=\frac{1}{2} m \left\|\vec{v}\right\|^2$$"""
        v = self.v_at(t)
        return 0.5 * self.m * dot(v,v)
    def E_at(self, t: scalar_T) -> scalar_T:
        r"""Total energy. $$E = K + U$$"""
        return self.K_at(t) + self.U_at(t)

    def _inplace_or_replace(self, inplace:bool, **kw):
        if inplace:
            for k, v in kw.items():
                setattr(self, k, v)
            return self
        else:
            return replace(self, **kw)

    def __add__(self, impulse: CollisionImpulse) -> 'Ball':
        return self.apply_impulse(t=impulse.t, dx=impulse.dx, dv=impulse.dv)
    def __iadd__(self, impulse: CollisionImpulse) -> 'Ball':
        return self.apply_impulse(t=impulse.t, dx=impulse.dx, dv=impulse.dv, inplace=True)
    
    def apply_impulse(self, t: scalar_T,
            dx: vector_T = vec_zero,
            dv: vector_T = vec_zero,
            da: vector_T = vec_zero,
            dP: vector_T = vec_zero,
            dF: vector_T = vec_zero,
            inplace: bool = False,
            ) -> 'Ball':

        da += dF / self.m
        dv += dP / self.m

        new_a = self.a + da
        new_v = self.v - da*t + dv
        new_x = self.x + (da/2*t - dv)*t + dx
        return self._inplace_or_replace(inplace, a=new_a, v=new_v, x=new_x)
    
    def apply_state(self, t: scalar_T,
            x: vector_T = None,
            v: vector_T = None,
            a: vector_T = None,
            P: vector_T = None,
            F: vector_T = None,
            inplace: bool = False,
            ):
        dx = vec_zero if x is None else self.x_at(t) - x
        dv = vec_zero if v is None else self.v_at(t) - v
        da = vec_zero if a is None else self.a_at(t) - a
        dP = vec_zero if P is None else self.P_at(t) - P
        dF = vec_zero if F is None else self.F_at(t) - F
        return self.apply_impulse(t=t,dx=dx,dv=dv,da=da,dP=dP,dF=dF,inplace=inplace)

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
