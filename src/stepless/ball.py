from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt

scalar_T = np.float64
default_scalar_T = np.float64
vector_T = np.ndarray[(2,), scalar_T]
default_vector_T = lambda value=(0,0): field(default_factory=lambda: np.array(value, dtype=np.float64), compare=False)

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

@dataclass
class Ball:
    x: vector_T
    """Virtual t=0 position of the center of this ball."""
    v: vector_T
    """Virtual t=0 velocity of the center of this ball."""
    a: vector_T
    """Acceleration of this ball."""
    r: scalar_T
    """Collision radius of this ball."""

    def x_at(self, t: scalar_T) -> vector_T:
        return (self.a / 2 * t + self.v) * t + self.x
    def v_at(self, t: scalar_T) -> vector_T:
        return self.a * t + self.v
    
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
    
    def apply_impulse(self, t: scalar_T,
            dx: vector_T = default_vector_T(),
            dv: vector_T = default_vector_T(),
            da: vector_T = default_vector_T(),
            ):
        new_a = self.a + da
        new_v = self.v - da*t + dv
        new_x = self.x + (da/2*t - dv)*t + dx
        return Ball(x=new_x, v=new_v, a=new_a, r=self.r)

