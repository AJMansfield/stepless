from stepless.ball import Ball
import numpy as np
from typing import Callable, Any

def centroid(t: float, bodies: list[Ball]):
    mass_pos = np.zeros_like(bodies[0].x_at(t))
    total_mass = 0.
    for ball in bodies:
        mass_pos += ball.m_at(t) * ball.x_at(t)
        total_mass += ball.m_at(t)
    return mass_pos / total_mass

def mass(t: float, bodies: list[Ball]):
    mass = np.zeros_like(bodies[0].m_at(t))
    for ball in bodies:
        mass += ball.m_at(t)

def momentum(t: float, bodies: list[Ball]):
    momentum = np.zeros_like(bodies[0].P_at(t))
    for ball in bodies:
        momentum += ball.P_at(t)
    return momentum

def kinetic_energy(t: float, bodies: list[Ball]):
    energy = 0.
    for ball in bodies:
        energy += ball.E_at(t)
    return energy

def hamiltonian(t: float, bodies: list[Ball]):
    h = 0.
    for ball in bodies:
        h += ball.E_at(t) + ball.U_at(t)
    return h

def assert_conservation_law_obeyed(
        law: Callable[[float, list[Ball]], Any],
        stages: list[tuple[float, list[Ball]]],
    ):
    law_value_it = iter(law(t, s) for t, s in stages)
    prev_law_value = next(law_value_it)
    for law_value in law_value_it:
        if not np.allclose(prev_law_value, law_value):
            assert False, f"{law_value!r} is not the same as {prev_law_value!r}"
        prev_law_value = law_value
    return True