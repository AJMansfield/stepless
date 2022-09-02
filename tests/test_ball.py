import pytest
import numpy as np
from stepless.ball import Ball
from typing import Callable, Any
from numpy import array as A

def random_vector():
    return np.random.rand(2) * 10

bodies_and_mass_T = list[tuple[Ball, float]]
def centroid(t: float, bodies: bodies_and_mass_T):
    mass_pos = np.zeros_like(bodies[0][0].x_at(t))
    total_mass = 0.
    for ball, mass in bodies:
        mass_pos += mass * ball.x_at(t)
        total_mass += mass
    return mass_pos / total_mass

def momentum(t: float, bodies: bodies_and_mass_T):
    momentum = np.zeros_like(bodies[0][0].v_at(t))
    for ball, mass in bodies:
        momentum += mass * ball.v_at(t)
    return momentum

def energy(t: float, bodies: bodies_and_mass_T):
    energy = 0.
    for ball, mass in bodies:
        energy += mass * np.linalg.norm(ball.v_at(t))**2
    return energy

def conservation_law_obeyed(t: float,
        law: Callable[[float, bodies_and_mass_T], Any],
        *stages:bodies_and_mass_T):
    law_value = [law(t, s) for s in stages]
    for v in law_value:
        if not np.allclose(law_value[0], v):
            return False
    return True

@pytest.mark.parametrize('n', range(5))
def test_impulse_dx(n):
    np.random.seed(n)
    b1 = Ball(x=random_vector(), v=random_vector(), a=random_vector(), r=1)
    t = np.random.rand()

    b1v, b1a = b1.v_at(t), b1.a_at(t)
    b2 = b1.apply_impulse(t=t, dx=random_vector())
    b2v, b2a = b2.v_at(t), b2.a_at(t)

    assert np.allclose(b1v, b2v)
    assert np.allclose(b1a, b2a)

@pytest.mark.parametrize('n', range(5))
def test_impulse_dv(n):
    np.random.seed(n)
    b1 = Ball(x=random_vector(), v=random_vector(), a=random_vector(), r=1)
    t = np.random.rand()

    b1x, b1a = b1.x_at(t), b1.a_at(t)
    b2 = b1.apply_impulse(t=t, dv=random_vector())
    b2x, b2a = b2.x_at(t), b2.a_at(t)

    assert np.allclose(b1x, b2x)
    assert np.allclose(b1a, b2a)

@pytest.mark.parametrize('n', range(5))
def test_impulse_da(n):
    np.random.seed(n)
    b1 = Ball(x=random_vector(), v=random_vector(), a=random_vector(), r=1)
    t = np.random.rand()

    b1x, b1v = b1.x_at(t), b1.v_at(t)
    b2 = b1.apply_impulse(t=t, da=random_vector())
    b2x, b2v = b2.x_at(t), b2.v_at(t)

    assert np.allclose(b1x, b2x)
    assert np.allclose(b1v, b2v)

@pytest.mark.parametrize('n', range(5))
def test_impulse_dx_dv(n):
    np.random.seed(n)
    b1 = Ball(x=random_vector(), v=random_vector(), a=random_vector(), r=1)
    t = np.random.rand()

    b1a = b1.a_at(t)
    b2 = b1.apply_impulse(t=t, dx=random_vector(), dv=random_vector())
    b2a = b2.a_at(t)

    assert np.allclose(b1a, b2a)

@pytest.mark.parametrize('n', range(5))
def test_impulse_dx_da(n):
    np.random.seed(n)
    b1 = Ball(x=random_vector(), v=random_vector(), a=random_vector(), r=1)
    t = np.random.rand()

    b1v = b1.v_at(t)
    b2 = b1.apply_impulse(t=t, dx=random_vector(), da=random_vector())
    b2v = b2.v_at(t)

    assert np.allclose(b1v, b2v)

@pytest.mark.parametrize('n', range(5))
def test_impulse_dv_da(n):
    np.random.seed(n)
    b1 = Ball(x=random_vector(), v=random_vector(), a=random_vector(), r=1)
    t = np.random.rand()

    b1x = b1.x_at(t)
    b2 = b1.apply_impulse(t=t, dv=random_vector(), da=random_vector())
    b2x = b2.x_at(t)

    assert np.allclose(b1x, b2x)

def test_collide():
    b1 = Ball(x=A([ 1.,0.]), v=A([-1.,0.]))
    m1 = 1.
    b2 = Ball(x=A([-1.,0.]), v=A([ 1.,0.]))
    m2 = 1.
    t = 0.
    e = 1.

    i1, i2 = b1.get_collision_impulse(b2, t=t).with_restitution(e).split(m1,m2)

    c1 = b1.apply_impulse(i1)
    c2 = b2.apply_impulse(i2)

    assert np.allclose(b1.x, c1.x)
    assert np.allclose(b2.x, c2.x)
    assert np.allclose(b1.v, -c1.v)
    assert np.allclose(b2.v, -c2.v)
    assert conservation_law_obeyed(t, centroid, [(b1,m1),(b2,m2)], [(c1,m1),(c2,m2)])
    assert conservation_law_obeyed(t, momentum, [(b1,m1),(b2,m2)], [(c1,m1),(c2,m2)])
    assert conservation_law_obeyed(t, energy, [(b1,m1),(b2,m2)], [(c1,m1),(c2,m2)])


def test_collide_misaligned():
    b1 = Ball(x=A([ 3.,0.]), v=A([-1.,0.]))
    m1 = 1.
    b2 = Ball(x=A([-1.,0.]), v=A([ 1.,0.]))
    m2 = 1.
    t = 0.
    e = 1.

    with pytest.warns():
        i = b1.get_collision_impulse(b2, t=t)

    i1, i2 = i.with_restitution(e).split(m1,m2)

    c1 = b1.apply_impulse(i1)
    c2 = b2.apply_impulse(i2)

    assert np.allclose(c1.x, A([2.,0.]))
    assert np.allclose(c2.x, A([0.,0.]))
    assert np.allclose(b1.v, -c1.v)
    assert np.allclose(b2.v, -c2.v)
    assert conservation_law_obeyed(t, centroid, [(b1,m1),(b2,m2)], [(c1,m1),(c2,m2)])
    assert conservation_law_obeyed(t, momentum, [(b1,m1),(b2,m2)], [(c1,m1),(c2,m2)])
    assert conservation_law_obeyed(t, energy, [(b1,m1),(b2,m2)], [(c1,m1),(c2,m2)])

def test_collide_immovable_object():
    b1 = Ball(x=A([ 1.,0.]), v=A([0.,0.]))
    m1 = np.inf
    b2 = Ball(x=A([-1.,0.]), v=A([ 1.,0.]))
    m2 = 1.
    t = 0.
    e = 1.

    i1, i2 = b1.get_collision_impulse(b2, t=t).with_restitution(e).split(m1,m2)

    c1 = b1.apply_impulse(i1)
    c2 = b2.apply_impulse(i2)

    assert np.allclose(b1.x, c1.x)
    assert np.allclose(b2.x, c2.x)
    assert np.allclose(b1.v, c1.v)
    assert np.allclose(b2.v, -c2.v)
    assert conservation_law_obeyed(t, energy, [(b2,m2)], [(c2,m2)])

def test_collide_immovable_object_and_unstoppable_force():
    b1 = Ball(x=A([ 1.,0.]), v=A([0.,0.]))
    m1 = np.inf
    b2 = Ball(x=A([-1.,0.]), v=A([ 1.,0.]))
    m2 = np.inf
    t = 0.
    e = 1.

    i1, i2 = b1.get_collision_impulse(b2, t=t).with_restitution(e).split(m1,m2)

    c1 = b1.apply_impulse(i1)
    c2 = b2.apply_impulse(i2)

    assert np.all(np.isnan(c1.x))
    assert np.all(np.isnan(c2.x))
    assert np.all(np.isnan(c1.v))
    assert np.all(np.isnan(c2.v))

def test_collide_inelastic():
    b1 = Ball(x=A([ 1.,0.]), v=A([-1.,0.]))
    m1 = 1.
    b2 = Ball(x=A([-1.,0.]), v=A([ 1.,0.]))
    m2 = 1.
    t = 0.
    e = 0.

    i1, i2 = b1.get_collision_impulse(b2, t=t).with_restitution(e).split(m1,m2)

    c1 = b1.apply_impulse(i1)
    c2 = b2.apply_impulse(i2)

    assert np.allclose(b1.x, c1.x)
    assert np.allclose(b2.x, c2.x)
    assert np.allclose(c1.v, A([0.,0.]))
    assert np.allclose(c2.v, A([0.,0.]))
    assert conservation_law_obeyed(t, centroid, [(b1,m1),(b2,m2)], [(c1,m1),(c2,m2)])
    assert conservation_law_obeyed(t, momentum, [(b1,m1),(b2,m2)], [(c1,m1),(c2,m2)])
    assert np.isclose(energy(t, [(c1,m1),(c2,m2)]), 0.)

def test_collide_small_vs_large():
    b1 = Ball(x=A([ 1.,0.]), v=A([-1.,0.]))
    m1 = 10.
    b2 = Ball(x=A([-1.,0.]), v=A([ 1.,0.]))
    m2 = 1.
    t = 0.
    e = 1.

    i1, i2 = b1.get_collision_impulse(b2, t=t).with_restitution(e).split(m1,m2)

    c1 = b1.apply_impulse(i1)
    c2 = b2.apply_impulse(i2)

    assert np.allclose(b1.x, c1.x)
    assert np.allclose(b2.x, c2.x)
    assert np.linalg.norm(c2.v) > np.linalg.norm(c1.v)
    assert conservation_law_obeyed(t, centroid, [(b1,m1),(b2,m2)], [(c1,m1),(c2,m2)])
    assert conservation_law_obeyed(t, momentum, [(b1,m1),(b2,m2)], [(c1,m1),(c2,m2)])
    assert conservation_law_obeyed(t, energy, [(b1,m1),(b2,m2)], [(c1,m1),(c2,m2)])