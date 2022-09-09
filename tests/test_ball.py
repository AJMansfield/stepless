import pytest
import numpy as np
from stepless.ball import Ball
from typing import Callable, Any
from numpy import array as A

from laws import centroid, momentum, kinetic_energy, assert_conservation_law_obeyed

def random_vector():
    return np.random.rand(2) * 10

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
    universe = []
    b1 = Ball(x=A([ 1.,0.]), v=A([-1.,0.]), m=1.)
    b2 = Ball(x=A([-1.,0.]), v=A([ 1.,0.]), m=1.)
    t = 0.
    e = 1.

    universe.append((t, [b1,b2]))

    i1, i2 = b1.get_collision_impulse(b2, t=t).with_restitution(e).split(b1,b2)

    c1 = b1.apply_impulse(i1)
    c2 = b2.apply_impulse(i2)

    universe.append((t, [c1,c2]))

    assert np.allclose(b1.x, c1.x)
    assert np.allclose(b2.x, c2.x)
    assert np.allclose(b1.v, -c1.v)
    assert np.allclose(b2.v, -c2.v)

    assert_conservation_law_obeyed(centroid, universe)
    assert_conservation_law_obeyed(momentum, universe)
    assert_conservation_law_obeyed(kinetic_energy, universe)


def test_collide_misaligned():
    universe = []
    b1 = Ball(x=A([ 3.,0.]), v=A([-1.,0.]), m=1.)
    b2 = Ball(x=A([-1.,0.]), v=A([ 1.,0.]), m=1.)
    t = 0.
    e = 1.

    universe.append((t, [b1,b2]))

    with pytest.warns():
        i = b1.get_collision_impulse(b2, t=t)

    i1, i2 = i.with_restitution(e).split(b1,b2)

    c1 = b1.apply_impulse(i1)
    c2 = b2.apply_impulse(i2)

    universe.append((t, [c1,c2]))

    assert np.allclose(c1.x, A([2.,0.]))
    assert np.allclose(c2.x, A([0.,0.]))
    assert np.allclose(b1.v, -c1.v)
    assert np.allclose(b2.v, -c2.v)

    assert_conservation_law_obeyed(centroid, universe)
    assert_conservation_law_obeyed(momentum, universe)
    assert_conservation_law_obeyed(kinetic_energy, universe)

def test_collide_immovable_object():
    universe = []
    b1 = Ball(x=A([ 1.,0.]), v=A([0.,0.]), m=np.inf)
    b2 = Ball(x=A([-1.,0.]), v=A([ 1.,0.]), m=1.)
    t = 0.
    e = 1.

    universe.append((t, [b2]))

    i1, i2 = b1.get_collision_impulse(b2, t=t).with_restitution(e).split(b1,b2)

    c1 = b1.apply_impulse(i1)
    c2 = b2.apply_impulse(i2)

    universe.append((t, [c2]))

    assert np.allclose(b1.v, c1.v)
    assert np.allclose(b2.v, -c2.v)

    assert_conservation_law_obeyed(kinetic_energy, universe)

def test_collide_immovable_object_and_unstoppable_force():
    b1 = Ball(x=A([ 1.,0.]), v=A([0.,0.]), m=np.inf)
    b2 = Ball(x=A([-1.,0.]), v=A([ 1.,0.]), m=np.inf)
    t = 0.
    e = 1.

    i1, i2 = b1.get_collision_impulse(b2, t=t).with_restitution(e).split(b1,b2)

    c1 = b1.apply_impulse(i1)
    c2 = b2.apply_impulse(i2)

    assert np.all(np.isnan(c1.x))
    assert np.all(np.isnan(c2.x))
    assert np.all(np.isnan(c1.v))
    assert np.all(np.isnan(c2.v))

def test_collide_inelastic():
    universe = []
    b1 = Ball(x=A([ 1.,0.]), v=A([-1.,0.]), m=1.)
    b2 = Ball(x=A([-1.,0.]), v=A([ 1.,0.]), m=1.)
    t = 0.
    e = 0. # inelastic!

    universe.append((t, [b1,b2]))

    i1, i2 = b1.get_collision_impulse(b2, t=t).with_restitution(e).split(b1,b2)

    c1 = b1.apply_impulse(i1)
    c2 = b2.apply_impulse(i2)

    universe.append((t, [c1,c2]))

    assert np.allclose(c1.v, A([0.,0.]))
    assert np.allclose(c2.v, A([0.,0.]))

    assert_conservation_law_obeyed(centroid, universe)
    assert_conservation_law_obeyed(momentum, universe)
    assert np.isclose(kinetic_energy(*universe[1]), 0.) # inelastic!

def test_collide_small_vs_large():
    universe = []
    b1 = Ball(x=A([ 1.,0.]), v=A([-1.,0.]), m=10.) # bigger ball
    b2 = Ball(x=A([-1.,0.]), v=A([ 1.,0.]), m=1.)
    t = 0.
    e = 1.

    universe.append((t, [b1,b2]))

    i1, i2 = b1.get_collision_impulse(b2, t=t).with_restitution(e).split(b1,b2)

    c1 = b1.apply_impulse(i1)
    c2 = b2.apply_impulse(i2)

    universe.append((t, [c1,c2]))

    print(f"{universe!r}")

    # ensure smaller one got shot away faster than the big one:
    assert np.linalg.norm(c2.v) > np.linalg.norm(c1.v)

    assert_conservation_law_obeyed(centroid, universe)
    assert_conservation_law_obeyed(momentum, universe)
    assert_conservation_law_obeyed(kinetic_energy, universe)