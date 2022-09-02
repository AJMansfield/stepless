import pytest
import numpy as np
from stepless.ball import Ball

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