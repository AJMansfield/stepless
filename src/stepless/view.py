from dataclasses import dataclass, replace
from stepless.types import scalar_T, vector_T
from stepless.ball import Ball
from uuid import UUID, uuid4
import itertools

@dataclass
class Universe:
    t: scalar_T
    contents: dict[UUID, Ball]
    def mk_key(self):
        return uuid4()

    def add(self, obj: Ball) -> 'BallView':
        key = self.mk_key()
        self.contents[key] = obj
        return BallView(self, key)
    
    def advance_past_next_collision(self):
        t, ka, kb = min(self._compute_next_collision_times())
        a = self.contents[ka]
        b = self.contents[kb]
        i = a.get_collision_impulse(b, t)
        i = i.with_restitution(1.)
        ia, ib = i.split(a, b)
        a.apply_impulse(ia)
        b.apply_impulse(ib)
        self.contents[ka] = a
        self.contents[kb] = b
        self.t = t
        return self
    
    def _compute_next_collision_times(self):
        for ka,kb in itertools.combinations(self.contents, 2):
            a = self.contents[ka]
            b = self.contents[kb]
            yield a.compute_next_collision_time(b), ka,kb

@dataclass
class BallView:
    universe: Universe
    key: UUID

    @property
    def t(self):
        return self.universe.t
    @property
    def ball(self):
        return self.universe.contents[self.key]
    
    @ball.setter
    def ball(self, new_ball):
        self.universe.contents[self.key] = new_ball
    
    @property
    def x(self) -> vector_T:
        return self.ball.x_at(self.t)
    @x.setter
    def x(self, new_x):
        self.ball = self.ball.apply_state(self.t, x=new_x)
        
    @property
    def v(self) -> vector_T:
        return self.ball.v_at(self.t)
    @v.setter
    def v(self, new_v):
        self.ball = self.ball.apply_state(self.t, v=new_v)
        
    @property
    def a(self) -> vector_T:
        return self.ball.a_at(self.t)
    @a.setter
    def a(self, new_a):
        self.ball = self.ball.apply_state(self.t, a=new_a)
        
    @property
    def r(self) -> vector_T:
        return self.ball.r_at(self.t)
    @r.setter
    def r(self, new_r):
        self.ball = replace(self.ball, r=new_r)
        
    @property
    def m(self) -> vector_T:
        return self.ball.m_at(self.t)
    @m.setter
    def m(self, new_m):
        self.ball = replace(self.ball, m=new_m)
        
    @property
    def P(self) -> vector_T:
        return self.ball.P_at(self.t)
    @P.setter
    def P(self, new_P):
        self.ball = self.ball.apply_state(self.t, P=new_P)
        
    @property
    def F(self) -> vector_T:
        return self.ball.F_at(self.t)
    @F.setter
    def F(self, new_F):
        self.ball = self.ball.apply_state(self.t, F=new_F)
        
    @property
    def U(self) -> scalar_T:
        return self.ball.U_at(self.t)
    @property
    def E(self) -> scalar_T:
        return self.ball.E_at(self.t)
        