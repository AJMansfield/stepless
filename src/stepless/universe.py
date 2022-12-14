
from dataclasses import dataclass, field
import itertools
from typing import ClassVar
from uuid import UUID, uuid4

from stepless.ball import Ball
from stepless.ballview import ImpulseableVarDescriptor, SetttableVarDescriptor, VarDescriptor
from stepless.types import scalar_T
from stepless.util import dot

@dataclass
class Universe:
    t: scalar_T
    contents: dict[UUID, Ball] = field(default_factory=dict)
    modified: set[UUID] = field(default_factory=set)

    def mk_key(self):
        return uuid4()

    def add(self, obj: Ball) -> 'BallUniverseView':
        key = self.mk_key()
        self.contents[key] = obj
        return BallUniverseView(self, key)

    def get(self, key: UUID) -> 'BallUniverseView':
        return BallUniverseView(self, key)
    
    def advance_past_next_collision(self):
        t, ka, kb = min(self._compute_next_collision_times())
        a = self.contents[ka]
        b = self.contents[kb]
        i = a.get_collision_impulse(b, t)
        i = i.with_restitution(dot(a.b,b.b))
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
    
    def __iter__(self):
        for key in self.contents:
            yield BallUniverseView(self, key)

@dataclass
class BallUniverseView:
    universe: Universe
    key: UUID

    @property
    def ball(self):
        return self.universe.contents[self.key]
    @ball.setter
    def _(self, value):
        self.universe.contents[self.key] = value
        self.universe.modified.add(self.key)

    @property
    def modify_inplace(self):
        False

    @property
    def t(self):
        return self.universe.t

    x: ClassVar = ImpulseableVarDescriptor()
    v: ClassVar = ImpulseableVarDescriptor()
    a: ClassVar = ImpulseableVarDescriptor()
    r: ClassVar = SetttableVarDescriptor()
    m: ClassVar = SetttableVarDescriptor()
    b: ClassVar = SetttableVarDescriptor()
    P: ClassVar = ImpulseableVarDescriptor()
    F: ClassVar = ImpulseableVarDescriptor()
    U: ClassVar = VarDescriptor()
    K: ClassVar = VarDescriptor()
    E: ClassVar = VarDescriptor()