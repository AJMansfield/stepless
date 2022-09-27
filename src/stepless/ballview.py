
from dataclasses import dataclass, replace
from typing import ClassVar

from stepless.ball import Ball
from stepless.types import scalar_T


class VarDescriptor:
    def __init__(self, name:str=None):
        self.var = name
    def __set_name__(self, owner, name):
        if self.var is None:
            self.var = name
    def __get__(self, obj: 'BallView', objtype=None):
        if obj is None: raise AttributeError
        return getattr(obj, self.var+"_at")(obj.t)

class SetttableVarDescriptor(VarDescriptor):
    def __set__(self, obj: 'BallView', value):
        if obj.modify_inplace:
            setattr(obj.ball, self.var, value)
        else:
            obj.ball = replace(obj.ball, **{self.var: value})

class ImpulseableVarDescriptor(VarDescriptor):
    def __set__(self, obj: 'BallView', value):
        if obj.modify_inplace:
            obj.ball.apply_state(t=obj.t, **{self.var: value}, inplace=True)
        else:
            obj.ball = obj.ball.apply_state(t=obj.t, **{self.var: value})


@dataclass
class BallView:
    ball: Ball
    t: scalar_T
    modify_inplace: bool = False

    x: ClassVar = ImpulseableVarDescriptor()
    v: ClassVar = ImpulseableVarDescriptor()
    a: ClassVar = ImpulseableVarDescriptor()
    r: ClassVar = SetttableVarDescriptor()
    m: ClassVar = SetttableVarDescriptor()
    P: ClassVar = ImpulseableVarDescriptor()
    F: ClassVar = ImpulseableVarDescriptor()
    U: ClassVar = VarDescriptor()
    K: ClassVar = VarDescriptor()
    E: ClassVar = VarDescriptor()

