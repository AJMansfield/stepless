
from dataclasses import InitVar, dataclass, field
from uuid import UUID
import heapq
from stepless.universe import Universe
from stepless.types import scalar_T
from stepless.util import dot
import numpy as np

class TimeTravelError(Exception):
    pass



@dataclass(slots=True, unsafe_hash=True)
class CollisionHeapKey:
    k1: UUID
    k2: UUID
    def __post_init__(self):
        self.k1, self.k2 = sorted((self.k1, self.k2))

@dataclass(order=True, slots=True)
class CollisionHeapItem:
    universe: InitVar[Universe]
    key: CollisionHeapKey = field(compare=False)
    void: bool = field(default=False, compare=False)
    t: scalar_T = None
    def __post_init__(self, universe):
        self.t = universe.contents[self.key.k1].compute_next_collision_time(universe.contents[self.key.k2], universe.t)

@dataclass
class CollisionHeap:
    _heap: list[CollisionHeapItem] = field(default_factory=list)
    _map: dict[CollisionHeapKey, CollisionHeapItem] = field(default_factory=dict) # so we can void old collisions
    void_count: int = 0
    entry_count: int = 0

    def push(self, timeline: 'Timeline', k1: UUID, k2: UUID):
        key = CollisionHeapKey(k1, k2)
        item = CollisionHeapItem(timeline, key)
        self._push(item)

    def _push(self, item: CollisionHeapItem):
        if item.key in self:
            self._map[item.key].void = True # a recomputed collision time automatically supercedes the old collision time
            self.void_count += 1 
            del self._map[item.key]

        if np.isfinite(item.t): # infinite = they don't collide; don't need to store that
            self._map[item.key] = item
            heapq.heappush(self._heap, item)
            self.entry_count += 1

    def next(self) -> scalar_T:
        while self._heap and self._heap[0].void:
            self._pop()
        if self._heap:
            return self._heap[0].t
        else:
            return np.inf

    def pop(self) -> tuple[scalar_T, UUID, UUID]:
        item = self._pop()
        return item.t, item.key.k1, item.key.k2
        
    def _pop(self) -> CollisionHeapItem:
        while self._heap:
            item = heapq.heappop(self._heap)
            self.entry_count -= 1
            if item.void:
                self.void_count -= 1
            else:
                del self._map[item.key]
                return item
    
    def __len__(self):
        return self.entry_count - self.void_count
    def __contains__(self, item: CollisionHeapKey):
        return item in self._map



@dataclass
class Timeline(Universe):

    future: CollisionHeap = field(default_factory=CollisionHeap)

    def recompute_future(self):
        unmodified = self.contents.keys() - self.modified
        while self.modified:
            k1 = self.modified.pop()
            for k2 in unmodified:
                self.future.push(self, k1, k2)
            unmodified.add(k1)

    def do_next_collision(self):
        t, k1, k2 = self.future.pop()
        b1 = self.contents[k1]
        b2 = self.contents[k2]
        i = b1.get_collision_impulse(b2, t)
        i = i.with_restitution(dot(b1.b,b2.b))
        i1, i2 = i.split(b1.m, b2.m)
        b1 += i1
        b2 += i2
        self.t = t
        self.modified.add(k1)
        self.modified.add(k2)
        self.recompute_future()
        

    def advance_to(self, t: scalar_T, allow_time_travel=False):
        if not allow_time_travel and t < self.t:
            raise TimeTravelError(f"Cannot step backwards from t={self.t} to t={t}.")
        if self.modified:
            self.recompute_future()
        while t > self.future.next():
            self.do_next_collision()
        self.t = t
    
    def add(self, obj: 'Ball') -> 'BallUniverseView':
        result = super().add(obj)
        self.modified.add(result.key)
        return result
        

