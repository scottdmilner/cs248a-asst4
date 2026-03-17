from dataclasses import dataclass, field
from enum import Enum
import slangpy as spy


MAX_BRANCHES = 4
MAX_NODES = 16


class ScatterType(Enum):
    UND = 0
    Any = 1
    Diffuse = 2
    Specular = 3

class RayType(Enum):
    UND = 0
    Any = 1
    Camera = 2
    Reflection = 3
    Transmission = 4
    Light = 5
    Background = 6

@dataclass
class LPEEvent:
    ray: RayType
    scatter: ScatterType

    def get_this(self) -> dict:
        return {
            "ray": self.ray.value,
            "scatter": self.scatter.value,
        }


@dataclass
class LPEState:
    event: LPEEvent = field(default_factory=lambda: LPEEvent(RayType.Any, ScatterType.Any))
    next_idxs: list[int] = field(default_factory=lambda: [0] * MAX_BRANCHES)
    isTerminal: bool = False

    def get_this(self) -> dict:
        return {
            "event": self.event.get_this(),
            "next_idxs": self.next_idxs,
            "isTerminal": self.isTerminal,
        }


@dataclass
class LPE:
    nodes: list[LPEState]
    node_active: list[bool] = field(default_factory=lambda: [
        True if i == 0 else False for i in range(MAX_NODES)
    ])
    isActive: bool = True

    def get_this(self) -> dict:
        return {
            "nodes": [n.get_this() for n in [
                *self.nodes,
                *[LPEState(isTerminal=True) for i in range(MAX_NODES - len(self.nodes))]
            ]],
            "node_active": self.node_active,
            "isActive": self.isActive
        }


def create_lpe_buf(
    module: spy.Module, lpes: list[LPE]
) -> spy.NDBuffer:
    device = module.device
    buffer = spy.NDBuffer(
        device=device,
        dtype=module.LPEAutomaton.as_struct(),
        shape=(max(len(lpes), 1),),
    )
    cursor = buffer.cursor()

    for idx, lpe in enumerate(lpes): # test LPE
        cursor[idx].write(lpe.get_this())
    cursor.apply()

    return buffer