from pyglm import glm
from dataclasses import dataclass, field
from enum import Enum
import slangpy as spy


MAX_BRANCHES = 0

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

@dataclass
class LPEEvent:
    ray: RayType
    scatter: ScatterType


@dataclass
class LPEState:
    currColor: glm.vec4 = glm.vec4(0.0, 0.0, 0.0, 1.0)
    next_idxs: list[int] = field(default_factory=lambda: [0] * MAX_BRANCHES)
    event: LPEEvent = field(default_factory=lambda: LPEEvent(RayType.Any, ScatterType.Any))

    def get_this(self) -> dict:
        return {}


@dataclass
class LPE:
    nodes: list[LPEState]
    node_active: list[bool]
    currentNode: int = 0
    isActive: bool = True
    completed: bool = False

    def update_param(self):
        pass

    def get_this(self) -> dict:
        self.update_param()

        return {
            "nodes": [
                {
                    "currColor": glm.vec4(0.0, 0.0, 0.0, 1.0),
                    "next_idxs": [1,0,0,0],
                    "event": {"ray": RayType.Camera.value, "scatter": ScatterType.Any.value},
                },
                {
                    "currColor": glm.vec4(0.0, 0.0, 0.0, 1.0),
                    "next_idxs": [1,2,0,0],
                    "event": {"ray": RayType.Any.value, "scatter": ScatterType.Any.value},
                },
                {
                    "currColor": glm.vec4(0.0, 0.0, 0.0, 1.0),
                    "next_idxs": [0,0,0,0],
                    "event": {"ray": RayType.Light.value, "scatter": ScatterType.Any.value},
                },
                *[
                    {
                        "currColor": glm.vec4(0.0, 0.0, 0.0, 1.0),
                        "next_idxs": [1,0,0,0],
                        "event": {"ray": RayType.Camera.value, "scatter": ScatterType.Any.value},
                    }
                    for i in range(13) # pad out the 16
                ]
            ],
            "node_active": [True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
            "currentNode": 0,
            "isActive": True,
            "completed": False
        }



def create_lpe_buf(
    module: spy.Module, lpes: list[LPE]
) -> spy.NDBuffer:
    device = module.device
    buffer = spy.NDBuffer(
        device=device,
        dtype=module["LPEAutomaton<float4>"].as_struct(),
        shape=(max(len(lpes), 1),),
    )
    cursor = buffer.cursor()
    for idx, lpe in enumerate([LPE([],[],0,True)]): # test LPE
        cursor[idx].write(lpe.get_this())
    cursor.apply()

    return buffer