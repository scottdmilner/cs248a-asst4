from __future__ import annotations
from dataclasses import dataclass, field

from cs248a_renderer.model.transforms import Transform3D

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyglm import glm


# SceneObject index to generate unique IDs.
_scene_object_index = 0


def get_next_scene_object_index() -> int:
    global _scene_object_index
    idx = _scene_object_index
    _scene_object_index += 1
    return idx


@dataclass
class SceneObject:
    name: str = field(default_factory=lambda: f"object_{get_next_scene_object_index()}")
    transform: Transform3D = field(default_factory=Transform3D)
    # Parent SceneObject. None if this is the root object.
    parent: SceneObject | None = None
    children: list[SceneObject] = field(default_factory=list)

    def get_transform_matrix(self) -> glm.mat4:
        """Compute the world transform matrix by accumulating parent transforms.

        :return: The 4x4 world transform matrix.
        """
        # TODO: Student implementation starts here.

        curr = self
        out = self.transform.get_matrix()

        while curr := curr.parent:
            out @= curr.transform.get_matrix()
        
        return out

        # TODO: Student implementation ends here.

    def desc(self, depth: int = 0) -> str:
        indent = "  " * depth
        desc_str = f"{indent}SceneObject(name={self.name}, transform={self.transform}, children=[\n"
        for child in self.children:
            desc_str += child.desc(depth + 1) + "\n"
        desc_str += f"{indent}])"
        return desc_str

    def __repr__(self):
        return self.desc()
