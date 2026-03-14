import logging
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
import slangpy as spy
from functools import reduce
import heapq

from cs248a_renderer.model.bounding_box import BoundingBox3D
from cs248a_renderer.model.primitive import Primitive
from tqdm import tqdm


logger = logging.getLogger(__name__)


@dataclass
class BVHNode:
    # The bounding box of this node.
    bound: BoundingBox3D = field(default_factory=BoundingBox3D)
    # The index of the left child node, or -1 if this is a leaf node.
    left: int = -1
    # The index of the right child node, or -1 if this is a leaf node.
    right: int = -1
    # The starting index of the primitives in the primitives array.
    prim_left: int = 0
    # The ending index (exclusive) of the primitives in the primitives array.
    prim_right: int = 0
    # The depth of this node in the BVH tree.
    depth: int = 0

    def get_this(self) -> dict:
        return {
            "bound": self.bound.get_this(),
            "left": self.left,
            "right": self.right,
            "primLeft": self.prim_left,
            "primRight": self.prim_right,
            "depth": self.depth,
        }

    @property
    def is_leaf(self) -> bool:
        """Checks if this node is a leaf node."""
        return self.left == -1 and self.right == -1

@dataclass(order=True)
class PriorityItem:
    priority: int
    item: Any = field(compare=False)


class BVH:
    def __init__(
        self,
        primitives: list[Primitive],
        max_nodes: int,
        min_prim_per_node: int = 1,
        num_thresholds: int = 16,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Builds the BVH from the given list of primitives. The build algorithm should
        reorder the primitives in-place to align with the BVH node structure.
        The algorithm will start from the root node and recursively partition the primitives
        into child nodes until the maximum number of nodes is reached or the primitives
        cannot be further subdivided.
        At each node, the splitting axis and threshold should be chosen using the Surface Area Heuristic (SAH)
        to minimize the expected cost of traversing the BVH during ray intersection tests.

        :param primitives: the list of primitives to build the BVH from
        :type primitives: List[Primitive]
        :param max_nodes: the maximum number of nodes in the BVH
        :type max_nodes: int
        :param min_prim_per_node: the minimum number of primitives per leaf node
        :type min_prim_per_node: int
        :param num_thresholds: the number of thresholds per axis to consider when splitting
        :type num_thresholds: int
        """
        self.nodes: list[BVHNode] = []

        # TODO: Student implementation starts here.
        
        AxisKey = {
            "X": lambda p: p.bounding_box.center.x,
            "Y": lambda p: p.bounding_box.center.y,
            "Z": lambda p: p.bounding_box.center.z,
        }

        large_leaves: list[PriorityItem] = []

        def split_node(node: BVHNode, primitives: list[Primitive]):
            best_H = float("inf")
            best_primsort: list[Primitive] = []
            nodeL: BVHNode
            nodeR: BVHNode
            for _ax,fn in AxisKey.items():
                primsort = sorted(primitives[node.prim_left:node.prim_right+1], key=fn)
                for n in range(num_thresholds):
                    partition = (n+1) * ((node.prim_right - node.prim_left) // (num_thresholds + 1))
                    leftBB = reduce(
                        lambda a,b: BoundingBox3D.union(a,b),
                        (p.bounding_box for p in primsort[:partition+1]),
                        BoundingBox3D()
                    )
                    rightBB = reduce(
                        lambda a,b: BoundingBox3D.union(a,b),
                        (p.bounding_box for p in primsort[partition+1:]),
                        BoundingBox3D()
                    )

                    sizeL = partition
                    sizeR = node.prim_right - node.prim_left - partition - 1
                    
                    # surface area heuristic
                    H = 24 + 56 * ((leftBB.area / node.bound.area) * sizeL + (rightBB.area / node.bound.area) * sizeR)
                    if H < best_H:
                        best_H = H
                        best_primsort = primsort

                        nodeL = BVHNode(leftBB, -1, -1, node.prim_left, node.prim_left + partition, node.depth+1)
                        nodeR = BVHNode(rightBB, -1, -1, node.prim_left + partition+1, node.prim_right, node.depth+1)
            
            node.left = len(self.nodes)
            sizeL = nodeL.prim_right - nodeL.prim_left + 1
            if sizeL > min_prim_per_node:
                heapq.heappush(large_leaves, PriorityItem(-sizeL, node.left))
            self.nodes.append(nodeL)

            node.right = len(self.nodes)
            sizeR = nodeR.prim_right - nodeR.prim_left + 1
            if sizeR > min_prim_per_node:
                heapq.heappush(large_leaves, PriorityItem(-sizeR, node.right))
            self.nodes.append(nodeR)

            primitives[node.prim_left:node.prim_right+1] = best_primsort

        fullBB = reduce(
            lambda a,b: BoundingBox3D.union(a, b), 
            (p.bounding_box for p in primitives),
            BoundingBox3D()
        )
        root = BVHNode(fullBB, -1, -1, 0, len(primitives) - 1, 0)
        self.nodes.append(root)
        on_progress(len(self.nodes), max_nodes)

        heapq.heappush(large_leaves, PriorityItem(-len(primitives) + 1, 0))

        while len(self.nodes) < max_nodes - 2: # need to be able to add 2
            try:
                leaf = heapq.heappop(large_leaves).item
                split_node(self.nodes[leaf], primitives)
                on_progress(len(self.nodes), max_nodes)
            except IndexError:
                # Nothing left to split
                break

        # TODO: Student implementation ends here.


def create_bvh_node_buf(module: spy.Module, bvh_nodes: list[BVHNode]) -> spy.NDBuffer:
    device = module.device
    node_buf = spy.NDBuffer(
        device=device, dtype=module.BVHNode.as_struct(), shape=(max(len(bvh_nodes), 1),)
    )
    cursor = node_buf.cursor()
    for idx, node in enumerate(bvh_nodes):
        cursor[idx].write(node.get_this())
    cursor.apply()
    return node_buf
