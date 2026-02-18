import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable
import numpy as np
import slangpy as spy

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

    def get_this(self) -> Dict:
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


class BVH:
    def __init__(
        self,
        primitives: List[Primitive],
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
        self.nodes: List[BVHNode] = []

        # TODO: Student implementation starts here.

        self.max_layers = int(np.log2(max_nodes)) - 1
        self.primitives = primitives
        self.max_nodes = max_nodes
        self.min_prim_per_node = min_prim_per_node
        self.num_thresholds = num_thresholds
        self.on_progress = on_progress
        self.axis_ranges = []
        
        root = BVHNode(prim_left=0, prim_right=len(primitives))
        for primitive in primitives:
            root.bound = BoundingBox3D.union(root.bound, primitive.bounding_box)
        self.nodes.append(root)
        self.create_nodes_from(root)

        # TODO: Student implementation ends here.

    def create_nodes_from(self, node: BVHNode):
        if node.depth >= self.max_layers or node.prim_right - node.prim_left <= self.min_prim_per_node:
            return
        
        min_sah = float("inf")
        split_pos = 0
        axis = 0
        left_bbox = BoundingBox3D()
        right_bbox = BoundingBox3D()
        for i in range(3):
            axis_min = float("inf")
            axis_max = float("-inf")
            for j in range(node.prim_left, node.prim_right):
                axis_min = min(axis_min, self.primitives[j].bounding_box.center[i])
                axis_max = max(axis_max, self.primitives[j].bounding_box.center[i])

            bucket_bboxes = [BoundingBox3D() for _ in range(self.num_thresholds)]
            bucket_prim_counts = [0 for _ in range(self.num_thresholds)]

            for j in range(node.prim_left, node.prim_right):
                prim_bbox = self.primitives[j].bounding_box
                bucket = int(self.num_thresholds * (prim_bbox.center[i] - axis_min) / (axis_max - axis_min))
                bucket = max(0, min(bucket, self.num_thresholds - 1))
                bucket_prim_counts[bucket] += 1
                bucket_bboxes[bucket] = BoundingBox3D.union(bucket_bboxes[bucket], prim_bbox)

            left_bboxes = [BoundingBox3D() for _ in range(self.num_thresholds)]
            left_prim_count = [0 for _ in range(self.num_thresholds)]
            right_bboxes = [BoundingBox3D() for _ in range(self.num_thresholds)]
            right_prim_count = [0 for _ in range(self.num_thresholds)]

            left_bboxes[0] = bucket_bboxes[0]
            left_prim_count[0] = bucket_prim_counts[0]
            right_bboxes[-1] = bucket_bboxes[-1]
            right_prim_count[-1] = bucket_prim_counts[-1]

            for j in range(1, self.num_thresholds):
                left_bboxes[j] = BoundingBox3D.union(left_bboxes[j-1], bucket_bboxes[j])
                left_prim_count[j] = left_prim_count[j - 1] + bucket_prim_counts[j]
                right_bboxes[self.num_thresholds - 1 - j] = BoundingBox3D.union(right_bboxes[self.num_thresholds - j], bucket_bboxes[self.num_thresholds - 1 - j])
                right_prim_count[self.num_thresholds - 1 - j] = right_prim_count[self.num_thresholds - j] + bucket_prim_counts[self.num_thresholds - 1 - j]

            section_size = (axis_max - axis_min) / self.num_thresholds
            for j in range(0, self.num_thresholds - 1):
                sah = left_bboxes[j].area * left_prim_count[j] + right_bboxes[j + 1].area * right_prim_count[j + 1]
                if sah < min_sah:
                    min_sah = sah
                    axis = i
                    split_pos = axis_min + section_size * (j + 1)
                    left_bbox = left_bboxes[j]
                    right_bbox = right_bboxes[j + 1]

        l = node.prim_left
        r = node.prim_right - 1
        while l <= r:
            if self.primitives[l].bounding_box.center[axis] <= split_pos:
                l += 1
            else:
                prim = self.primitives[l]
                self.primitives[l] = self.primitives[r]
                self.primitives[r] = prim
                r -= 1

        left_child = BVHNode(bound=left_bbox, prim_left=node.prim_left, prim_right=l, depth=node.depth + 1)
        node.left = len(self.nodes)
        self.nodes.append(left_child)
        right_child = BVHNode(bound=right_bbox, prim_left=l, prim_right=node.prim_right, depth=node.depth + 1)
        node.right = len(self.nodes)
        self.nodes.append(right_child)

        self.on_progress(len(self.nodes), self.max_nodes)

        self.create_nodes_from(left_child)
        self.create_nodes_from(right_child)

def create_bvh_node_buf(module: spy.Module, bvh_nodes: List[BVHNode]) -> spy.NDBuffer:
    device = module.device
    node_buf = spy.NDBuffer(
        device=device, dtype=module.BVHNode.as_struct(), shape=(max(len(bvh_nodes), 1),)
    )
    cursor = node_buf.cursor()
    for idx, node in enumerate(bvh_nodes):
        cursor[idx].write(node.get_this())
    cursor.apply()
    return node_buf
