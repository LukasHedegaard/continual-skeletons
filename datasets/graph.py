from typing import List, Tuple

import numpy as np
from ride.logging import getLogger

logger = getLogger(__name__)


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    _, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)  # noqa: E741
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


class Graph:
    def __init__(self, inward: List[Tuple[int, int]], num_node: int):
        self.num_node = num_node
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.inward = inward
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        self.A = get_spatial_graph(
            self.num_node, self.self_link, self.inward, self.outward
        )

    def print(self, image=False):
        logger.info("Graph Adjacency Matrix:")
        logger.info(self.A)

        if image:
            import matplotlib.pyplot as plt

            for i in self.A:
                plt.imshow(i, cmap="gray")
                plt.show()
