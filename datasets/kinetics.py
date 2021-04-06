from datasets.graph import Graph

# Joint index:
# {0,  "Nose"}
# {1,  "Neck"},
# {2,  "RShoulder"},
# {3,  "RElbow"},
# {4,  "RWrist"},
# {5,  "LShoulder"},
# {6,  "LElbow"},
# {7,  "LWrist"},
# {8,  "RHip"},
# {9,  "RKnee"},
# {10, "RAnkle"},
# {11, "LHip"},
# {12, "LKnee"},
# {13, "LAnkle"},
# {14, "REye"},
# {15, "LEye"},
# {16, "REar"},
# {17, "LEar"},

# Edge format: (origin, neighbor)
INWARD = [
    (4, 3),
    (3, 2),
    (7, 6),
    (6, 5),
    (13, 12),
    (12, 11),
    (10, 9),
    (9, 8),
    (11, 5),
    (8, 2),
    (5, 1),
    (2, 1),
    (0, 1),
    (15, 0),
    (14, 0),
    (17, 15),
    (16, 14),
]

NUM_NODES = 18

graph = Graph(inward=INWARD, num_node=NUM_NODES)
