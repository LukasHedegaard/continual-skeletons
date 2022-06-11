"""
Adapted from https://github.com/lshiwjx/2s-AGCN
"""


import numpy as np

sets = {"train", "val"}

datasets = {"ntu60/xview", "ntu60/xsub", "ntu120/xset", "ntu120/xsub", "kinetics"}

for dataset in datasets:
    for set in sets:
        print(dataset, set)
        data_jpt = np.load("./data/{}/{}_data_joint.npy".format(dataset, set))
        data_bone = np.load("./data/{}/{}_data_bone.npy".format(dataset, set))
        N, C, T, V, M = data_jpt.shape
        data_jpt_bone = np.concatenate((data_jpt, data_bone), axis=1)
        np.save("./data/{}/{}_data_joint_bone.npy".format(dataset, set), data_jpt_bone)
