"""
Adapted from https://github.com/lshiwjx/2s-AGCN
"""


import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

sets = {"train", "val"}

datasets = {"ntu60/xview", "ntu60/xsub", "ntu120/xset", "ntu120/xsub", "kinetics"}

parts = {"joint", "bone"}

for dataset in datasets:
    for set in sets:
        for part in parts:
            print(dataset, set, part)
            data = np.load("./data/{}/{}_data_{}.npy".format(dataset, set, part))
            N, C, T, V, M = data.shape
            fp_sp = open_memmap(
                "./data/{}/{}_data_{}_motion.npy".format(dataset, set, part),
                dtype="float32",
                mode="w+",
                shape=(N, 3, T, V, M),
            )
            for t in tqdm(range(T - 1)):
                fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
            fp_sp[:, :, T - 1, :, :] = 0
