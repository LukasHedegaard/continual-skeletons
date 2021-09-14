import pickle

import numpy as np
import ride
import torch
from pytorch_lightning.utilities.parsing import AttributeDict
from ride.core import RideClassificationDataset
from torch.utils.data import DataLoader, Dataset

from datasets import kinetics, ntu_rgbd, tools


class GraphDatasets(RideClassificationDataset):
    @staticmethod
    def configs() -> ride.Configs:
        c = RideClassificationDataset.configs()
        c.add(
            name="dataset_name",
            type=str,
            default="dummy",
            choices=["ntu60", "ntu120", "kinetics", "dummy"],
            description="Name of dataset",
        )
        c.add(
            name="dataset_random_choose",
            type=int,
            default=0,
            choices=[0, 1],
            strategy="choice",
            description="Randomly choose a portion of the input sequence during training.",
        )
        c.add(
            name="dataset_random_shift",
            type=int,
            default=0,
            choices=[0, 1],
            strategy="choice",
            description="Randomly pad zeros at the begining or end of sequence during training.",
        )
        c.add(
            name="dataset_random_move",
            type=int,
            default=0,
            choices=[0, 1],
            strategy="choice",
            description="Randomly move joints during training.",
        )
        c.add(
            name="dataset_normalization",
            type=int,
            default=0,
            choices=[0, 1],
            strategy="choice",
            description="Normalize input sequence.",
        )
        c.add(
            name="dataset_window_size",
            type=int,
            default=-1,
            strategy="constant",
            description=(
                "The length of the output sequence. "
                "If insufficient frames are available, zero padding is used."
            ),
        )
        c.add(
            name="dataset_classes",
            type=str,
            default="",
            strategy="constant",
            description="Path to .yaml list of dataset classes.",
        )
        c.add(
            name="dataset_input_channels",
            type=int,
            default=3,
            strategy="constant",
            description="Number of input channels in dataset.",
        )
        c.add(
            name="dataset_train_data",
            type=str,
            default="",
            strategy="constant",
            description="Path to .npy training data.",
        )
        c.add(
            name="dataset_val_data",
            type=str,
            default="",
            strategy="constant",
            description="Path to .npy val data.",
        )
        c.add(
            name="dataset_test_data",
            type=str,
            default="",
            strategy="constant",
            description="Path to .npy test data. If none is supplied, `dataset_val_data` is used.",
        )
        c.add(
            name="dataset_train_labels",
            type=str,
            default="",
            strategy="constant",
            description="Path to .pkl train labels.",
        )
        c.add(
            name="dataset_val_labels",
            type=str,
            default="",
            strategy="constant",
            description="Path to .pkl val labels.",
        )
        c.add(
            name="dataset_test_labels",
            type=str,
            default="",
            strategy="constant",
            description="Path to .pkl test labels. If none is supplied, `dataset_val_labels` are used.",
        )
        return c

    def __init__(self, hparams: AttributeDict):
        super().__init__(hparams)

        C_in = self.hparams.dataset_input_channels
        (self.output_shape, self.input_shape, self.graph) = {
            "dummy": ((60,), (C_in, 300, ntu_rgbd.NUM_NODES, 2), ntu_rgbd.graph),
            "ntu60": ((60,), (C_in, 300, ntu_rgbd.NUM_NODES, 2), ntu_rgbd.graph),
            "ntu120": ((120,), (C_in, 300, ntu_rgbd.NUM_NODES, 2), ntu_rgbd.graph),
            "kinetics": ((400,), (C_in, 300, kinetics.NUM_NODES, 2), kinetics.graph),
        }[self.hparams.dataset_name]

        self.classes = (
            [str(i) for i in range(self.output_shape[0])]
            if self.hparams.dataset_name == "dummy"
            else ride.utils.io.load_yaml(self.hparams.dataset_classes)
        )
        assert len(self.classes) == self.output_shape[0]

        Ds = DummyDataset if self.hparams.dataset_name == "dummy" else GraphDataset

        train_args = dict(
            random_choose=self.hparams.dataset_random_choose,
            random_shift=self.hparams.dataset_random_shift,
            random_move=self.hparams.dataset_random_move,
            window_size=self.hparams.dataset_window_size,
            normalization=self.hparams.dataset_normalization,
        )

        dataloader_args = dict(
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.num_workers > 1,
        )

        self._train_dataloader = DataLoader(
            dataset=Ds(
                data_path=self.hparams.dataset_train_data,
                label_path=self.hparams.dataset_train_labels,
                **train_args
            ),
            shuffle=True,
            **dataloader_args
        )
        self._val_dataloader = DataLoader(
            dataset=Ds(
                data_path=self.hparams.dataset_val_data,
                label_path=self.hparams.dataset_val_labels,
            ),
            shuffle=False,
            **dataloader_args
        )
        self._test_dataloader = DataLoader(
            Ds(
                data_path=self.hparams.dataset_test_data
                or self.hparams.dataset_val_data,
                label_path=self.hparams.dataset_test_labels
                or self.hparams.dataset_val_labels,
            ),
            shuffle=False,
            **dataloader_args
        )

    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    def val_dataloader(self) -> DataLoader:
        return self._val_dataloader

    def test_dataloader(self) -> DataLoader:
        return self._test_dataloader


class GraphDataset(Dataset):
    def __init__(
        self,
        data_path,
        label_path,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=-1,
        normalization=False,
        mmap_mode="r",
    ):
        """Initialise a Graph dataset

        Args:
            data_path ([type]): Path to data
            label_path ([type]): Path to labels
            random_choose (bool, optional): Randomly choose a portion of the input sequence. Defaults to False.
            random_shift (bool, optional): Randomly pad zeros at the begining or end of sequence. Defaults to False.
            random_move (bool, optional): Randomly move joints. Defaults to False.
            window_size (int, optional): The length of the output sequence. Defaults to -1.
            normalization (bool, optional): Normalize input sequence. Defaults to False.
            mmap_mode (str, optional): Use mmap mode to load data, which can save the running memory. Defaults to "r".
        """
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.mmap_mode = mmap_mode
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # Data: N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except Exception:
            # For pickle file from python2
            with open(self.label_path, "rb") as f:
                self.sample_name, self.label = pickle.load(f, encoding="latin1")

        self.data = np.load(self.data_path, mmap_mode=self.mmap_mode)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = (
            data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        )
        self.std_map = (
            data.transpose((0, 2, 4, 1, 3))
            .reshape((N * T * M, C * V))
            .std(axis=0)
            .reshape((C, 1, V, 1))
        )

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label, index


class DummyDataset(Dataset):
    def __init__(
        self,
        input_shape=(3, 300, ntu_rgbd.NUM_NODES, 2),
        num_classes=60,
        num_samples=100,
        *args,
        **kwargs
    ):
        # num_channels, num_frames, num_vertices, num_skeletons = self.input_shape
        self.input_shape = input_shape
        self.output_shape = (num_classes,)
        self.data = torch.rand(size=(num_samples, *input_shape), dtype=torch.float32)
        self.labels = torch.randint(low=0, high=num_classes, size=self.output_shape)

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        return self.data[index]
