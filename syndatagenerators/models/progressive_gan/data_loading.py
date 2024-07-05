
from math import log2
from torch.utils.data import Dataset
import torch.nn.functional as F

from syndatagenerators.data_preparation.datasets import LondonDataset


class ProgressiveDataset(Dataset):

    def __init__(self, assets: list, data_dir: str = '../GAN/data/Small_LCL_Data',
                 output_dir='./train_data', window_length: int = 8, target_len: int = 128,
                 overlap: int = 64):
        super().__init__()
        self.window_length = window_length
        self.assets = assets
        self.target_len = target_len
        self.overlap = overlap
        dataset = LondonDataset(assets, window_length=target_len, overlap=overlap, data_dir=data_dir,
                                train_data_dir=output_dir, labels=False)
        self.window_length = window_length
        self.data = dataset.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        series = self.data[item]
        target_length = series.size(1)
        assert (log2(target_length) % 1 == 0), "samples in train data need to be a power of 2"
        assert (log2(self.window_length) % 1 == 0), "window length needs to be a power of 2"
        assert self.window_length <= target_length, "window can not be greater then the length of the samples"

        reduce_factor = int(log2(target_length)) - int(log2(self.window_length)) # determines factor for downsizing
        reduced_sample = F.avg_pool1d(series, kernel_size=2**reduce_factor)
        return reduced_sample.view(1, -1)

