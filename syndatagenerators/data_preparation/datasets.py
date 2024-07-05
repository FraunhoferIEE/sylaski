from typing import Optional, Union
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from syndatagenerators.data_preparation import load_train_data_lsm, TimeSeriesSplitter, min_max_scale_data
from syndatagenerators.data_preparation.database import DBHandler
from syndatagenerators.data_preparation.data_preprocessing import slice_windows, sliding_window
import datetime as dt
import pandas as pd
from os import path


class PowerDataModule(pl.LightningDataModule):
    def __init__(self, dataset="wpuq", assets: list = [], window_length: int = 24, overlap: int = 0, ts_freq=15,
                 save_file: str = None, overwrite = False, labeled: bool = False, start_date: str = '2019-01-01T01:00:00Z',
                 end_date: str = '2019-01-31T01:00:00Z', split_date=None, batch_size: int = 64,
                 num_workers: int = 8, drop_na=False, min_max_scale=True, load_metadata=False, split_base = "midnight"):
        """
        Args:

            dataset: the dataset to use. One of ["wpuq", "snh", "lsm", "openmeter"].
            assets: list of households to use for training
            window_length: window length (in hours) of the individual sequences
            overlap: overlap between the sequences
            ts_freq: frequency (to be used in the TimeSeriesSplitter)
            save_file: File where to save and reload data from (in .hdf format) or None to deactivate. If the file already exists, data will be loaded from the file instead of from the database, unless overwrite is set to True.
            overwrite: If set to true and save_file is not None, data will be loaded from the database and save_file will be overwritten.
            labels: whether to use labels (i.e. the household ID's)
            start_date: starting time point of the extracted data
            split_date: point at which to split the data into train and test set. Set to None to only create train + validation data.
            end_date: end time point of the extracted data
            batch_size: batch size to be used in the train dataloader
            num_workers: number of workers to be used in the train dataloader
            drop_na: whether to drop na values
            min_max_scale: whether to perform min-max-scaling
            load_metadata: whether to also load metadata.
            split_base: From where to start slicing series into windows. Either "first" (first series element) or "midnight" (Starting from midnight).
        """
        super().__init__()
        self.dataset = dataset
        self.data = None
        self.window_length = window_length
        self.overlap = overlap
        self.ts_freq = ts_freq
        self.assets = assets
        self.save_file = save_file
        self.overwrite = overwrite
        self.labeled = labeled
        self.start_date = start_date
        self.end_date = end_date
        self.split_date = split_date
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_train = None
        self.data_val = None
        self.drop_na = drop_na
        self.min_max_scale = min_max_scale
        self.load_metadata = load_metadata
        self.split_base = split_base

    def _load_from_db(self):
        # Load data and metadata from DB.
        self.db = DBHandler(self.dataset)
        df = self.db.get_data(sensor_id = self.assets, 
                            time_resolution = str(self.ts_freq)+"min",
                            ts_from = self.start_date,
                            ts_to = self.end_date)
        if self.load_metadata:
            self.metadata = self.db.get_metadata()
        return df
    
    def _load_from_file(self):
        # Load data and metadata from save_file.
        df = pd.read_hdf(self.save_file, key="data")
        if self.load_metadata:
            self.metadata = pd.read_hdf(self.save_file, key="metadata")
        return df
    
    def _save_to_file(self, df):
        # Save data and metadata to save_file.
        df.to_hdf(self.save_file, key="data")
        if self.load_metadata:
            self.metadata.to_hdf(self.save_file, key="metadata")
            
    def prepare_data(self):
        # Load data from the database or from a saved file.
        if self.save_file is None or not path.exists(self.save_file) or self.overwrite :
            df = self._load_from_db()
            if self.save_file is not None:
                self._save_to_file(df)
        else:
            df = self._load_from_file()
        
        # Preprocess and split dataset.
        self.data = self._preprocess(df)
        self._train_val_test_split()
        
    def _preprocess(self, df):
        powercol = 'w' if 'w' in df.columns else 'kwh_hh'
        df = df.drop_duplicates(subset=['time', 'id'])
        df = df.pivot(index='time', columns='id', values=powercol)
        
        if self.drop_na:
            df = df.dropna(axis="columns", how="all")
        if self.min_max_scale:
            # Remember min- and max values for backtransformation before performing min-max-scaling.
            self.minmax = (df.min().min(), df.max().max())
            df = ((df - self.minmax[0])/(self.minmax[1]-self.minmax[0]))
        return df
    
    def _train_val_test_split(self):
        df = self.data
        
        # Split DataFrame into train and test based on given date.
        # TODO: datetime64 is not timezone aware anymore.
        # Hence, split date will be converted to UTC+00, which results in unexpected behaviour.
        if self.split_date is None:
            self.split_date = np.datetime64(self.split_date)
            self.train_df = df
            self.test_df = None
        else:
            self.train_df = df[df.index <= self.split_date]
            self.test_df  = df[df.index >= self.split_date]
        
        # Split train and test set based on sliding window.
        series_length = (self.window_length*60)//self.ts_freq
        window_length = dt.timedelta(hours=self.window_length)
        window_offset = dt.timedelta(hours=self.window_length-self.overlap)
        
        self.data_train = slice_windows(self.train_df, series_length=series_length, with_labels=self.labeled,
                                        window_generator=sliding_window(self.train_df,
                                                                        base=self.split_base,
                                                                        window_length=window_length,
                                                                        window_offset = window_offset))
        if self.test_df is not None:
            self.data_test = slice_windows(self.test_df, series_length=series_length, with_labels=self.labeled,
                                           window_generator=sliding_window(self.test_df, 
                                                                           base=self.split_base,
                                                                           window_length=window_length,
                                                                           window_offset=window_offset))
        
        # Split train into train and validation set.
        # TODO: This should be made reproducible.
        # See https://stackoverflow.com/questions/55820303/fixing-the-seed-for-torch-random-split
        length_train = int(len(self.data_train) * 0.9)
        length_val = len(self.data_train) - length_train
        self.data_train, self.data_val = random_split(self.data_train, [length_train, length_val])
    
    def back_transform(self, data):
        """
        Transforms some given data back to the original scaling, as determined in the _preprocess function through min-max-scaling.
        Only applicable when min-max-scaling has been performed through this DataLoader.
        Args:
            data: some real valued data which supports basic arithmetic operations, e.g., given as numpy.array, torch.tensor or pandas.DataFrame consisting of floats.
        """
        return data*(self.minmax[1]-self.minmax[0]) + self.minmax[0]

    def setup(self, stage: Optional[str] = None):
        self.prepare_data()

    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False, batch_size=self.batch_size)
    

class LondonDataset(Dataset):
    """
    Dataset for the London Smart Meter Data. It uses preprocessed training data containing samples of a
    specific length of one or several households.

    """

    def __init__(self, assets: list = ['MAC000008'], window_length: int = 24, overlap: int = 0, ts_freq=30,
                 train_data_dir: str = './train_data/', labels: bool = True, start_date: str = '2011-11-30T01:00:00Z',
                 end_date: str = '2014-11-30T24:00:00Z'):
        """
        Args:
            assets: list of households to use for training
            window_length: window length (in hours) of the individual sequences
            overlap: overlap between the sequences
            ts_freq: frequency (to be used in the TimeSeriesSplitter)
            train_data_dir: directory where train data is saved
            labels: whether to use labels (i.e. the household ID's)
            start_date: starting time point of the extracted data
            end_date: end time point of the extracted data
        """
        super().__init__()
        self.window = window_length
        self.overlap = overlap
        self.freq = ts_freq
        self.assets = assets
        self.train_data_dir = train_data_dir
        self.labeled = labels
        self.start_date = start_date
        self.end_date = end_date
        self.data = load_train_data_lsm(assets=assets, start_date=start_date, end_date=end_date, with_ids=labels,
                                        window_len=window_length, overlap=overlap, freq=ts_freq, save_dir=train_data_dir
                                        )

    def __len__(self):
        if self.labeled:
            [series, labels] = self.data
            return len(series)
        else:
            return len(self.data)

    def __getitem__(self, item):
        if self.labeled:
            [series, labels] = self.data
            sample = series[item].type(torch.float32)
            label = labels[item].type(torch.int)
            return sample, label
        else:
            return self.data[item].type(torch.float32)


class ContrastiveDataset(Dataset):
    """
    Dataset for training the embedding of the individual household ID's.
    """

    def __init__(self, samples: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            samples: torch.Tensor of the samples used for the dataset
            labels: labels to be used for the contrastive learning process
        """
        super().__init__()
        self.X = samples.type(torch.float32)
        self.task_id = labels.long().ravel()
        self.task_lists = []
        for tid in range(torch.max(self.task_id).item() + 1):
            self.task_lists.append(list(np.where(self.task_id.numpy() == tid)[0]))

    def __getitem__(self, index):

        tid = self.task_id[index]
        task_list = self.task_lists[tid]
        similar_id = task_list[torch.randint(0, len(task_list), size=(1,))]
        dissimilar_id = torch.randint(0, len(self.task_id), size=(1,)).item()
        if dissimilar_id == tid:
            dissimilar_id = (tid + 1) % torch.max(self.task_id)

        x = self.X[index]
        x_similar = self.X[similar_id]
        x_dissimilar = self.X[dissimilar_id]
        return x, x_similar, x_dissimilar, tid, self.task_id[dissimilar_id]

    def __len__(self):
        return len(self.task_id)


class ClassificationDataset(Dataset):
    """
    Dataset for testing the discriminative score in the evaluation of the model.
    Labels real data samples as '1' and fake samples as '0'.
    """

    def __init__(self, real_data: torch.Tensor, fake_data: torch.Tensor):
        """

         real_data: torch.Tensor of shape [n_samples, feature_dim, seq_len]
                    containing real samples
         fake_data: torch.Tensor of shape [n_samples, feature_dim, seq_len]
                    containing fake samples
        """
        super(ClassificationDataset, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.real_data = real_data.transpose(1, 2).to(torch.float32)  # transpose s.t. features are the last axis
        self.fake_data = fake_data.transpose(1, 2).to(torch.float32)

        labels_real = torch.ones(len(real_data), dtype=torch.int64).to(self.device)
        labels_fake = torch.zeros(len(fake_data), dtype=torch.int64).to(self.device)
        self.data = [torch.cat([self.real_data, self.fake_data]), torch.cat([labels_real, labels_fake])]

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        sample = self.data[0][idx].to(self.device)
        label = self.data[1][idx].to(self.device)
        return sample, label


class LondonDataModule(pl.LightningDataModule):
    def __init__(self, assets: list = ['MAC000008'], window_length: int = 24, overlap: int = 0, ts_freq=30,
                 train_data_dir: str = './train_data/', labels: bool = True, start_date: str = '2011-11-30T01:00:00Z',
                 end_date: str = '2014-11-30T24:00:00Z', batch_size: int = 64, batch_size_val: int = 500,
                 num_workers: int = 8):
        """
        Args:

            assets: list of households to use for training
            window_length: window length (in hours) of the individual sequences
            overlap: overlap between the sequences
            ts_freq: frequency (to be used in the TimeSeriesSplitter)
            train_data_dir: directory where train data is saved
            labels: whether to use labels (i.e. the household ID's)
            start_date: starting time point of the extracted data
            end_date: end time point of the extracted data
            batch_size: batch size to be used in the train dataloader
            batch_size_val: batch size to be used in the validation dataloader
            num_workers: number of workers to be used in the train dataloader
        """
        super().__init__()
        self.data = None
        self.window = window_length
        self.overlap = overlap
        self.freq = ts_freq
        self.assets = assets
        self.train_data_dir = train_data_dir
        self.labeled = labels
        self.start_date = start_date
        self.end_date = end_date

        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers
        self.data_train = None
        self.data_val = None

    def prepare_data(self):
        self.data = load_train_data_lsm(assets=self.assets, start_date=self.start_date, end_date=self.end_date,
                                        with_ids=self.labeled, window_len=self.window, overlap=self.overlap,
                                        freq=self.freq, save_dir=self.train_data_dir
                                        )

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.prepare_data()
            length_train = int(len(self.data) * 0.9)
            length_val = len(self.data) - length_train
            self.data_train, self.data_val = random_split(self.data, [length_train, length_val])

    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.batch_size_val)


class SineDataset(Dataset):
    """
    Toy data set of noisy sine waves suitable for basic testing of time series synthesis.
    """
    def __init__(self, dim: int=128, noise_level: float=0.1, size: int=5000):
        """
        Initialize the dataset.
        Args:
            dim: The dimension of each time series to create.
            noise_level: How strong Gaussian noise is added to the sine wave.
            size: Number samples to create.
        """
        super().__init__()
        self.data = np.stack([self._noisy_sinewave(dim, noise_level) for _ in range(size)])
        self.data.shape = (size, 1, dim)

    def _noisy_sinewave(self, dim: int=128, noise_level=0.1):
        """
        Creates a single noisy sine wave.
        dim: The dimension of the sine wave.
        """
        return np.sin(np.linspace(0, np.pi*2, dim)) + noise_level*np.random.standard_normal(dim)

    def __getitem__(self, index):
        return torch.tensor(self.data[index]).type(torch.float32)

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    ASSETS = ['MAC' + str(i).zfill(6) for i in range(2, 100)]

    dataset = LondonDataset(assets=ASSETS, window_length=32)
    print(len(dataset))

class TSToTSDataset(Dataset):
    '''
    This DS represents two time series that can be transformed into one another e.g. with cycleGANs.
    Use two separate .pt files as input, it will yield a dict consisting of samples from both DS: {A: A, B: B}
    '''
    def __init__(self, file_A: Union[str, torch.tensor], file_B: Union[str, torch.tensor], transform=None):
        """
        Args:
            file_A (string): Path to the file of the first ts set.
            file_B (string): Path to the file of the second ts set.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if type(file_A) == str: 
            self.A = torch.load(file_A)
        else:
            self.A = file_A
        if type(file_B) == str:
            self.B = torch.load(file_B)
        else:
            self.B = file_B

        # sanity check. maybe this is not necessary?
        assert len(self.A) == len(self.B)

        self.transform = transform

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_A = self.A[idx]
        sample_B = self.B[idx]
        if self.transform != None:
            sample_A = self.transform(sample_A)
            sample_B = self.transform(sample_B)

        sample = {'A': sample_A, 'B': sample_B}

        return sample

class TStoTSDatamodule(pl.LightningDataModule):
    '''
    PyTorch Lightning Data Module modeling a DS of two time series transformable into another e.g. for cycleGANs.
    '''
    def __init__(self, fileA: Union[str, torch.tensor], fileB: Union[str, torch.tensor], batch_size: int, num_workers: int) -> None:
        """
        Args:
            file_A (string): Path to the file of the first ts set.
            file_B (string): Path to the file of the second ts set.
            batchsize (integer): Size of the batch that should be generated.
            num_workers (integer): Number of workers generated for loading the data. Can 
            be as large as number of available threads.
        """
        super(TStoTSDatamodule, self).__init__()
        self.fileA = fileA
        self.fileB = fileB
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.channels = 1

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = TSToTSDataset(self.fileA, self.fileB)

        test_split = int(0.25 * len(dataset))
        train_split = len(dataset) - test_split
        self.train, self.test = random_split(dataset, [train_split, test_split])

    def _base_dl(self, ds, shuffle=True):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def train_dataloader(self) -> DataLoader:
        return self._base_dl(self.train)

    def val_dataloader(self) -> DataLoader:
        return self._base_dl(self.test, False)

