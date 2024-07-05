import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Union, Optional
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


def lcl_to_hdf5(root=r'./Small LCL Data', file_out='londonSmartMeter.h5', overwrite=True):
    """
    Converts the LCL  data set to hdf5 format.
    Args:
        root: Path to the LCL data set.
        file_out: Output path of the HDF5-file to create.
        overwrite: If set to True and file_out already exists, it will be overwritten.
    """
    import pandas as pd
    import os

    if os.path.isfile(file_out):
        if overwrite:
            print(f'HDF5 file {root} exists and will be overwritten')
        else:
            print(f'HDF5 file {root} already exists. Delete it or set overwrite=True if you want to recreate it.')
            return

    hdf = pd.HDFStore(file_out, mode='a')

    for file_nr in range(168):
        csv_file_path = root + os.sep + 'LCL-June2015v2_' + str(file_nr) + '.csv'
        df = pd.read_csv(csv_file_path, delimiter=',', decimal='.',
                         na_values='Null', parse_dates=['DateTime'])
        # df.drop_duplicates(inplace=True)  #entfallen spaeter sowieso durch resampling
        df.sort_values("DateTime", inplace=True)
        df.set_index('DateTime', inplace=True)
        df.columns = ['KWH_per_hald_hour' if s == 'KWH/hh (per half hour) ' else s for s in df.columns]
        # list of unique household IDs
        LCLid_list = list(df['LCLid'].unique())
        print(f'file_nr = {file_nr}, LCLid_list[0] = {LCLid_list[0]}, LCLid_list[-1] = {LCLid_list[-1]}')
        for LCLid in LCLid_list:
            df_sgl = df[df['LCLid'] == LCLid]
            hdf.put(key=LCLid, value=df_sgl, format='table', append=True, data_columns=True)

    hdf.close()


def filter_households(data: dict, min_test_required: float = .75) -> dict():
    """
    filter_households(data, min_test_required) removes all households from data,
    where not enough test-samples are provided. E.g.
    data_filtered = filter_households(data, 0.5)
    would require, that 50% of all test time steps are available.
    """
    n_samples_test = []
    for household in data:
        n_samples_test.append(len(household['power_test']))
    max_num_time_steps = max(n_samples_test)
    min_abs_required = max_num_time_steps * min_test_required
    data_filtered = []
    for k, household in enumerate(data):
        if n_samples_test[k] > min_abs_required:
            data_filtered.append(household)
    return data_filtered


def load_households_from_hdf(hdf_file, start_mac, n_households, min_test_required, test_start_date):
    """
    Loads household LSM data from a prepared hdf5 file. Splits the data into train and test set.

    Args:
        hdf_file: path to the hdf5 file.
        start_mac: first household mac address to be considered.
        n_households: number of households to use.
        min_test_required: minimum required ratio of available test time steps.

    Returns:
        Dictionary containing train and test data as well as corresponding timestamps.
    """
    # first MAC number not to be considered
    end_mac = start_mac + n_households
    keys = [f'/MAC{k:06}' for k in range(start_mac, end_mac)]
    data = []
    with pd.HDFStore(hdf_file, mode='r') as hdf:
        for key_id, key in tqdm(enumerate(keys)):

            if key in hdf:
                df = hdf.get(key)

                # Remove columns to allow for resampling
                # experimental, might need the stdorToU attribute at a later point
                df = df.drop(columns=["LCLid", "stdorToU"])

                df = df.resample('30min').mean()
                df.dropna(inplace=True)
                # from pandas to numpy floats
                power_values = df['KWH_per_hald_hour'].values.astype(np.float32)
                time = df.index.values  # np.datetime64
                train_bool = time < test_start_date
                train_idx = np.where(train_bool)[0]
                test_idx = np.where(np.logical_not(train_bool))[0]

                power_train = power_values[train_idx]
                power_test = power_values[test_idx]
                time_train = time[train_idx]
                time_test = time[test_idx]
                data.append(dict(
                    key=key,
                    power_train=power_train,
                    power_test=power_test,
                    time_train=time_train,
                    time_test=time_test
                ))

    hdf.close()
    data = filter_households(data, min_test_required)
    return data


def timeseries_to_day_matrix(time, power, return_time_mat=False, drop_na=True):
    """
    Reshapes time series into a matrix, where each column corresponds to
    time of day and each row correspnds to a day
    X_train = timeseries_to_day_matrix(time_train, power_train)
    X_test = timeseries_to_day_matrix(time_test, power_test)
    """
    df = pd.DataFrame(dict(time=time, power=power))
    df.set_index('time', inplace=True)
    df = df.resample('30min').mean()
    min_date = np.min(df.index)
    bool_idx = df.index >= (np.datetime64(f'{min_date.year}-{min_date.month:02.0f}-{min_date.day:02.0f}') + 1)
    max_date = np.max(df.index)
    bool_idx = np.logical_and(bool_idx,
    (df.index < np.datetime64(f'{max_date.year}-{max_date.month:02.0f}-{max_date.day:02.0f}') - 1))
    df = df[bool_idx]
    power_mat = df['power'].values.reshape(-1,48)
    time_mat = df.index.values.reshape(-1,48)
    if drop_na:
        keep_idx = np.logical_not(np.any(np.isnan(power_mat), axis=1))
        power_mat = power_mat[keep_idx]
        time_mat = time_mat[keep_idx]

    if return_time_mat:
        return power_mat, time_mat
    else:
        return power_mat


class DataChallengeDataModule(pl.LightningDataModule):
    """
    """
    def __init__(self, hdf5data: Union[str, list], batch_size: int=64, sample_dim=None, household_id=None, num_workers=8):
        """
        Args:
            hdf5file: Either the path to the hdf5 file created through the lcl_to_hdf5(...) method
                      or the data itself as loaded through the load_households_from_hdf(...) method.
            batch_size: The batch size to use.
            sample_dim: The length of the time series. If None, the original dimensions in hdf5data will be used.
                        Otherwise, data will be filled with zeros at the end of the time series. For this reason, 
                        must be greater than or equal to the original input dimension.
            household_id: Id of household to use. If None (default), data from all households will be used.
            num_workers: Number of workers to be used in data loaders.
        """
        super().__init__()
        self.batch_size = batch_size
        self.sample_dim = sample_dim
        self.num_workers = num_workers

        # TODO: as init params
        if isinstance(hdf5data, str):
            self.data = load_households_from_hdf(
              hdf_file=hdf5data,
              start_mac=2,
              n_households=120,
              min_test_required=0.75,
              test_start_date=np.datetime64("2013-01-01 00:00")
            )
        elif isinstance(hdf5data, list):
            self.data = hdf5data
        else:
            raise TypeError(f"hdf5data must be a string or list. Passed data is of type {type(hdf5data)}")
        
        self.household_id = household_id
        self.setup()

    def setup(self, stage: Optional[str] = None):
        train_data, test_data = [], []
        for item in self.data:
            # If the DataModule is configured for a single household, iterate through
            # the data until the corresponding household is found.
            # TODO: O(n^2) when data is prepared for all households. Force passing data per household to constructor instead.
            if self.household_id is not None and item["key"] != self.household_id:
                continue
            
            X_train = timeseries_to_day_matrix(item["time_train"], item["power_train"], return_time_mat=False, drop_na=True)
            X_test = timeseries_to_day_matrix(item["time_test"], item["power_test"], return_time_mat=False, drop_na=True)
            train_data.append(self.__pad_with_zeros(X_train))
            test_data.append(self.__pad_with_zeros(X_test))
            
            # TODO: Add (optional) labels, e.g., for conditional GANs.
            
            if self.household_id:
                break
        
        self.train_data = self.__prepare_tensor(np.concatenate(train_data))
        self.test_data = self.__prepare_tensor(np.concatenate(test_data))
    
    def __pad_with_zeros(self, data):
        if self.sample_dim:
            data_dim = data.shape[1]
            if self.sample_dim < data_dim:
                raise ValueError(f"sample_dim must not be lower than data dimension. sample_dim: {sample_dim}, data_dim: {data_dim}")
            return np.pad(data, ((0, 0), (0, self.sample_dim - data_dim)), constant_values=0)
        return data
        
    def __prepare_tensor(self, data):
        return torch.from_numpy(data.reshape(data.shape[0],1,-1))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)
    
    
if __name__ == "__main__":
    # Paths
    lsmpath = r'/share/data1/mjuergens/SyLasKI/data/Small_LCL_Data/'
    hdf5file = '/share/data1/bschaefermeier/datasets/londonSmartMeter.h5'

    # first MAC number to be considered
    start_mac = 2
    # number of households
    n_households = 120
    # start date of the test period
    test_start_date = np.datetime64("2013-01-01 00:00")
    # minimum required ratio of available test time steps
    min_test_required = 0.75

    # Create hdf5 file from LSM Data if it does not exist yet.
    lcl_to_hdf5(root=lsmpath, file_out=hdf5file, overwrite=False)

    # Load data from the hdf5 file and split into train/test set.
    data = load_households_from_hdf(
      hdf_file=hdf5file,
      start_mac=start_mac,
      n_households=n_households,
      min_test_required=min_test_required,
      test_start_date=test_start_date
    )
    print("\nNumber of households:", len(data))

    # Retrieve data for one household
    for item in data:
        household_id = item["key"]
        train_data = item["power_train"]
        time_train = item["time_train"]
        test_data = item["power_test"]
        time_test = item["time_test"]
        break

    # Prepare matrix with training data and timestamps for one household. Each row corresponds to one day.
    X, times = timeseries_to_day_matrix(time_train, train_data, return_time_mat=True, drop_na=True)
    print(X)
    print(times)
