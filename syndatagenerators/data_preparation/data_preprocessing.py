import configparser

import torch
import pandas as pd
import numpy as np
import os
import datetime as dt

from syndatagenerators.data_preparation import TimeSeriesSplitter, min_max_scale_data, get_ts_loandonsmartmeter
from syndatagenerators.data_preparation.database import LondonSmartMeter


def load_train_data_lsm(assets: list = ['MAC000002', 'MAC000003'], start_date: str = '2011-11-30T01:00:00Z',
                        end_date: str = '2014-11-30T24:00:00Z', with_ids: bool = False, window_len: int = 32,
                        overlap: int = 0, freq: int = 30, save_dir: str = './train_data', check_exist: bool = True):
    """
    loads the preprocessed training data from the London Smart Meter data.
    Args:
        assets: list of household id's
        start_date: starting time point of the data
        end_date: end time point of the data
        with_ids: whether to load ID's of the data as a label
        window_len: hourly length of the extracted time series.
        overlap: overlap between the extracted time series.
        freq: frequency of the time series splitter (in minutes).
        save_dir: directory where train data is saved.
        check_exist: whether to check if train file already exists
    Returns:
        train_data: torch.tensor of preprocessed train data
    """
    if save_dir != None:
        exists, file_name = _check_exists(save_dir, assets, window_len, overlap, with_ids, check_exist)
    else:
        exists = False
    if not exists:
        df = _get_data_lsm(assets=assets, time_resolution=f'{freq}min', start_date=start_date, end_date=end_date,
                            drop_na=False, min_max_scale=True)
        train_data = split_and_save_train_data(df, seq_len=window_len, overlap=overlap, freq=freq, id_hs=with_ids,
                                            save_dir=save_dir, check_exist=check_exist)

    else:
        data_dir = os.path.join(save_dir, file_name)
        train_data = torch.load(data_dir)

    return train_data


def _get_data_lsm(config_file: str = 'last_db.ini', assets: list = ['MAC000002', 'MAC000003'],
                  time_resolution='30min',
                  start_date='2011-11-30T01:00:00Z', end_date='2014-11-30T24:00:00Z', drop_na: bool = False,
                  col_ids: bool = True, min_max_scale: bool = True):
    """
    Function for loading the London Smart Meter data for teh given assets and time period as a dataframe.
    Args:
        config_file: name of configuration file where use and password are saved
        assets: list of household id's
        time_resolution: frequency of the time series
        start_date: starting time point of the data
        end_date: end time point of the data
        drop_na: whether to drop Nan values
        col_ids: whether to use household id's as columns
        min_max_scale: whether to scale the data to the range (0,1).
    Returns:
        df: pd.Dataframe

    """
    HERE = os.path.dirname(__file__)
    config = configparser.ConfigParser()
    config.read(os.path.join(HERE, config_file))
    pgsql_info = dict(config["last"])

    db = LondonSmartMeter(user=pgsql_info['user'], password=pgsql_info['password'])
    db.connect()
    engine = db.engine

    df = get_ts_loandonsmartmeter(sensor_id=assets, time_resolution=time_resolution, ts_from=start_date,
                                  ts_to=end_date, engine=engine)
    if col_ids:
        df = df.pivot(index='time', columns='LCLid', values='kwh_hh')
        df.index = df.index.astype('datetime64[ns]')
    if drop_na:
        df = df.dropna()
    if min_max_scale:
        df[df.columns] = min_max_scale_data(df[df.columns], ivl=(0, 1))

    return df


def split_and_save_train_data(df: pd.DataFrame, seq_len: int = 48, overlap: int = 0, freq: int = 30, id_hs: bool = True,
                              save_dir: str = './train_data', check_exist: bool = True):
    """
    Splits a given dataframe in chunks of time series of a specific length and saves the resulting data as a pkl file.
    Args:
        df: dataframe of with the household id's as columns.
        seq_len: length (in hours) of the individual time series.
        overlap: overlap between the extracted time series.
        freq: frequency of the time series splitter (in minutes).
        id_hs: whether to save the id of the respective household in the okl file.
        save_dir: directory where train data is saved.
        check_exist: whether to check if train file already exists or redo splitting anyway.
        save_data: whether to save to loaded data sets. additionally does not load data from disk is present. 
    """
    assets = df.columns
    if save_dir != None:
        exists, file_name = _check_exists(save_dir, assets, seq_len, overlap, id_hs, check_exist)
        data_dir = os.path.join(save_dir, file_name)
    else:
        exists = False
    if not exists:
        list_samples = []
        if id_hs:
            list_labels = []
        splitter = TimeSeriesSplitter(window_length=seq_len, overlap=overlap, ts_freq=freq)
        for i, id_h in enumerate(df.columns):
            df_id = df[id_h].dropna()
            for ts in splitter.split_time_series(df_id):
                sample = torch.from_numpy(ts.T).view(1, 1, -1)
                list_samples.append(sample)
                if id_hs:
                    list_labels.append(torch.tensor([i], dtype=torch.int64))

        train_samples = torch.cat(list_samples)

        if id_hs:
            train_labels = torch.cat(list_labels).view(-1, 1)
            train_data = [train_samples, train_labels]
            if save_dir != None:
                torch.save(train_data.copy(), data_dir)
                print("Train data saved")
        else:
            train_data = train_samples
            if save_dir != None:
                torch.save(train_data.clone(), data_dir)
                print("Train data saved")
    if save_dir != None:
        train_data = torch.load(data_dir)

    return train_data


def _check_exists(data_dir: str, assets: list, seq_len: int, overlap: int, labels: bool = True,
                  check_exists: bool = True):
    """
    Helper function checking if the training data file already exists.
    Args:
        data_dir: directory where training data si saved
        assets: list of assets/ households
        seq_len: length of the time series
        overlap: overlap between time series
        labels: whether to use labels or not
    Returns:
        exists: boolean whether file already exists
        file_name: filename of training data
    """
    os.makedirs(data_dir, exist_ok=True)
    if labels:
        file_name = f'{len(assets)}_train_data_London_{seq_len}_{overlap}_labeled.pkl'
    else:
        file_name = f'{len(assets)}_train_data_London_{seq_len}_{overlap}.pkl'

    data_dir = os.path.join(data_dir, file_name)
    exists = os.path.exists(data_dir)
    if not check_exists:
        exists = False

    return exists, file_name


def load_preprocessed_data(data_dir: str = './data/Small_LCL_Data/', output_dir: str = './train_data/',
                           assets=['MAC000008', 'MAC000002'], labels: bool = True, window: int = 24, overlap: int = 23,
                           freq: int = 30):
    """
    Loads and preprocesses data from the London Smart Meter Data, saves it as training data in a .pkl file.
    Args:
        data_dir: directory where files can be loaded from
        output_dir: directory where output file is saved in
        assets: list of assets (households) to be used
        labels: decides whether label (int, specifying household) is saved for each sample
        window: length of each series (in hours)
        overlap: overlap used for extracting windows
        freq: frequency of loaded time series (in min)
    """
    os.makedirs(output_dir, exist_ok=True)
    if labels:
        file = f'{len(assets)}_train_data_London_{window}_{overlap}_labeled.pkl'
    else:
        file = f'{len(assets)}_train_data_London_{window}_{overlap}.pkl'
    directory = os.fsencode(data_dir)
    data_path = os.path.join(output_dir, file)

    if not os.path.exists(data_path):  # check if file with train data already exists, if not, preprocess
        splitter = TimeSeriesSplitter(window_length=window, overlap=overlap, ts_freq=freq)
        list_samples = []
        if labels:
            list_labels = []
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            print(f'Process file {filename}')
            df = pd.read_csv(os.path.join(data_dir, filename))
            df = df.replace('Null', np.nan)
            df['KWH/hh (per half hour) '] = df['KWH/hh (per half hour) '].astype(float)
            df = df.drop_duplicates()
            df = df.pivot(index='DateTime', columns='LCLid',
                          values='KWH/hh (per half hour) ')  # set ids as columns
            df.index = df.index.astype('datetime64[ns]')

            for i, asset in enumerate(assets):
                if asset in df.columns:  # check if asset names exist in columns of dataframe
                    df_asset = df[asset].dropna()
                    for ts in splitter.split_time_series(df_asset):
                        sample = torch.from_numpy(ts.T).view(1, 1, -1)
                        sample = min_max_scale_tensor(sample)
                        list_samples.append(sample)
                        if labels:
                            list_labels.append(torch.tensor([i], dtype=torch.int64))
        train_samples = torch.cat(list_samples)
        if labels:
            train_labels = torch.cat(list_labels).view(-1, 1)
            train_data = [train_samples, train_labels]
            torch.save(train_data.copy(), data_path)
        else:
            train_data = train_samples
            torch.save(train_data.clone(), data_path)
        print("Train data saved")
    train_data = torch.load(data_path)
    return train_data


def min_max_scale_tensor(data, ivl=(0, 1)):
    """
    Scales each feature to the given range using min max scaling.
    Args:
        data: data of shape [num_samples, num_features, seq_len]
        ivl: scaling interval. Default is [0,1].
    Returns:
        data
    """
    num_features = data.shape[1]
    for i in range(num_features):
        x_feature = data[:, i, :]
        x_max = x_feature.max()
        x_min = x_feature.min()
        x_scaled = ((x_feature - x_min) / (x_max - x_min)) * (ivl[1] - ivl[0]) + ivl[0]
        data[:, i, :] = x_scaled

    return data

def standard_conf():
    HERE = os.path.dirname(__file__)
    config = configparser.ConfigParser()
    config.read(os.path.join(HERE, 'last_db.ini'))
    return dict(config["last"])

def sliding_window(df, base="first", window_length = dt.timedelta(days=1), window_offset = dt.timedelta(days=1)):
    """
    Sliding window genrator function over a DataFrame with DatetimeIndex.
    Args:
        df: A DataFrame with a DatetimeIndex.
        base: Where to start slicing the frame into windows. If set to "first", windows are started at. If set to "midnight", windows start at midnight on the first full day.
        window_length: timedelta indicating the window length.
        window_offset: timedelta indicating the offset between two consecutive windows.
    """
    firstday = df.index.min()
    if base == "first":
        base = firstday
    elif base == "midnight":
        base = firstday
        if firstday.hour + firstday.minute + firstday.second != 0:
            try:
                base = dt.datetime(firstday.year, firstday.month, firstday.day, 0, 0, 0, tzinfo=firstday.tzinfo) + dt.timedelta(days=1)
            except Exception:
                print(firstday)
                print(df)

    while base < df.index.max():
        dfslice = df[base:base+window_length-dt.timedelta(milliseconds=1)]
        base += window_offset
        yield dfslice

def slice_windows(df, series_length, window_generator, with_labels=False):
    from torch.nn.functional import one_hot
    list_data = []
    list_labels = []
    
    # Iterate over sliding windows 
    for dfslice in window_generator:
        for series_id in dfslice.columns:
            series = dfslice[series_id]
            
            # Ignore slices where any value is nan, None or similar.
            # Ignore slices with too few data (e.g., when time changing to summer time)
            if series.isnull().any() or len(series) < series_length:
                continue
            
            # Append to data and label list
            list_data.append(series)
            if with_labels:
                list_labels.append(series_id)
    
    # Convert data to tensor 
    list_data = torch.cat([torch.tensor(series).view(1,1,-1) for series in list_data])
    
    # Convert labels to tensor based on one-hot-encoding
    if with_labels:
        idx_to_label = dict(enumerate(df.columns))
        label_to_idx = {v:k for k,v in idx_to_label.items()}
        list_labels = [label_to_idx[x] for x in list_labels if x in label_to_idx]
        list_labels = one_hot(torch.tensor(list_labels))
        
        # Also return map to convert back from indices to labels. 
        # Use np.argmax(list_labels) to convert from one hot encoding to indices.
        return [list_data, list_labels]#, idx_to_label
    else:
        return list_data

if __name__ == '__main__':
    ASSETS = ['MAC' + str(i).zfill(6) for i in range(2, 10)]
    print(ASSETS)
    data = _get_data_lsm(assets=ASSETS, drop_na=False)
    print(data.shape)
    train_data = split_and_save_train_data(data)
