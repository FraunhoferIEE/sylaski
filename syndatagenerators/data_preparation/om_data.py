import torch
import pandas as pd
import numpy as np
#import joblib

from tqdm import tqdm
from datetime import timedelta
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


def get_summer_start(ids):
    # returns the timestamp for the beginning of summertime of the given year
    ids = ids[ids.month == 3]
    ids = ids[ids.weekday == 6]
    ids = ids[ids.hour == 1]
    ids = ids[ids.minute == 0]

    return ids[-1]

def get_summer_end(ids):
    # returns the timestamp for the end of summertime of the given year
    ids = ids[ids.month == 10]
    ids = ids[ids.weekday == 6]
    ids = ids[ids.hour == 1]
    ids = ids[ids.minute == 0]

    return ids[-1]

def make_ref_df(year):
    # create a reference dataframe for seasonal features
    ids = pd.date_range(start=f"{year}-01-01 00:00:00", end=f"{year}-12-31 23:30:00", freq='0.25H')
    st_start = get_summer_start(ids)
    st_end = get_summer_end(ids)
    st = np.ones(len(ids))
    st[ids < st_start] = 0
    st[ids > st_end] = 0

    t = np.linspace(0, 2*np.pi, len(ids)+1)
    year_sin = np.sin(t[:-1])
    year_cos = np.cos(t[:-1])
    data = {
        'Summer_Time': st,
        'Year_Sin': year_sin,
        'Year_Cos': year_cos}
    
    return pd.DataFrame(data=data, index=ids)

def add_weekly_sin_cos(df):
    # add sine and cosine for weekly cycles
    t = np.linspace(0, 2*np.pi, 7+1)
    for day_id, t in enumerate(t):
        df.loc[df.index.weekday==day_id, 'Week_Sin'] = np.sin(t)
        df.loc[df.index.weekday==day_id, 'Week_Cos'] = np.cos(t)
        
    return df

def add_daily_sin_cos(df):
    # add sine and cosine for weekly cycles
    t = np.linspace(0, 2*np.pi, 24+1)
    for hour_id, t in enumerate(t):
        df.loc[df.index.hour==hour_id, 'Day_Sin'] = np.sin(t)
        df.loc[df.index.hour==hour_id, 'Day_Cos'] = np.cos(t)
        
    return df

def add_seasons(df: pd.DataFrame):
    df.loc[np.logical_and(df.index.month > 2, df.index.month < 6), 'Season'] = 0
    df.loc[np.logical_and(df.index.month > 5, df.index.month < 9), 'Season'] = 1
    df.loc[np.logical_and(df.index.month > 8, df.index.month < 12), 'Season'] = 2
    df.loc[np.logical_or(df.index.month < 3, df.index.month == 12), 'Season'] = 3

    return df

def scale_df(df:pd.DateOffset):
    scaler = MinMaxScaler()
    idx = pd.Index([df.columns[0]]).append(df.columns[2:-1])
    print(idx)
    scaled_data = scaler.fit_transform(df[idx])
    #joblib.dump(scaler, f'scaler/{scaler_name}')
    odf1 = df[df.columns[1:2]]
    odf2 = df[df.columns[9]]
    sdf = pd.DataFrame(data=scaled_data, index=df.index, columns=idx)
    ndf = pd.concat([odf1, sdf, odf2], axis=1)
    return ndf

def prep_df(df):
    # preprocessing of dataframe:
    # add summertime and seasonal sin&cos functions
    # drop NaNs
    # drop days with missing data
    # scale features to [0, 1]
    #df.rename(columns = {'w':'KWH_per_half_hour'}, inplace = True)
    years = [*set([year for year in df.index.year])]
    ref_df = pd.concat([make_ref_df(year) for year in years], axis=0)
    ref_df = add_weekly_sin_cos(ref_df)
    ref_df = add_daily_sin_cos(ref_df)
    ref_df = add_seasons(ref_df)
    df = df.join(ref_df)
    df = scale_df(df)
    df = df.dropna()

    return df

def get_df_from_hdf5(store_path, file_ids, all_files=False):
    # returns a list of dataframes according to the list of given ids
    with pd.HDFStore(store_path) as store:
        keys = store.keys()
        if all_files:
            file_ids = range(len(keys))
        df = pd.concat([prep_df(store.get(keys[f])) for f in tqdm(file_ids)])
    
    return df

def get_df_from_hdf5_with_keys(store_path, keys):
    # returns a list of dataframes according to the list of given ids
    with pd.HDFStore(store_path) as store:
        df = pd.concat([prep_df(store.get(key)) for key in keys])
    
    return df

def prep_sequences(df, seq_len):
    # select first timestamp at 00:00
    id_start = df.loc[(df.index.hour==0) & (df.index.minute==0)].index[0]
    df = df[id_start:]
    dfs = [df.loc[(df.index[0] + timedelta(minutes=i*15*seq_len)):(df.index[0] + timedelta(minutes=i*15*seq_len) + timedelta(minutes=15*(seq_len-1)))] for i in range((len(df)//seq_len))]

    df = pd.concat([df for df in dfs if len(df) == seq_len])
    
    return df

class OMDataset(Dataset):
    # torch Dataset for OpenMeter
    def __init__(self, df, cat_columns, cont_columns, target_columns, seq_len:int=96, seq_shift=96):
        self.seq_len = seq_len
        self.seq_shift = seq_shift
        self.df = pd.concat([prep_sequences(df, seq_len=self.seq_len) for df in [df.loc[df.sensor_id == mac] for mac in np.unique(df.sensor_id)]])
        self.len = len(self.df)
        self.encode_labels(cat_columns)
        cat = self.df[cat_columns].values
        cont = self.df[cont_columns].values
        target = self.df[target_columns].values
        cat = torch.tensor(cat).long()
        cont = torch.tensor(cont).float()
        target = torch.tensor(target).float()
        self.cat, self.cont, self.target = cat, cont, target
        
    def encode_labels(self, cat_columns):
        enc = LabelEncoder()
        for c in cat_columns:
            self.df[c] = enc.fit_transform(self.df[c])
        
    def __len__(self):
        n_samples = (self.len//self.seq_len)
        return n_samples

    def __getitem__(self, index:int):
        cat = self.cat[self.seq_shift*index:self.seq_shift*index+self.seq_len]
        cont = self.cont[self.seq_shift*index:self.seq_shift*index+self.seq_len]
        target = self.target[self.seq_shift*index:self.seq_shift*index+self.seq_len]
        return cat, cont, target
    
def get_one_batch(dl):
# returns a single batch from a torch dataloader
    for x_cat, x_cont, y in dl:
        return x_cat, x_cont, y