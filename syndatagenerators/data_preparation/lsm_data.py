import os
import torch
import pandas as pd
import numpy as np

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
    ids = pd.date_range(start=f"{year}-01-01 00:00:00", end=f"{year}-12-31 23:30:00", freq='0.5H')
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
        # print(df.loc[df.index.weekday==day_id])
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

def scale_df(df):
    scaler = MinMaxScaler()
    scaler_name = df[df.columns[0]][0]
    scaled_data = scaler.fit_transform(df[df.columns[2:]])
    # joblib.dump(scaler, f'../../trained_models/scaler/{scaler_name}')
    odf = df[df.columns[:2]]
    sdf = pd.DataFrame(data=scaled_data, index=df.index, columns=df.columns[2:])
    ndf = pd.concat([odf, sdf], axis=1)
    return ndf

def prep_df(df):
    # preprocessing of dataframe:
    # add summertime and seasonal sin&cos functions
    # drop NaNs
    # drop days with missing data
    # scale features to [0, 1]
    df.rename(columns = {'KWH_per_hald_hour':'KWH_per_half_hour'}, inplace = True)
    years = [*set([year for year in df.index.year])]
    ref_df = pd.concat([make_ref_df(year) for year in years], axis=0)
    ref_df = add_weekly_sin_cos(ref_df)
    ref_df = add_daily_sin_cos(ref_df)
    df = df.join(ref_df)
    #df = scale_df(df)
    df = df.dropna()

    return df

def get_df_from_hdf5(store_path, file_ids, load_df=False):
    raw_data_dir = './raw_data/'
    file_name = f'df_{len(file_ids)}.csv'
    os.makedirs(raw_data_dir, exist_ok=True)
    if load_df:
        df = pd.read_csv(os.path.join(raw_data_dir, file_name), index_col=[0])
        df.index = pd.to_datetime(df.index)
    else:
        with pd.HDFStore(store_path) as store:
            keys = store.keys()
            df = pd.concat([prep_df(store.get(keys[f])) for f in tqdm(file_ids)])
            df.to_csv(os.path.join(raw_data_dir, file_name))
    return df

def prep_sequences(df, seq_len):
    # select first timestamp at 00:00
    id_start = df.loc[(df.index.hour==0) & (df.index.minute==0)].index[0]
    df = df[id_start:]
    dfs = [df.loc[(df.index[0] + timedelta(minutes=i*30*seq_len)):(df.index[0] + timedelta(minutes=i*30*seq_len) + timedelta(minutes=30*(seq_len-1)))] for i in range((len(df)//seq_len))]
    df = pd.concat([df for df in dfs if len(df) == seq_len])
    
    return df

class LSMDataset(Dataset):
    # torch Dataset for LondonSmartMeter
    def __init__(self, store_path, file_ids, cat_columns, cont_columns, target_columns, seq_len:int=48, seq_shift=48, load_df=False,):
        self.seq_len = seq_len
        self.seq_shift = seq_shift
        df = get_df_from_hdf5(store_path=store_path, file_ids=file_ids, load_df=load_df)
        self.df = pd.concat([prep_sequences(df, seq_len=self.seq_len) for df in [df.loc[df.LCLid == mac] for mac in np.unique(df.LCLid)]])
        self.len = len(self.df)
        self.encode_labels(cat_columns)
        cat = self.df[cat_columns].values
        cont = self.df[cont_columns].values
        target = self.df[target_columns].values
        cat = torch.tensor(cat).long()
        cont = torch.tensor(cont).float()
        target = torch.tensor(target).float()
        self.cat, self.cont, self.target = cat, cont, target
        self.samples = target
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
    
    #assumes utc
def getSummerStart(year):
    dt = pd.Timestamp(year=year, month=4, day=1) 
    dt += pd.Timedelta(days=6-dt.dayofweek)
    dt -= pd.Timedelta(days=7)
    dt += pd.Timedelta(hours=2)
    return dt

#assumes utc
def getSummerEnd(year):
    dt = pd.Timestamp(year=year, month=11, day=1) 
    dt += pd.Timedelta(days=6-dt.dayofweek)
    dt -= pd.Timedelta(days=7)
    dt += pd.Timedelta(hours=2)
    return dt

def getSummertimes(dates:pd.DatetimeIndex):
    summerStarts = {}
    summerEnds = {}
    years = np.unique(dates.year)
    for year in years:
        summerStarts[f"{year}"] = getSummerStart(year)
        summerEnds[f"{year}"] = getSummerEnd(year)
    summertime = np.zeros(len(dates))
    for idx, date in enumerate(dates):
        summertime[idx] = 1 if (date > summerStarts[f"{date.year}"] and date < summerEnds[f"{date.year}"]) else 0
    return summertime

def getDayPeriodicEmbedding(dates:pd.DatetimeIndex):
    day_sin = np.sin((dates.hour + dates.minute / 60) * (2. * np.pi/24))
    day_cos = np.cos((dates.hour + dates.minute / 60) * (2. * np.pi/24))
    return np.stack((day_sin, day_cos), axis=-1)

def getWeekPeriodicEmbedding(dates:pd.DatetimeIndex):
    week_sin = np.sin(dates.day_of_week * (2 * np.pi / 7))
    week_cos = np.cos(dates.day_of_week * (2 * np.pi / 7))
    return np.stack((week_sin, week_cos), axis=-1)

def getYearPeriodicEmbedding(dates:pd.DatetimeIndex):
    daysInYear = np.array([365 + (1 if date.is_leap_year else 0 ) for date in dates])
    year_sin = np.sin(dates.day_of_year * (2 * np.pi / daysInYear))
    year_cos = np.cos(dates.day_of_year * (2 * np.pi / daysInYear))
    return np.stack((year_sin, year_cos), axis=-1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = pd.date_range(start=f"{2023}-01-01 00:00:00", end=f"{2023}-12-31 23:30:00", freq='0.5H')
    day = getDayPeriodicEmbedding(x)
    week = getWeekPeriodicEmbedding(x)
    year = getYearPeriodicEmbedding(x)
    st = getSummertimes(x)

    plt.figure(figsize=(10,5))
    plt.plot(x, st, label="summertime")
    plt.xlim(x[0], x[-1])
    plt.legend()
    plt.show()
    print(np.arange(len(x)).shape)
    plt.figure(figsize=(10,5))
    plt.plot(x[:], year[:,1], label="year_cos")
    plt.plot(x[:], year[:,0], label="year_sin")
    plt.xlim(x[0], x[-1])
    plt.legend()
    plt.show()
    plt.figure(figsize=(10,5))
    plt.scatter(x[:48*7], week[:48*7,1], label="week_cos")
    plt.scatter(x[:48*7], week[:48*7,0], label="week_sin")
    plt.xlim(x[0], x[48*7-1])
    plt.legend()
    plt.show()
    plt.figure(figsize=(10,5))
    plt.plot(x[:48], day[:48,1], label="day_cos")
    plt.plot(x[:48], day[:48,0], label="day_sin")
    plt.xlim(x[0], x[48-1])
    plt.legend()
    plt.show()