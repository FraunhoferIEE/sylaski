# %%
import pandas as pd
import os
from pathlib  import Path
import torch
import numpy as np
from torch.utils.data import Dataset
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

def create_LSM_datastore(hdf_file:str='londonSmartMeter.h5'):
    root = os.path.dirname(__file__)
    path = Path(root)
    # name of the hdf5 output file
    hdf = pd.HDFStore(str(path.parent.absolute()) + os.sep + hdf_file, mode='r')
    return hdf

def create_OM_datastore(hdf_file:str='openMeter.h5'):
    root = os.path.dirname(__file__)
    path = Path(root)
    # name of the hdf5 output file
    hdf = pd.HDFStore(str(path.parent.absolute()) + os.sep + hdf_file, mode='r')
    return hdf

def retrieve_keys(hdf:pd.HDFStore) -> list:
    # get list of available keys in the hdf file
    keys = hdf.keys()
    # print a subset of keys
    print('First 10 keys: ',keys[:10])
    #print(f'Number of keys {len(keys)}')
    return keys

def read_sgl_houshold_df(key:str, hdf:pd.HDFStore, dropna:bool=True) -> pd.DataFrame:
    df = hdf.get(key)
    if(dropna):
        df.dropna(inplace=True)
    return df

def read_multiple_households_df(keys:list[str], hdf:pd.HDFStore, dropna:bool=True) -> pd.DataFrame:
    df = hdf.get(keys[0])
    for i in tqdm(range(1, len(keys))):
        df = pd.concat([df, hdf.get(keys[i])])
    if(dropna):
        df.dropna(inplace=True)
    return df

def read_number_households_df(number_of_households:int,  hdf:pd.HDFStore) -> pd.DataFrame:
    return read_multiple_households_df(hdf.keys()[:number_of_households], hdf=hdf, dropna=False)

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

class SglHouseholdDataset(Dataset):
    def __init__(self, df:pd.HDFStore, seq_len:int=48):
        self.seq_len = seq_len
        # data loading
        x = df['KWH_per_hald_hour'].values.astype(np.float32)
        # normalize
        x = x/np.max(x)
        self.x = torch.from_numpy(x)
        # length is
        self.n_samples = (len(x)//self.seq_len)-1

    def __getitem__(self, index:int):
        start_idx = index*self.seq_len
        #torch.manual_seed(42)
        rand_offset = torch.randint(self.seq_len,(1,))
        relative_idx =  torch.arange(self.seq_len)
        idx = start_idx + relative_idx + rand_offset
        return self.x[idx]#.reshape(1,-1)

    def __len__(self):
        # len(dataset)
        return self.n_samples

class SglHouseholdSequenceDataset(Dataset):
    def __init__(self, df:pd.HDFStore, seq_len:int=48):
        self.seq_len = seq_len
        # data loading
        x = df['KWH_per_hald_hour'].values.astype(np.float32)
        # normalize
        x = x/np.max(x)
        self.x = torch.from_numpy(x)
        # length is
        self.n_samples = (len(x)//self.seq_len)-1

    def __getitem__(self, index:int):
        start_idx = index*self.seq_len
        rand_offset = torch.randint(self.seq_len,(1,))
        relative_idx =  torch.arange(self.seq_len)
        idx = start_idx + relative_idx + rand_offset
        return self.x[idx].reshape(1,-1).t()

    def __len__(self):
        # len(dataset)
        return self.n_samples
    
class MultiHouseholdDataset(Dataset):#always starts at 0:00, all sequences that contain nans are removed completly, that way the sequences do not contain jumps
    def __init__(self, df:pd.DataFrame, seq_len) -> None:
        super().__init__()
        self.seq_len = seq_len
        macs = np.unique(df.LCLid)
        self.samples = torch.cat([self.make_sequences(df.loc[df.LCLid == key]) for key in macs])
        self.samples = self.samples / torch.max(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
    
    def __len__(self):
        return self.samples.size(0)
    
    def make_sequences(self, df:pd.DataFrame):
        start_index = df.loc[(df.index.hour==0) & (df.index.minute==0)].index[0]
        df = df[start_index:]
        df = df[~df.index.duplicated(keep='first')]
        df = df.resample("30min").asfreq()
        df = df[:(len(df) // self.seq_len)*self.seq_len]
        if(not df.index.is_monotonic_increasing):
            print(f"Warning: index of LCLid {df.LCLid[0]} is not monotonic increasing!")
        x = df['KWH_per_hald_hour'].values.astype(np.float32)
        x = x.reshape((-1, self.seq_len))
        x = x[~np.isnan(x).any(axis=1)]
        return torch.from_numpy(x)
    
class OMCondDataset(Dataset):#always starts at 0:00, all sequences that contain nans are removed completly, that way the sequences do not contain jumps
    def __init__(self, df:pd.DataFrame, seq_len:int=96, cat_conditions:list[str]=["sensor_id", "city", "federal_state", "usage", "post_code"], cont_conditions:list[str]=["week", "year", "summertime", "area", "temperature"]) -> None:
        super().__init__()
        self.seq_len = seq_len
        le = LabelEncoder()
        for cat in cat_conditions:
            df[cat] = le.fit_transform(df[cat].values)

        self.lens_cat_conditions = []
        if "area" in cont_conditions:
            self.area_max = df["area"].max()
        if "temperature" in cont_conditions:
            self.temp_max = np.nanmax(df["temperature"].values)
            self.temp_min = -23.1 
            
        for cat in cat_conditions:
            self.lens_cat_conditions.append(len(np.unique(df[cat])))
        macs = np.unique(df["sensor_id"])
        clean_dfs = [self.cleanDF(df.loc[df["sensor_id"] == key]) for key in macs]
        self.samples = torch.cat([self.makeSequences(clean_df) for clean_df in clean_dfs])
        self.cont = torch.cat([self.makeContConditions(clean_df, cont_conditions) for clean_df in clean_dfs])
        self.num_cont = self.cont.shape[2]
        self.cat = torch.cat([self.makeCatConditions(clean_df, cat_conditions)for clean_df in clean_dfs])
        self.dropNan()
        if(self.samples.shape[0] == 0):
            raise ValueError("There are no sequences that don't contain atleast one nan! Meaning there are no sequences in this dataset")
        self.max = self.samples.max()
        self.mean = self.samples.mean()
        self.std = self.samples.std()
        self.samples = self.samples / self.max

    def __getitem__(self, index):
        return self.cat[index], self.cont[index], self.samples[index]
    
    def __len__(self):
        return self.samples.size(0)
    
    def cleanDF(self, df):
        start_index = df.loc[(df.index.hour==0) & (df.index.minute==0)].index[0]
        df = df[start_index:]
        df = df[~df.index.duplicated(keep='first')]
        df = df.resample("15min").asfreq()
        df = df[:(len(df) // self.seq_len)*self.seq_len]
        if(not df.index.is_monotonic_increasing):
            print(f"Warning: index of LCLid {df.LCLid[0]} is not monotonic increasing!")
        return df    

    def makeSequences(self, df:pd.DataFrame):
        #make samples
        df.loc[df['w']> 35000, 'w'] = np.nan
        x = df['w'].values.astype(np.float32)
        x = x.reshape((-1, self.seq_len))
        return torch.from_numpy(x)
    
    def makeContConditions(self, df:pd.DataFrame, conditions:list):
        cont = np.empty((len(df), 0))
        if("day" in conditions):
            cont = np.append(cont, getDayPeriodicEmbedding(df.index), axis=1)
        if("week" in conditions):
            cont = np.append(cont, getWeekPeriodicEmbedding(df.index), axis=1)
        if("year" in conditions):
            cont = np.append(cont, getYearPeriodicEmbedding(df.index), axis=1)
        #normalize cont up to here to be in [0,1]
        cont = (cont + 1) / 2


        if("summertime" in conditions):
            cont = np.append(cont, getSummertimes(df.index).reshape(-1,1), axis=1)

        if("area" in conditions):
            values = df["area"].values.reshape(-1,1)/self.area_max
            cont = np.append(cont, values, axis=1)
        if("temperature" in conditions):
            temps = df["temperature"].values
            negs = np.where(temps < self.temp_min)[0]
            not_neg = np.where(temps >= self.temp_min)[0]
            temps[negs] = np.interp(negs, not_neg, temps[not_neg])
            temps = (temps - self.temp_min) / (self.temp_max - self.temp_min)
            #temps = temps / self.temp_max
            if(np.count_nonzero(np.where(temps < self.temp_min)[0]) > 0):
                print(f"Something went wrong while interpolating temps of {df['sensor_id'][0]}")
            cont = np.append(cont, temps.reshape(-1,1), axis=1)
        if(cont.shape[1] > 0):
            cont = cont.reshape(-1, self.seq_len, cont.shape[1])
        else:
            cont = np.empty((len(df)//self.seq_len, self.seq_len, 0))
        cont = torch.from_numpy(cont).to(torch.float32)
        return cont
        
    def makeCatConditions(self, df:pd.DataFrame, conditions:list):
        cat = np.empty((len(df), 0))
        if("sensor_id" in conditions):
            cat = np.append(cat, df["sensor_id"].values.reshape(-1,1), axis=1)
        if("cluster_id" in conditions):
            cat = np.append(cat, df["cluster_id"].values.reshape(-1,1), axis=1)
        if("category" in conditions):
            cat = np.append(cat, df["category"].values.reshape(-1,1), axis=1)
        if("city" in conditions):
            cat = np.append(cat, df["city"].values.reshape(-1,1), axis=1)
        if("federal_state" in conditions):
            cat = np.append(cat, df["federal_state"].values.reshape(-1,1), axis=1)
        if("usage" in conditions):
            cat = np.append(cat, df["usage"].values.reshape(-1,1), axis=1)
        if("post_code" in conditions):
            cat = np.append(cat, df["post_code"].values.reshasype(-1,1), axis=1)
        if(cat.shape[1] > 0):
            cat = cat.reshape(-1, self.seq_len, cat.shape[1])
        else:
            cat = np.empty((len(df)//self.seq_len, self.seq_len, 0))

        cat = torch.from_numpy(cat).to(torch.int32)
        return cat
        

    def dropNan(self):
        not_nan_idx = ~(np.isnan(self.samples).to(torch.bool).any(axis=1))
        self.samples = self.samples[not_nan_idx]
        if(self.cont is not None):
            self.cont = self.cont[not_nan_idx]
        if(self.cat is not None):
            self.cat = self.cat[not_nan_idx]

    def retransform(self, samples):
        return samples * self.max
    
    

if __name__== '__main__':
    key_id=6
    hdf = create_LSM_datastore()
    keys = retrieve_keys(hdf)
    key = keys[key_id]
    # Retrieve data with a certain key
    start_time = time.time()
    df = read_sgl_houshold_df(key, hdf)
    print(f'Reading time current key: {(time.time()-start_time)*1000:.3f}ms')
    # plot data
    df[10000:(10000+48*10)].plot()
    plt.title(
        'LCLid: ' + keys[key_id][1:] +
        ' (' + df['stdorToU'][0] + ')')
    plt.show()
    print(f'Current key name: {keys[key_id]}')
    print(f'Number of samples (current key): {len(df)}')
    # %%
    df = read_multiple_households_df(['/MAC000002', '/MAC000003'], hdf)
    print(df)
    hdf.close()

