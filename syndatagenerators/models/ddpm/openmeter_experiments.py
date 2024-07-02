import sys
sys.path.append('/home/npopkov/syndatagenerators') # <---path to the repo


import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import time
from torch.utils.data import DataLoader
from syndatagenerators.models.ddpm.utils.DiffusionUtilities import DiffusionUtilities
from syndatagenerators.models.ddpm.architectures.UNetPredictor2 import UNet_conditional
from syndatagenerators.models.ddpm.utils.Trainer import Trainer
from syndatagenerators.models.ddpm.utils.Generator import Generator
from syndatagenerators.models.ddpm.utils.dataprep_utils import create_OM_datastore,read_multiple_households_df,OMCondDataset, getWeekPeriodicEmbedding,getYearPeriodicEmbedding
from syndatagenerators.models.ddpm.utils.utils import createDirs, saveConditionsTxt
from syndatagenerators.models.ddpm.utils.experiments import evalModel_cond_multivar, challengeGen

# initialize seed-----------------------------------------------
SEED = 17
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# --------------------------------------------------------------

# hyperparam init-----------------------------------------------
batch_size = 32
n_epochs = 40
lr = 3e-4
n_steps, betaMin, betaMax = 1000, 10**-6, 0.002
seq_len= 96
TIME = time.strftime('%Y-%m-%d-%H%M', time.localtime())
NAME = f'{n_steps}-{batch_size}-{betaMin}-{betaMax}-{n_epochs}'
TAG = f'{TIME}-{NAME}'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
key = "web_expert_tests"
current_path = f'/home/npopkov/syndatagenerators/syndatagenerators/models/ddpm/experiments/'
store_path = f'{current_path}{key}/{TAG}/'
createDirs(key,TAG,current_path)

# --------------------------------------------------------------
cats = ["cluster_id"] #["sensor_id", "city", "federal_state", "usage", "post_code"]
conts = ["week", "year"] #["week", "year", "summertime", "area"]

saveConditionsTxt(store_path,cats,conts)


hdf = create_OM_datastore(hdf_file='20240119_OM_privat-19-02.h5')
keys = hdf.keys()
number_ids = len(keys)

df = read_multiple_households_df(keys,hdf)
#df = df[df['area']<300.0]
hdf.close()

dataset = OMCondDataset(df=df,seq_len=seq_len,cat_conditions=cats,cont_conditions=conts)

# split data for train and test (80/20)
train_ds , test_ds = torch.utils.data.random_split(dataset,[0.8,0.2])

#tweek the size of the test batch here
test_batch_size = int(len(test_ds)/500)*500

train_loader = DataLoader(dataset=train_ds,
                        batch_size=batch_size, 
                        shuffle=True,
                        drop_last=True)

test_loader = DataLoader(dataset=test_ds, 
                         batch_size=test_batch_size, 
                         shuffle=False,
                         drop_last=True)

# pulling a sample from the test dataset for later use
# it consists of categorical and continuous conditions and the corresponding time series
sample_cat, sample_cont, sample = next(iter(test_loader))

# the neural network part of the ddpm that predicts the noise
network = UNet_conditional(n_steps= n_steps,
                           c_in=1,
                           c_out=1,
                           num_classes=dataset.lens_cat_conditions,
                           num_cont = dataset.num_cont,
                           device=device)

# the ddpm class that combines the network and many mathematical operations needed to perform denoising
ddpm = DiffusionUtilities(network=network.to(device),
                          size=seq_len,
                          betaMin=betaMin,
                          betaMax=betaMax,
                          n_steps=n_steps,
                          device=device)

# trainer util class that trains the ddpm
trainer = Trainer(ddpm=ddpm,
                  loader=train_loader,
                  batch_size=batch_size,
                  train_size=seq_len,
                  n_epochs=n_epochs,
                  lr=lr,
                  store_path=store_path,
                  device=device,
                  tag=TAG,
                  household=key)

# generator util class that generates samples from the ddpm, especially useful for evaluation
generator = Generator(model=ddpm,
                      train_dataset=train_ds,
                      n_steps=n_steps,
                      device=device,
                      tag=TAG)

trainer.train_ddpm()
#store_path = "/home/npopkov/syndatagenerators/syndatagenerators/models/ddpm/experiments/web_expert_tests/2024-06-24-2007-200-32-1e-06-0.02-25/"



challengeGen(train_size=seq_len,gen_size=seq_len, network=network,generator=generator,keys=keys,store_path=store_path,dataset=dataset)

evalModel_cond_multivar(sample=sample,
                        test_batch_size=test_batch_size,
                        network=network,
                        generator=generator,
                        device=device,
                        train_size=seq_len,
                        gen_size=seq_len,
                        store_path=store_path,
                        cats=sample_cat,
                        conts=sample_cont,)