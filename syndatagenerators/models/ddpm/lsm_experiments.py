import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import DataLoader
from DiffusionUtilities import DiffusionUtilities
from UNetPredictor2 import UNet_conditional
from syndatagenerators.models.ddpm.utils.Trainer import Trainer
from Generator import Generator
import time
from syndatagenerators.models.ddpm.utils import createDirs
from syndatagenerators.models.ddpm.experiments import evalModel_cond_multivar,analysisGen_multivar
from syndatagenerators.data_preparation.lsm_data import LSMDataset

def testCycle(iteration):
    # initialize seed-----------------------------------------------
    SEED = 17
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # --------------------------------------------------------------

    # hyperparam init-----------------------------------------------
    batch_size = 32
    n_epochs = 25
    lr = 1e-3
    n_steps, betaMin, betaMax = 200, 10**-6, 0.02
    gen_size= 48
    train_size = 48
    TIME = time.strftime('%Y-%m-%d-%H%M', time.localtime())
    NAME = f'{n_steps}-{batch_size}-{betaMin}-{betaMax}-{n_epochs}'
    TAG = f'{TIME}-{NAME}'


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    key = f"lsm_{iteration}"
    createDirs(key,TAG)
    store_path = f'../../../experiments/{key}/{TAG}/'

    # --------------------------------------------------------------

    number_ids = 100
    file_ids = np.arange(number_ids*iteration,(number_ids*(iteration+1)),1)
    cat_columns = ['LCLid']
    cont_columns = ['Summer_Time', 'Year_Sin', 'Year_Cos', 'Week_Sin', 'Week_Cos']
    target_columns = ['KWH_per_half_hour']

    with open(f'{store_path}conditions.txt', 'w') as fp:
        for item in cat_columns:
            # write each item on a new line
            fp.write("%s\n" % item)
        for item in cont_columns:
            # write each item on a new line
            fp.write("%s\n" % item)

    train_dataset = LSMDataset(store_path="../../../data/londonSmartMeter.h5",
                            file_ids=file_ids,
                            cat_columns=cat_columns,
                            cont_columns=cont_columns,
                            target_columns=target_columns,
                            load_df=False,
                            seq_shift=train_size)

    test_dataset = LSMDataset(store_path="../../../data/londonSmartMeter.h5",
                            file_ids=file_ids,
                            cat_columns=cat_columns,
                            cont_columns=cont_columns,
                            target_columns=target_columns,
                            load_df=False,
                            seq_shift=gen_size)

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size, shuffle=True,drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=5000, shuffle=True,drop_last=True)
    sample_cat,sample_cont,sample = next(iter(test_loader))


    network = UNet_conditional(n_steps= n_steps,c_in=1,c_out=1,num_classes=[number_ids],num_cont = len(cont_columns),device=device)

    ddpm = DiffusionUtilities(network=network.to(device),
                            train_size=train_size,
                            gen_size=gen_size,
                            betaMin=betaMin,
                            betaMax=betaMax,
                            n_steps=n_steps,
                            device=device)
    trainer = Trainer(ddpm=ddpm,
                    loader=train_loader,
                    batch_size=batch_size,
                    train_size=train_size,
                    n_epochs=n_epochs,
                    lr=lr,
                    store_path=store_path,
                    device=device,
                    tag=TAG,
                    household=key)

    generator = Generator(model=ddpm,
                        train_dataset=train_dataset,
                        n_steps=n_steps,
                        device=device,
                        tag=TAG)

    trainer.trainLSM_multivar()
    evalModel_cond_multivar(sample,network,generator,train_size,gen_size,store_path,device,sample_cat.to(device),sample_cont.to(device))

for i in range(55):
    print(f"[CURRENT TRAINING CYCLE]: {i}")
    testCycle(i)