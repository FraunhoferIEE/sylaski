import matplotlib.pyplot as plt
from syndatagenerators.models.ddpm.utils.utils import loadModel,writeData,writeData_multivar,checkTSNE,checkCovMat,buildKeyList
import torch
import numpy as np
import pandas as pd
from syndatagenerators.metrics.mmd_score import new_mmd
from syndatagenerators.metrics.discriminative_score import discriminative_score
from syndatagenerators.data_analysis.clustering import Clustering
from syndatagenerators.models.ddpm.utils.Generator import Generator

def evalModel(sample,network,embedder,generator,store_path,store_path_net,key,TAG):

    loadModel(model=network, path=store_path)
    loadModel(model=embedder.net, path=store_path_net)
    with torch.no_grad():
        writeData(500, key, generator=generator,times=10,TAG=TAG)

    genSet = np.loadtxt(f'../../../outputs/{key}/{TAG}.txt')
    genSet = torch.from_numpy(genSet).squeeze().view(-1, 1, 48)
    sample = sample.squeeze().view(-1, 1, 48)
    realTSN, fakeTSN = checkTSNE(sample, genSet, key,TAG)
    score = new_mmd(sample, genSet).squeeze().cpu().detach().numpy()
    print(f"mmd {score}")
    with open(f'../../../figures/{key}/{TAG}/scores.txt', "ab") as f:
        np.savetxt(f, [score])
    score = discriminative_score(sample, genSet)
    with open(f'../../../figures/{key}/{TAG}/scores.txt', "ab") as f:
        np.savetxt(f, [score])
    checkCovMat(sample, genSet, key,TAG)

def evalModel_cond(sample,network,generator,store_path,key,TAG,device,cats,conts):

    loadModel(model=network, path=store_path)
    with torch.no_grad():
        writeData(500, key, generator=generator,times=10,TAG=TAG)

    genSet = np.loadtxt(f'../../../outputs/{key}/{TAG}.txt')
    #genSet = np.loadtxt(f'../../../outputs/MultiHousehold/2023-07-18-1008-200-32-1e-06-0.02-50.txt')
    genSet = torch.from_numpy(genSet).squeeze().view(-1,1, 96).float()
    sample = sample.squeeze().view(-1, 1,96).float()
    realTSN, fakeTSN = checkTSNE(sample, genSet, key,TAG)
    checkCovMat(sample, genSet, key,TAG)
    score = new_mmd(sample.squeeze(), genSet.squeeze(),device)
    print(f"mmd {score}")
    with open(f'../../../figures/{key}/{TAG}/scores.txt', "ab") as f:
        np.savetxt(f, [score])
    score = discriminative_score(sample, genSet)
    with open(f'../../../figures/{key}/{TAG}/scores.txt', "ab") as f:
        np.savetxt(f, [score])

def evalModel_cond_multivar(sample,test_batch_size,network,generator,train_size,gen_size,store_path,device,cats,conts):

    loadModel(model=network, path=f'{store_path}network.pt')
    with torch.no_grad():
        writeData_multivar(500, generator,store_path,train_size,gen_size,times=int(test_batch_size/500),cats=cats.to(device),conts=conts.to(device))

        genSet = np.loadtxt(f'{store_path}5ksamples.txt')
        genSet = torch.from_numpy(genSet).squeeze().view(-1,1, train_size).float()
        sample = sample.squeeze().view(-1, 1,train_size).float()
        realTSN, fakeTSN = checkTSNE(sample, genSet, store_path)
        checkCovMat(sample, genSet, store_path,gen_size)
        score = new_mmd(sample.squeeze(), genSet.squeeze(),device)
        print(f"mmd {score}")
        with open(f'{store_path}scores.txt', "ab") as f:
            np.savetxt(f, [score])
        score = discriminative_score(sample, genSet)
        with open(f'{store_path}scores.txt', "ab") as f:
            np.savetxt(f, [score])

def clustering_metric(csvPath,name,household,TAG):
    ddpm = pd.read_csv(csvPath, index_col=[-1])
    Clustering.plot_cluster(ddpm)
    plt.savefig(f"../../../figures/{household}/{TAG}/clustering.png")
    plt.close()



def challengeEval(household,network,embedder,generator,store_path,store_path_net,TAG):
    # Create the numpy array with datetime64[m] dtype
    loadModel(model=network, path=store_path)
    loadModel(model=embedder.net, path=store_path_net)
    arr = np.arange('2013-01-01 00:00', '2014-03-01 00:00', dtype='datetime64[m]')
    keyList = buildKeyList()
    # Get the indices of the datetime values that correspond to half hour intervals
    half_hour_indices = np.where(arr.astype('datetime64[s]').astype('int64') % (30 * 60) == 0)[0]

    # Filter the array using the half-hourly indices
    filtered_arr = arr[half_hour_indices]

    # Create a new DataFrame from the filtered array
    df = pd.DataFrame(filtered_arr, columns=['Datetime'])

    with torch.no_grad():
        for i,key in enumerate(keyList):
            print(f"Starting process for household no.{i}")
            values = generator.generate_challenge_cond(i)
            values = values.view(48*424).cpu().detach().numpy()
            df.insert(i,key,values)
        
    df.to_csv(f'../../../figures/{household}/{TAG}/cond.csv',index=False)

def analysisGen(household,network,generator,store_path,TAG):
    # Create the numpy array with datetime64[m] dtype
    loadModel(model=network, path=store_path)
    dates = pd.date_range(start=f"{2021}-01-01 00:00:00", end=f"{2022}-12-31 23:45:00", freq='0.25H')

    # Create a new DataFrame from the filtered array
    df = pd.DataFrame(index=dates)

    with torch.no_grad():
        for i in range(386):
            print(f"Starting process for household no.{i}")
            values = generator.generateAnalysis(int(len(dates)/96),i)
            values = values.view(len(dates)).cpu().detach().numpy()
            df.insert(i,i,values)
        
    df.to_csv(f'../../../figures/{household}/{TAG}/cond.csv')

def analysisGen_multivar(train_size,gen_size,network,generator: Generator,store_path):
    # Create the numpy array with datetime64[m] dtype
    loadModel(model=network, path=f'{store_path}network.pt')
    dates = pd.date_range(start=f"{2021}-01-01 00:00:00", end=f"{2022}-12-31 23:45:00", freq='0.25H')

    # Create a new DataFrame from the filtered array
    df = pd.DataFrame(index=dates)

    with torch.no_grad():
        for i in range(386):
            print(f"Starting process for household no.{i}")
            values = generator.generateAnalysis_multivar(int(len(dates)/train_size),i,train_size,gen_size)
            values = generator.train_dataset.retransform(values)
            values = values.view(len(dates)).cpu().detach().numpy()
            df.insert(i,i,values)
        
    df.to_csv(f'{store_path}synthSamples.csv')

def challengeGen(train_size,gen_size,network,generator: Generator,keys,store_path,dataset):
    # Create the numpy array with datetime64[m] dtype
    loadModel(model=network, path=f'{store_path}network.pt')
    dates = pd.date_range(start=f"{2021}-01-01 00:00:00", end=f"{2022}-12-31 23:45:00", freq='0.25H')

    # Create a new DataFrame from the filtered array
    df = pd.DataFrame(index=dates)

    with torch.no_grad():
        
        for i in range(dataset.lens_cat_conditions[0]):

            print(f"Starting process for household no.{i}")
            values = generator.generateChallenge(dates,int(len(dates)/train_size),i,train_size,gen_size)
            values = dataset.retransform(values)
            values = values.view(len(dates)).cpu().detach().numpy()
            df.insert(i,i,values)
            
        df.to_csv(f'{store_path}challengeSamples.csv')