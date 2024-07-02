import torch
import matplotlib.pyplot as plt
import time
import pandas as pd
import random
from syndatagenerators.models.ddpm.utils.DiffusionUtilities import DiffusionUtilities
from syndatagenerators.data_preparation.lsm_data import getWeekPeriodicEmbedding,getDayPeriodicEmbedding,getYearPeriodicEmbedding,getSummertimes

class Generator():
    def __init__(self,
                 model:DiffusionUtilities,
                 train_dataset,
                 tag: str,
                 device: str = "cpu",
                 n_steps: int = 1000):
        
        self.ddpm = model
        self.train_dataset = train_dataset
        self.tag = tag
        self.device = device
        self.n_steps = n_steps


    def generate_n(self,n):
        dates = pd.date_range(start=f"{2013}-01-01 00:00:00", end=f"{2014}-09-28 23:30:00", freq='0.5H')
        day = getDayPeriodicEmbedding(dates)
        week = getWeekPeriodicEmbedding(dates)
        year = getYearPeriodicEmbedding(dates)
        st = getSummertimes(dates)
        x_cont_all = torch.stack([torch.from_numpy(st),
                         torch.from_numpy(day)[:,0],
                         torch.from_numpy(day)[:,1],
                         torch.from_numpy(week)[:,0],
                         torch.from_numpy(week)[:,1],
                         torch.from_numpy(year)[:,0],
                         torch.from_numpy(year)[:,1]
                         ]).view(-1,48,7)[:n]
        x_cat = torch.randint(0,119,(n,48,1)).to(self.device)
        
        start_time = time.time()
       
        x = torch.rand((n,1,48)).float().to(self.device)
        x_cont = x_cont_all.float().to(self.device)

        res = self.embedd_gen(x_cat,x_cont,x)

        end_time = time.time() - start_time
        print(f"Generating took {int(end_time/60)} min {int(end_time%60)} sec")
        return res.view(-1,1,48)
    
    def generate_n_cond(self,n):
        
        start_time = time.time()

        y = torch.randint(low= 0,high=self.ddpm.network.num_classes[0],size=(n,1)).to(self.device)
        if len(self.ddpm.network.num_classes) > 1:
            for size in self.ddpm.network.num_classes[1:]:
                y = torch.cat([y,torch.randint(low= 0,high=size,size=(n,1)).to(self.device)],dim=1)
        
        x_cont_all = self.train_dataset.cont
        randNumb = random.randint(0,x_cont_all.shape[0]-(n+1))
        cont = x_cont_all[randNumb:randNumb+n]
        cont = cont[:,0,:].float().to(self.device)

        x = torch.rand((n,1,96)).float().to(self.device)
        res = self.cond_gen(x=x,y=y,cont=cont)

        end_time = time.time() - start_time
        print(f"Generating took {int(end_time/60)} min {int(end_time%60)} sec")
        return res.view(-1,1,96) 
    
    def generate_n_cond_multivar(self,n,train_size,gen_size,cats,conts):
        
        start_time = time.time()
    

        y = cats[:,0,:]
        cont = conts[:,0,:].float().to(self.device)

        x = torch.randn((n,1,train_size)).float().to(self.device)
        res = self.cond_gen(x=x,y=y,cont=cont)

        end_time = time.time() - start_time
        print(f"Generating took {int(end_time/60)} min {int(end_time%60)} sec")
        return res[:,0,:gen_size]
    
    def generateAnalysis(self,batch_size,counter):
        start_time = time.time()
        dates = pd.date_range(start=f"{2021}-01-01 00:00:00", end=f"{2022}-12-31 23:45:00", freq='0.25H')
        week = getWeekPeriodicEmbedding(dates)
        year = getYearPeriodicEmbedding(dates)
        st = getSummertimes(dates)
        
        x_cont_all = torch.cat([
                        torch.from_numpy(week),
                        torch.from_numpy(year),
                        ],dim=1).view(-1,96,4)
        
        area_uniques = self.train_dataset.cont[:,:,5].unique()
        area_cont = area_uniques[counter%len(area_uniques)].repeat((batch_size,96,1))
        cont = torch.cat([x_cont_all,area_cont],dim=2)
        cont = cont[:,0,:].float().to(self.device)

        y = torch.randint(low= 0,high=self.ddpm.network.num_classes[0],size=(batch_size,1)).to(self.device)
        if len(self.ddpm.network.num_classes) > 1:
            for size in self.ddpm.network.num_classes[1:]:
                y = torch.cat([y,torch.randint(low= 0,high=size,size=(batch_size,1)).to(self.device)],dim=1)
        x = torch.rand((batch_size,1,96)).float().to(self.device)
        
        res = self.cond_gen(x=x,y=y,cont=cont)
        end_time = time.time() - start_time
        print(f"Generating took {int(end_time/60)} min {int(end_time%60)} sec")
        return res.view(-1,1,96)
    
    def generateAnalysis_multivar(self,batch_size,counter,train_size,gen_size):
        start_time = time.time()
        dates = pd.date_range(start=f"{2021}-01-01 00:00:00", end=f"{2022}-12-31 23:45:00", freq='0.25H')
        week = getWeekPeriodicEmbedding(dates)
        year = getYearPeriodicEmbedding(dates)
        st = getSummertimes(dates)
        
        x_cont_all = torch.cat([
                        torch.from_numpy(week),
                        torch.from_numpy(year),
                        #torch.from_numpy(st).unsqueeze(1)
                        ],dim=1).view(-1,96,4)
        
        area_uniques = self.train_dataset.cont[:,:,4].unique()
        area_cont = area_uniques[counter%len(area_uniques)].repeat((batch_size,96,1))
        cont = torch.cat([x_cont_all,area_cont],dim=2)
        cont = cont[:,0,:].float().to(self.device)

        y = torch.randint(low= 0,high=self.ddpm.network.num_classes[0],size=(batch_size,1)).to(self.device)
        if len(self.ddpm.network.num_classes) > 1:
            for size in self.ddpm.network.num_classes[1:]:
                y = torch.cat([y,torch.randint(low= 0,high=size,size=(batch_size,1)).to(self.device)],dim=1)
        x = torch.rand((batch_size,4,train_size)).float().to(self.device)
        
        res = self.cond_gen(x=x,y=y,cont=cont)
        end_time = time.time() - start_time
        print(f"Generating took {int(end_time/60)} min {int(end_time%60)} sec")
        return res[:,0,:gen_size]
    
    def generateChallenge(self,dates,batch_size,i,train_size,gen_size):
        start_time = time.time()
        week = getWeekPeriodicEmbedding(dates)
        year = getYearPeriodicEmbedding(dates)
        
        cont = torch.cat([
                        torch.from_numpy(week),
                        torch.from_numpy(year)
                        ],dim=1).view(-1,96,4)
        
        cont = cont[:,0,:].float().to(self.device)

        y = torch.tensor([i]).repeat([batch_size,1]).to(self.device)
        x = torch.rand((batch_size,1,train_size)).float().to(self.device)
        
        res = self.cond_gen(x=x,y=y,cont=cont)
        end_time = time.time() - start_time
        print(f"Generating took {int(end_time/60)} min {int(end_time%60)} sec")
        return res[:,0,:gen_size]

    def generate_challenge_cond(self,cond):
        start_time = time.time()
        dates = pd.date_range(start=f"{2013}-01-01 00:00:00", end=f"{2014}-02-28 23:30:00", freq='0.5H')
        day = getDayPeriodicEmbedding(dates)
        week = getWeekPeriodicEmbedding(dates)
        year = getYearPeriodicEmbedding(dates)
        st = getSummertimes(dates)
        x_cont_all = torch.stack([torch.from_numpy(st),
                        torch.from_numpy(day)[:,0],
                        torch.from_numpy(day)[:,1],
                        torch.from_numpy(week)[:,0],
                        torch.from_numpy(week)[:,1],
                        torch.from_numpy(year)[:,0],
                        torch.from_numpy(year)[:,1]
                        ]).view(-1,48,7)
        x_cat = torch.tensor([cond]).repeat([424,48,1]).to(self.device)
        
        #for date in range(424):
        x = torch.rand((424,1,48)).float().to(self.device)
        x_cont = x_cont_all.float().to(self.device)

        res = self.embedd_gen(x_cat,x_cont,x)
        # if date != 0:
        #     results = torch.cat((results,res[0]),dim=0)
        # else:
        #     results = res[0]
        # print(results.shape)

        end_time = time.time() - start_time
        print(f"Generating took {int(end_time/60)} min {int(end_time%60)} sec")
        return res.view(-1,1,48)

    def embedd_gen(self, x_cat, x_cont,y):
        for t in range(self.n_steps-1,-1,-1):
            timestamp = torch.tensor([t]).to(self.device)
            y = self.ddpm.subtractNoise(x_cat,x_cont,y,timestamp)
        return y
    
    def cond_gen(self,x,y,cont):
        for t in range(self.n_steps-1,-1,-1):
            t = torch.tensor([t]).to(self.device)
            x = self.ddpm.subtractNoise(x=x,t=t,y=y,cont=cont)
        return x
    
    def generate_one_cond_sample(self,cat,cont,seq_len=96):
        x = torch.rand((1,1,seq_len)).float().to(self.device)
        for t in range(self.n_steps-1,-1,-1):
            t = torch.tensor([t]).to(self.device)
            x = self.ddpm.subtractNoise(x=x,t=t,y=cat.to(self.device),cont=cont.to(self.device))
        return x
