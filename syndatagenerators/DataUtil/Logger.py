import multiprocessing as mp
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from metrics.mmd_score import mmd
from typing import List
from metrics.Scores import Scores
from statsmodels.tsa.stattools import acf
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE

def bootstrap(population:np.array) -> List:
    bootstrap = []
    for _ in range(len(population)):
        bootstrap.append(population[torch.randint(low=0, high=len(population), size=(1,1))])
    return bootstrap


def showReconstructions(x:torch.Tensor, x_recon:torch.Tensor, key:str, df:pd.DataFrame):
    samples = x_recon.detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    for k in range(min(16, x.size(0))):
        plt.subplot(4, 4, 1 + k)
        plt.plot(x[k, :].cpu(), label='observation')
        plt.plot(samples[k,:], label='sample')
        plt.ylim([0, 1])
        if k % 4 == 0:
            plt.ylabel('norm. power')
        if k > 11:
            plt.xlabel('time idx')
        print(Scores.energy_score(x[k,:].cpu().numpy(), samples[k,:]))
    plt.legend()
    title_str = 'LCLid: ' + key + ' (' + df['stdorToU'][0] + ')'
    plt.suptitle('\n\nSampled VAE-Reconstruction and Observation\n' + title_str)
    plt.show()

def showSamples(model:torch.nn.Module, key:str, df:pd.DataFrame):
    x_samples = model.getSample(16).squeeze().detach().cpu()
    plt.figure(figsize=(10,10))
    for k in range(16):
        plt.subplot(4,4,1+k)
        plt.plot(x_samples[k,:],label='sample')
        plt.ylim([0,1])
        if k%4 == 0:
            plt.ylabel('norm. power')
        if k>11:
            plt.xlabel('time idx')
    plt.legend()
    title_str = 'LCLid: ' + key + ' (' + df['stdorToU'][0] + ')'
    plt.suptitle('\n\nRandomly Generated Samples From VAE\n' + title_str)
    plt.show()

def getBootstrapSampleMean(samples:torch.Tensor, nlags:int=30):
    acf_samples_mean = np.zeros_like(len(samples), dtype=np.float32)
    bootstraped_samples = bootstrap(samples.numpy())
    for i in range(len(bootstraped_samples)):
        try:
            acf_samples_mean = acf_samples_mean + acf(bootstraped_samples[i], nlags=nlags, fft=True)
        except Exception as e:
            pass
    acf_samples_mean = acf_samples_mean / len(bootstraped_samples)
    return acf_samples_mean

def getACFGraph(samples:torch.Tensor, gt:torch.Tensor, pool:mp.Pool, num_bootstraps:int=500, nlags:int=30) -> plt.figure:
    np.seterr(all="raise")
    figure = plt.figure()
    plt.axhline(0, color="black", linestyle="--")
    means = []
    for i in range(num_bootstraps // mp.cpu_count()):
        means = means + pool.starmap(getBootstrapSampleMean, [(samples, nlags) for _ in range(mp.cpu_count())])
    means = means + pool.starmap(getBootstrapSampleMean, [(samples, nlags) for _ in range((num_bootstraps % mp.cpu_count()))])
    acf_samples_std = []
    for i in range(len(means[0])):
        acf_samples_std.append(np.sqrt(np.std([means[j][i] for j in range(len(means))])))
    acf_samples_std = np.array(acf_samples_std)
    #print(psutil.Process().memory_info().rss / (1024 * 1024))
    mean = 0
    for i in range(len(means)):
        mean = mean + means[i]
        line, = plt.plot(means[i], color="orange", alpha=0.05)
    line.set_label("acf-Mean sample")
    
    mean = mean / num_bootstraps

    plt.plot(mean + acf_samples_std, color="gray", linestyle="--")
    plt.plot(mean - acf_samples_std, color="gray", linestyle="--", label="acf-mean +/- std")
    plt.plot(mean + 2 * acf_samples_std, color="gray", linestyle=":")
    plt.plot(mean - 2 * acf_samples_std, color="gray", linestyle=":", label="acf-mean +/- 2std")
    
    acf_gt_mean =  np.zeros_like(len(gt), dtype=np.float32)
    for i in range(len(gt)):
        try:
            acf_gt_mean = acf_gt_mean + acf(gt[i].detach().cpu().numpy(), nlags=nlags, fft=True)
        except Exception as e:
            pass
    acf_gt_mean = acf_gt_mean / len(gt)
    plt.plot(acf_gt_mean, label="acf-Mean gt")
    plt.legend()
    plt.xlim([0, nlags])
    plt.xlabel("lag")
    
    np.seterr(all="print")
    return figure

def getGenerationPerformanceGraph(seq_length:int, x:torch.Tensor, x_recon:torch.Tensor) -> plt.figure:
    figure = plt.figure(figsize=(10,10))
    for k in range(min(16, x.size(0))):
        plt.subplot(4,4,1+k)
        plt.plot(x[k,:],label='real')
        plt.plot(x_recon[k,:],label='reconstructed')
        plt.ylim([0,1])
        plt.xlim([0, seq_length])
        if k%4 == 0:
            plt.ylabel('norm. power')
        if k>11:
            plt.xlabel('time idx')
    plt.legend()
    return figure

def getGenerationsGraph(y:torch.Tensor, x=None) -> plt.figure:
    figure = plt.figure(figsize=(10,10))
    for k in range(min(16, y.size(0))):
        plt.subplot(4,4,1+k)
        if x is not None: 
            plt.plot(x[k,:],label='real')
        plt.plot(y[k,:],label='synth')
        plt.ylim([0,1])
        if k%4 == 0:
            plt.ylabel('norm. power')
        if k>11:
            plt.xlabel('time idx')
    plt.legend()
    return figure

def getVarMeanCompareGraph(data, samples):
    data_var, data_mean = torch.var_mean(data, 0)
    data_var = torch.sqrt_(data_var)
    samples_var, samples_mean = torch.var_mean(samples, 0)
    samples_var = torch.sqrt_(samples_var)
    figure = plt.figure()
    #plt.xlim([0, seq_length])
    plt.ylim([0,1])
    plt.ylabel('norm. power')
    plt.xlabel('time index')
    plt.plot(data_mean, label="real data")
    plt.plot(data_mean + data_var, color="gray", linestyle="--")
    plt.plot(data_mean - data_var, color="gray", linestyle="--", label="real-mean +/- std")
    plt.plot(data_mean + 2 * data_var, color="gray", linestyle=":")
    plt.plot(data_mean - 2 * data_var, color="gray", linestyle=":", label="real-mean +/- 2std")

    plt.plot(samples_mean, label="samples data", color="red", alpha=0.66)
    plt.plot(samples_mean + samples_var, color="red", linestyle="--", alpha=0.66)
    plt.plot(samples_mean - samples_var, color="red", linestyle="--", label="samples-mean +/- std", alpha=0.66)
    plt.plot(samples_mean + 2 * samples_var, color="red", linestyle=":", alpha=0.66)
    plt.plot(samples_mean - 2 * samples_var, color="red", linestyle=":", label="samples-mean +/- 2std", alpha=0.66)
    plt.legend()
    return figure

def getQuantileGraph(data:torch.tensor, samples:torch.tensor):# hier noch die variance zugeben und dann oben nicht mehr benutzen
    q_data = torch.quantile(data, torch.tensor([0.1, 0.9]), dim=0, keepdim=True).squeeze()
    data_var, data_mean = torch.var_mean(data, 0)
    data_var = torch.sqrt_(data_var)
    q_samples = torch.quantile(samples, torch.tensor([0.1, 0.9]), dim=0, keepdim=True).squeeze()
    samples_var, samples_mean = torch.var_mean(samples, 0)
    samples_var = torch.sqrt_(samples_var)
    figure = plt.figure()
    #plt.xlim([0, seq_length])
    plt.axhline(0, color="black", linestyle="--")
    plt.ylim([np.minimum(-0.05, q_samples.min().numpy() - 0.05), np.maximum(1, q_samples.max().numpy() + 0.05)])
    plt.ylabel('norm. power')
    plt.xlabel('time index')
    plt.plot(data_mean, label="real data mean")
    plt.plot(data_mean + data_var, color="gray", linestyle="-.", label="real-mean + std",)
    plt.plot(q_data[0], color="gray", linestyle=(0, (1, 1)), label="data 10% quantile")
    plt.plot(q_data[1], color="gray", linestyle=(0, (3, 3)), label="data 90% quantile")
 
    plt.plot(samples_mean, label="samples data mean", color="red", alpha=0.66)
    plt.plot(samples_mean + samples_var, color="red", linestyle="-.", label="samples-mean + std", alpha=0.66)
    plt.plot(q_samples[0], color="red", linestyle=(0, (1, 1)), label="samples 10% quantile", alpha=0.66)
    plt.plot(q_samples[1], color="red", linestyle=(0, (3, 3)), label="samples 90% quantile", alpha=0.66)
    plt.legend()
    return figure

def getCondPerformanceGraph(data:torch.Tensor, synth:torch.tensor, seq_length):
    print(data.size())
    figure = plt.figure()
    plt.xlim([0, seq_length])
    plt.ylim([0,1])
    plt.ylabel('norm. power')
    plt.xlabel('time index')
    #for i in range(data.size(0)):
    line, = plt.plot(data[0], color="red", alpha=0.3)
    line.set_label("real")
    for i in range(synth.size(0)):
        line, = plt.plot(synth[i], color="blue", alpha=0.3)
    line.set_label("synth")
    plt.legend()
    return figure

def getTSNEGraph(data, y, pool:mp.Pool):
    tsne = TSNE()
    tsne_results = []
    for result in pool.map(tsne.fit_transform, (data, y)):
        tsne_results.append(result)
    figure = plt.figure()
    plt.scatter(tsne_results[0][:,0], tsne_results[0][:,1], alpha=max(20/len(data), 0.02), label="real")
    plt.scatter(tsne_results[1][:,0], tsne_results[1][:,1], alpha=max(10/len(data), 0.01), label="synth")
    plt.legend()
    return figure

def getDistributionGraph(data, samples):
    figure = plt.figure(figsize=(30,30))
    for i in range(48):
        plt.subplot(7,7, 1+i)
        plt.hist(data[:,i], bins="auto", label="real")
        plt.hist(samples[:, i], bins="auto", fc=(1, 0.5, 0, 0.3), label="synth")
    plt.legend()
    return figure



def logModel(model, writer_dir, input, cfg):
    model.eval()
    writer = SummaryWriter(writer_dir)
    writer.add_graph(model, input_to_model=input)
    writer.add_text("cfg", str(cfg), 0)
    writer.flush()
    writer.close()
    model.train(True)


class LogProcess(mp.Process):

    def __init__(self):
        super(LogProcess, self).__init__()
        self.terminate_value = mp.Value("i", 0)
        self.inq = mp.Queue()
        manager = mp.Manager()
        self.pool = manager.Pool(10)
        self.lock = mp.Lock()

    def run(self):
        while((self.terminate_value.value == 0) or (self.inq.qsize() != 0)):
            try:
                target, args = self.inq.get(block=True, timeout=5)
                process = mp.Process(target=target, args=(args + (self.lock, self.pool)))
                process.start()
                process.join()
            except:
                pass
        self.pool.close()
        self.pool.join()

class VAELogger():

    def __init__(self, model, test_data_local, test_data_cuda, seq_length, device, extra_tags:str, cfg):
        self.model = model
        self.test_dataset_local = test_data_local
        self.test_dataset_cuda = test_data_cuda
        self.seq_length = seq_length
        self.device = device
        self.writer_dir = "runs/VAE/"+model.__class__.__name__+"-"+str(extra_tags)+"+"+time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())
        self.fixed_latent_vektor = model.getLatentVector(len(test_data_local), seq_length, device=device)
        #self.fixed_latent_vektor = torch.randn(len(test_data_cuda), model.latent_dim).to(device)
        
        self.lp = LogProcess()
        self.lp.start()

        logModel(model, self.writer_dir, test_data_cuda[:16], cfg)

    def log(self, loss, epoch, s_time=None):
        with torch.no_grad():
            self.model.eval()

            y = self.model.decode(self.fixed_latent_vektor.detach()).squeeze()
            mmd_score = mmd(y, self.test_dataset_cuda, self.device, None)

            y = y.detach().cpu().squeeze()
            x = self.test_dataset_cuda[:16]
            x_recon = self.model(x).detach().cpu().squeeze()
            self.model.train(True)
            
            x = self.test_dataset_local[:16]
            try:
                mse_loss = loss["mse_loss"]
                kl_loss = loss["kl_loss"]
                self.lp.inq.put((logVAE, (mse_loss, kl_loss, epoch, mmd_score, y, x, x_recon, self.test_dataset_local, self.seq_length, self.writer_dir)))
                print(f'epoch: {epoch}, mse: {mse_loss:.5f}, kl: {kl_loss:.5f}, mmd: {mmd_score:.5f}, {time.time() - s_time if (s_time is not None) else "None" } seconds')
            except:
                print("Loss doesn't contain the right keys")

            return mmd_score

    
def logVAE(mse_loss, kl_loss, epoch, mmd_score, y, x, x_recon, test_dataset_local, seq_length, writer_dir, lock, pool):
    stime = time.time()
    acf_figure = getACFGraph(y, test_dataset_local, pool)
    generation_figure = getGenerationPerformanceGraph(seq_length, x, x_recon)
    
    #var_mean_compare = getVarMeanCompareGraph(test_dataset_local, y)
    q_graph = getQuantileGraph(test_dataset_local, y)
    generations_figure = getGenerationsGraph(y, test_dataset_local)
    tsneGraph = getTSNEGraph(test_dataset_local, y, pool)


    lock.acquire()
    writer = SummaryWriter(writer_dir)
    writer.add_figure("Autocorrelation", acf_figure, epoch)

    writer.add_scalar("MSE-Loss", mse_loss, epoch)
    writer.add_scalar("KL-Loss", kl_loss, epoch)
    writer.add_scalar("MMD", mmd_score, epoch)

    writer.add_figure("reconstructions", generation_figure, epoch)
    #writer.add_figure("Var Mean", var_mean_compare, epoch)
    writer.add_figure("Quantiles", q_graph, epoch)
    writer.add_figure("samples", generations_figure, epoch)
    writer.add_figure("TSNE", tsneGraph, epoch)
    writer.flush()
    writer.close()
    lock.release()
    print(f"Logging epoch {epoch} took {time.time() - stime} seconds")
 
class GANLogger():

    def __init__(self, model, test_data_local, test_data_cuda, seq_length, device, extra_tags:str, cfg):
        self.generator = model.generator
        self.test_dataset_local = test_data_local
        self.test_dataset_cuda = test_data_cuda
        self.seq_length = seq_length
        self.device = device
        self.writer_dir = "runs/GAN/"+model.__class__.__name__+"-"+str(extra_tags)+"+"+time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())
        self.fixed_latent_vektor = model.getLatentVector(len(test_data_local), seq_length, device=device)
        self.lp = LogProcess()
        self.lp.start()

        #logModel(self.discriminator, self.writer_dir, test_data_cuda[:16])
        logModel(self.generator, self.writer_dir, self.fixed_latent_vektor, cfg)

    def log(self, loss, epoch, s_time=None):
        with torch.no_grad():
            self.generator.eval()

            y = self.generator(self.fixed_latent_vektor.detach()).squeeze()
            mmd_score = mmd(y, self.test_dataset_cuda, self.device, None)

            y = y.detach().cpu().squeeze()
            self.generator.train(True)

            try:
                generator_loss = loss["generator_loss"]
                discriminator_loss = loss["discriminator_loss"]
                self.lp.inq.put((logGAN, (generator_loss, discriminator_loss, epoch, mmd_score, y, self.test_dataset_local, self.writer_dir)))
                print(f'epoch: {epoch}, generator-loss: {generator_loss:.5f}, discriminator-loss: {discriminator_loss:.5f}, mmd: {mmd_score:.5f}, {time.time() - s_time if (s_time is not None) else "None" } seconds')
            except:
                print("Loss doesn't contain the right keys")
            return mmd_score    
        
def logGAN(generator_loss, discriminator_loss, epoch, mmd_score, y, test_dataset_local, writer_dir, lock, pool):
    stime = time.time()
    acf_figure = getACFGraph(y, test_dataset_local, pool)
    
    q_graph = getQuantileGraph(test_dataset_local, y)
    generations_figure = getGenerationsGraph(y, test_dataset_local)
    #var_mean_compare = getVarMeanCompareGraph(test_dataset_local, y)
    tsneGraph = getTSNEGraph(test_dataset_local, y, pool)

    lock.acquire()
    writer = SummaryWriter(writer_dir)
    writer.add_figure("Autocorrelation", acf_figure, epoch)

    writer.add_scalar("Generator-Loss", generator_loss, epoch)
    writer.add_scalar("Discriminator-Loss", discriminator_loss, epoch)
    writer.add_scalar("MMD", mmd_score, epoch)

    writer.add_figure("samples", generations_figure, epoch)
    #writer.add_figure("Var Mean", var_mean_compare, epoch)
    writer.add_figure("Quantiles", q_graph, epoch)
    writer.add_figure("TSNE", tsneGraph, epoch)
    writer.flush()
    writer.close()
    lock.release()
    print(f"Logging epoch {epoch} took {time.time() - stime} seconds")


class CGANLogger():

    def __init__(self, model, test_cond_local, test_data_local, test_cond_cuda, test_data_cuda, seq_length, device, extra_tags:str, cfg):
        self.generator = model.generator
        self.test_cond_local = test_cond_local
        self.test_data_local = test_data_local.squeeze()
        self.test_cond_cuda = test_cond_cuda
        self.test_data_cuda = test_data_cuda.squeeze()
        self.seq_length = seq_length
        self.device = device
        self.writer_dir = "runs/GAN/"+model.__class__.__name__+"-"+str(extra_tags)+"+"+time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())
        self.fixed_latent_vektor = model.getLatentVector(len(test_data_local), seq_length, device=device)
        
        self.lp = LogProcess()
        self.lp.start()

        #logModel(self.discriminator, self.writer_dir, test_data_cuda[:16])
        logModel(self.generator, self.writer_dir, (self.fixed_latent_vektor, self.test_cond_cuda), cfg)

    def log(self, loss, epoch, s_time=None):
        with torch.no_grad():
            self.generator.eval()

            y = self.generator(self.fixed_latent_vektor.detach(), self.test_cond_cuda.detach()).squeeze()
            mmd_score = mmd(y, self.test_data_cuda, self.device, None)

            y = y.detach().cpu().squeeze()
            self.generator.train(True)

            try:
                generator_loss = loss["generator_loss"]
                discriminator_loss = loss["discriminator_loss"]
                self.lp.inq.put((logGAN, (generator_loss, discriminator_loss, epoch, mmd_score, y, self.test_data_local, self.writer_dir)))
                print(f'epoch: {epoch}, generator-loss: {generator_loss:.5f}, discriminator-loss: {discriminator_loss:.5f}, mmd: {mmd_score:.5f}, {time.time() - s_time if (s_time is not None) else "None" } seconds')
            except:
                print("Loss doesn't contain the right keys")
            return mmd_score    
        