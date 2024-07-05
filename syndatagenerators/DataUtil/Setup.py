import pandas as pd

from DataUtil import Logger

from data_preparation import dataprep_utils

from Trainers.vaeTrainer import BasicVAETrainer
from models.vae.FFVAE import FFVAE
from models.vae.CNNVAE import BasicCNNVAE
from models.vae.HybridVAE import OneToManyHybridLSTMFFVAE

from Trainers.ganTrainer import WassersteinGPGANTrainer
from models.GAN.CNNGAN import DCWGAN
from models.GAN.HybridGAN import ManyToManyHybridLSTMFFWGAN
from models.GAN.WGAN import WGAN

from Trainers.cganTrainer import WassersteinGPCEmbeddingsGANTrainer
from models.GAN.Conditional.CHybridGAN import BasicEmbeddingCManyToManyHybridLSTMWFFGAN

def getModel(cfg):
    if(cfg.Model.name == "WGAN"):
        return WGAN(num_hiddenunits=cfg.Model.num_hiddenunits, latent_dim=cfg.Model.latent_dim, num_hiddenlayers=cfg.Model.num_hiddenlayers, seq_length=cfg.Dataset.seq_length, device=cfg.Model.device)
    elif(cfg.Model.name == "FFVAE"):
        return FFVAE(latent_dim=cfg.Model.latent_dim, seq_length=cfg.Dataset.seq_length, num_hiddenunits=cfg.Model.num_hiddenunits, num_hiddenlayers=cfg.Model.num_hiddenlayers, device=cfg.Model.device)
    elif(cfg.Model.name == "BasicCNNVAE"):
        return BasicCNNVAE(latent_dim=cfg.Model.latent_dim, seq_length=cfg.Dataset.seq_length, num_hiddenchannels=cfg.Model.num_hiddenchannels, 
            num_hiddenlayers=cfg.Model.num_hiddenlayers, device=cfg.Model.device)
    elif(cfg.Model.name == "OneToManyHybridLSTMFFVAE"):
        return OneToManyHybridLSTMFFVAE(cfg.Model.latent_dim, num_hiddenunits=cfg.Model.num_hiddenunits, seq_length=cfg.Dataset.seq_length, num_hiddenlayers=cfg.Model.num_hiddenlayers, device=cfg.Model.device)
    elif(cfg.Model.name == "DCWGAN"):
        return DCWGAN(seq_length=cfg.Dataset.seq_length, latent_dim=cfg.Model.latent_dim, num_hiddenchannels=cfg.Model.num_hiddenchannels, num_hiddenlayers=cfg.Model.num_hiddenlayers, device=cfg.Model.device)
    elif(cfg.Model.name == "ManyToManyHybridLSTMFFWGAN"):
        return ManyToManyHybridLSTMFFWGAN(num_hiddenunits=cfg.Model.num_hiddenunits, num_features=cfg.Model.num_features, num_hiddenlayers=cfg.Model.num_hiddenlayers, seq_length=cfg.Dataset.seq_length, 
            device=cfg.Model.device)    
    elif(cfg.Model.name == "BasicEmbeddingCManyToManyHybridLSTMWFFGAN"):
        return BasicEmbeddingCManyToManyHybridLSTMWFFGAN(num_hiddenunits=cfg.Model.num_hiddenunits, num_hiddenlayers=cfg.Model.num_hiddenlayers, num_features=cfg.Model.num_features, device=cfg.Model.device, seq_length=cfg.Dataset.seq_length,
                                                         lens_cat_conditions=dataset.lens_cat_conditions, num_cont=dataset.num_cont, embedding_dim=cfg.Model.embedding_dim, embedding_out=cfg.Model.embedding_out)
    else:
        raise KeyError(f"Unknown Model: {cfg.Model.name}")

#maybe seperate dataframe and dataset
def getDataset(cfg):
    if(cfg.Dataset.name == "LondonSmartMeter"):
        hdf = dataprep_utils.create_LSM_datastore(hdf_file=cfg.Dataset.path)
        df = dataprep_utils.read_multiple_households_df(cfg.Dataset.keyset, hdf)
        hdf.close()
        return dataprep_utils.LSMDataset(df, seq_len=cfg.Dataset.seq_length)
    elif(cfg.Dataset.name == "OMDataset"):
        hdf = dataprep_utils.create_OM_datastore(hdf_file=cfg.Dataset.path)
        df = dataprep_utils.read_number_households_df(cfg.Dataset.number_of_households, hdf)
        ds = dataprep_utils.OMDataset(df, seq_len=cfg.Dataset.seq_length)
        hdf.close()
        return ds
    elif(cfg.Dataset.name == "PrivateOMNumberCondDataset"):
        hdf = dataprep_utils.create_OM_datastore(hdf_file=cfg.Dataset.path)
        df = dataprep_utils.read_number_households_df(cfg.Dataset.num_households, hdf)
        hdf.close()
        ds =  dataprep_utils.OMCondDataset(df, seq_len=cfg.Dataset.seq_length, cat_conditions=cfg.Dataset.cat_conditions, cont_conditions=cfg.Dataset.cont_conditions)
        return ds
    else:
        raise KeyError(f"Unknown Dataset: {cfg.Dataset.name}")

def getTrainer(cfg, model, dataset):
    if(cfg.Trainer.name == "BasicVAETrainer"):
        return BasicVAETrainer(model=model, dataset=dataset, seq_length=cfg.Dataset.seq_length, test_data_coefficient=cfg.Dataset.test_data_coefficient, 
            batch_size=cfg.Trainer.batchsize, device=cfg.Model.device, kl_weight=cfg.Trainer.kl_weight, lr=cfg.Trainer.lr)
    elif(cfg.Trainer.name == "WassersteinGPGANTrainer"):
        return WassersteinGPGANTrainer(model=model, dataset=dataset, seq_length=cfg.Dataset.seq_length, test_data_coefficient=cfg.Dataset.test_data_coefficient, 
            batch_size=cfg.Trainer.batchsize, device=cfg.Model.device, learning_rate=cfg.Trainer.learning_rate, critic_iterations=cfg.Trainer.critic_iterations, gp_scale=cfg.Trainer.gp_scale)
    else:
        raise KeyError(f"Unknown Trainer: {cfg.Trainer.name}")

def getLogger(cfg, model, trainer):
    extra_tags = cfg.Logger.extra_tags+"-"+cfg.Dataset.name +"-"+trainer.summary
    if(cfg.Logger.name == "VAELogger"):
        return  Logger.VAELogger(model=model, test_data_cuda=trainer.test_data_cuda, test_data_local=trainer.test_data_local, seq_length=cfg.Dataset.seq_length, device=cfg.Model.device, extra_tags=extra_tags, cfg=cfg)
    elif(cfg.Logger.name == "GANLogger"):
        return Logger.GANLogger(model=model, test_data_local=trainer.test_data_local, test_data_cuda=trainer.test_data_cuda, seq_length=cfg.Dataset.seq_length, device=cfg.Model.device, extra_tags=extra_tags, cfg=cfg)
    elif(cfg.Logger.name == "CGANLogger"):
        return Logger.CGANLogger(model=model, test_cond_local=trainer.test_cond_local, test_data_local=trainer.test_data_local, test_cond_cuda=trainer.test_cond_cuda, test_data_cuda=trainer.test_data_cuda, 
            seq_length=cfg.Dataset.seq_length, device=cfg.Model.device, extra_tags=extra_tags, cfg=cfg)
    else:
        raise KeyError(f"Unknown Logger: {cfg.Logger.name}")
