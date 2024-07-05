import os
import argparse

import pytorch_lightning as pl
from syndatagenerators.data_preparation.datasets import LondonDataModule
from wgan_baseline_model import BaselineWGAN, Generator, Critic
from syndatagenerators.models.utils.callbacks import DiscriminativeCallback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


if __name__ == '__main__':
    # define directory where data is loaded from
    DATA_DIR = '../../data/Small_LCL_Data/'
    # define directory where train data is saved
    TRAIN_DATA_DIR = 'train_data'
    #directory where model is saved/loaded from
    CKPT_DIR = 'ckpt/'
    #asset lists (of London Smart Meter)
    list_assets = ['MAC000002', 'MAC000003', 'MAC000004', 'MAC000005', 'MAC000006', 'MAC000007', 'MAC000008',
                   'MAC000009', 'MAC000010', 'MAC000011']

    TRAIN = True
    LOAD = False
    EARLY_STOPPING = False

    #argument parsing for wgan
    parser = argparse.ArgumentParser(description='wgan Baseline')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default 64)')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--name', type=int, default=3, help='name under which model is saved')
    parser.add_argument('--input_shape', type=tuple, default=(1, 96), help='input shape of training samples (i.e. tu'
                                                                          'ple of feature dims and seq len')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimension of latent noise vector')
    parser.add_argument('--kernel_size_dis', type=int, default=3, help='length of kernel in conv blocks of'
                                                                       ' discriminator')
    parser.add_argument('--kernel_size_gen', type=int, default=3, help='length of kernel in conv blocks of generator')
    parser.add_argument('--channel_nb_dis', type=int, default=32, help='channel dim in discriminator')
    parser.add_argument('--channel_nb_gen', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='RMSProp', help='optimizer used in training')
    parser.add_argument('--lr_dis', type=float, default=0.0001, help='learning rate of discriminator')
    parser.add_argument('--lr_gen', type=float, default=0.0001, help='learning rate of generator')
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='penalty term parameter')
    parser.add_argument('--n_critic', type=int, default=5, help='number of training iterations of critic w.r.t.'
                                                                ' generator')

    args = parser.parse_args()
    model = BaselineWGAN(Generator, Critic, args)
    dm = LondonDataModule(assets=list_assets, window_length=48, overlap=46,
                          train_data_dir=TRAIN_DATA_DIR,
                          labels=False)

    ###CALLBACKS
    FILENAME = f'{model.__class__.__name__}_{model.name}'
    checkpoint_callback = ModelCheckpoint(dirpath=CKPT_DIR, filename=FILENAME, every_n_epochs=10)

    if EARLY_STOPPING:
        early_stopping_callback = EarlyStopping('discriminative_loss', patience=30,
                                                stopping_threshold=-0.1)
        trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[#DiscriminativeCallback(),
                                                                                     checkpoint_callback,
                                                                                     early_stopping_callback],
                         check_val_every_n_epoch=5)
    else:
        trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[checkpoint_callback],
                                                                                    #          DiscriminativeCallback()],
                             check_val_every_n_epoch=5)
    if TRAIN:
        trainer.fit(model, dm)
    if LOAD:
        ckpt_path = os.path.join(CKPT_DIR, FILENAME)
        model.load_from_checkpoint(checkpoint_path=ckpt_path)
