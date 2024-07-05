import os
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from syndatagenerators.models.progressive_gan.trainer import TrainerProGAN
from syndatagenerators.models.cond_progressive_gan.trainer import TrainerCPGAN
from syndatagenerators.models.utils.callbacks import DiscriminativeCallback
from syndatagenerators.data_preparation.datasets import LondonDataset

if __name__ == '__main__':
    # define directory where train data is saved
    TRAIN_DATA_DIR = '../train_data'
    # directory where model is saved/loaded from
    CKPT_DIR = '../ckpt/'
    # path to configuration file
    CONFIG_PATH = 'utils/config_progan.yml'

    # asset lists (of London Smart Meter)
    list_assets = ['MAC' + str(i).zfill(6) for i in range(2, 10)]
    # list_assets_val = ['MAC' + str(i).zfill(6) for i in range(51, 55)]

    # specify training parameters

    # use ID's of households as condition?
    LABELS = True
    # train model?
    TRAIN = True
    # load from checkpoint?
    LOAD = False
    # early stopping regarding predefined metric?
    EARLY_STOPPING = True

    config = yaml.safe_load(open(CONFIG_PATH, 'r'))
    train_params = config['train_params']
    dis_params = config['dis_params']
    gen_params = config['gen_params']

    # initialize model
    if LABELS:
        nb_cls = len(list_assets)
        model = TrainerCPGAN(train_params=train_params, dis_params=dis_params, gen_params=gen_params, nb_classes=nb_cls)
    else:
        model = TrainerProGAN(train_params=train_params, dis_params=dis_params, gen_params=gen_params)

    # dataloaders for training and validation
    dataset = LondonDataset(assets=list_assets, window_length=config['data_params']['window_len'],
                            overlap=config['data_params']['overlap'], train_data_dir=TRAIN_DATA_DIR,
                            labels=LABELS)


    # split into train and validation set
    dataset_train, dataset_val = random_split(dataset, [len(dataset) - 1000, 1000])

    # dataloaders for training and validation
    loader_train = DataLoader(dataset_train, batch_size=config['train_params']['batch_size'], shuffle=True,
                              num_workers=8)
    loader_val = DataLoader(dataset_val, batch_size=len(dataset_val), shuffle=False)

    # CALLBACKS
    FILENAME = f'{model.__class__.__name__}_{model.name}'
    checkpoint_callback = ModelCheckpoint(dirpath=CKPT_DIR, filename=FILENAME, every_n_epochs=10)

    if EARLY_STOPPING:
        early_stopping_callback = EarlyStopping('generator loss', patience=30,
                                                stopping_threshold=-5)

        trainer = pl.Trainer(max_epochs=config['train_params']['epochs'],
                             callbacks=[DiscriminativeCallback(), checkpoint_callback, early_stopping_callback],
                             check_val_every_n_epoch=5)

    else:
        trainer = pl.Trainer(max_epochs=config['train_params']['epochs'],
                             callbacks=[DiscriminativeCallback(), checkpoint_callback],
                             check_val_every_n_epoch=5)

    if LOAD:
        ckpt_pth = os.path.join(CKPT_DIR, FILENAME)
        model.load_from_checkpoint(checkpoint_path=ckpt_pth)

    if TRAIN:
        trainer.fit(model, loader_train, loader_val)
