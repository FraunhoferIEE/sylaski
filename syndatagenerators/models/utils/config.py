import torch
from typing import List, Optional


def _get_data_params(start_date: str = '2021-08-01', end_date: str = '2021-10-31', window_length: int = 24,
                     window_overlap: int = 0, asset_ids: Optional[List] = None,
                     num_classes: int =2):
    data_params = {
        'start_date': start_date,
        'end_date': end_date,
        'window_length': window_length,
        'overlap': window_overlap,
        'asset_ids': asset_ids,
        'num_classes': num_classes
    }
    return data_params


def _get_train_params(model_cls, dis_cls, gen_cls,
                      name=1000, epochs=2000, sample_cycle=10, batchsize=64,
                      lr=0.0001, input_shape=(1, 24),
                      noise=False, clip_val=0.01, n_critic=5,
                      lambda_gp=0.1, optimizer='Adam'):
    train_params = {
        'model_cls': model_cls,
        'dis_cls': dis_cls,
        'gen_cls': gen_cls,
        'epochs': epochs,
        'batchsize': batchsize,
        'lr': lr,
        'input_shape': input_shape,
        'n_critic': n_critic,
        'add_noise': noise,
        'sample_cycle': sample_cycle,
        'name': name,
        'clip_value': clip_val,
        'lambda_gp': lambda_gp
    }

    if optimizer == 'Adam':
        train_params['optimizer'] = torch.optim.Adam
        train_params['opt_args'] = {'betas': (0.9, 0.999)}

    elif optimizer == 'RMSprop':
        train_params['optimizer'] = torch.optim.RMSprop
        train_params['opt_args'] = {'alpha': 0.99}

    return train_params


def _get_generator_params(norm_gen='batch', activ_fct=None, kernel_size=3, stride=1,
                          padding=1, dilation=1, latent_dim=100, input_shape=(1, 24)):
    gen_params = {
        'noise_type': 'normal',
        'input_shape': input_shape,
        'normalization': norm_gen,
        'activ_fct': activ_fct,
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'latent_dim': latent_dim,
        'dilation': dilation
    }

    if activ_fct == 'tanh':
        gen_params['ivl'] = (-0.5, 0.5)

    else:
        gen_params['ivl'] = (0, 1)

    return gen_params


def _get_discriminator_params(input_shape=(1, 24), norm_dis='spectral', kernel_size=3,
                              stride=2, padding=1, dilation=1, dropout='dropout', drop_rate=0.0):
    dis_params = {
        'input_shape': input_shape,
        'normalization': norm_dis,
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'drop_rate': drop_rate,
        'dropout': dropout,  # dropout: "normal dropout or spatial?
        'dilation': dilation}

    return dis_params


def _get_params(start_date: str = '2021-08-01', end_date: str = '2021-10-31', window_length: int=24,
                asset_ids: Optional[List] = None, window_overlap: int = 0,
                model_cls=WGAN, dis_cls=Critic, gen_cls=Generator, name: int = 1000,
                epochs: int = 1000, optimizer: str = 'Adam', sample_cycle: int = 10, batchsize: int = 64,
                lr: float = 0.0001, clip_val: float = 0.01, lambda_gp: float = 0.1, input_shape=(1, 24),
                noise: bool = False, n_critic: int = 5, norm_gen: str = 'batch', norm_dis: str = 'spectral',
                activ_fct: bool = None, kernel_gen: int = 3, kernel_dis: int = 3, stride_dis: int = 2,
                padding_dis: int = 1, dilation_dis: int = 1, dropout_dis: str = 'dropout',
                droprate_dis: float = 0.2
                ):
    """
    Parameters
    ----------
    model_cls :  optional
        class of used model. The default is GAN.
    name: int
        name under which model shall be saved. The default is 1000.
    epochs : int, optional
        number of training epochs. The default is 2000.
    optimizer : string, optional
        optimizer used for training. Options: {'Adam', 'RMSprop'}. Default is 'Adam'.
    batchsize : int, optional
        batch size of dataloader used for training. The default is 64.
    lr : float, optional
        learning rate. The default is 0.0001.
    input_shape : tuple, optional
        shape of input samples. The default is (1,24).
    noise : boolean, optional
        decides if normally distributed noise shall be added to the data. The default is False.
    norm_gen : string, optional
        normalization used in generator architecture. The default is 'batch'.
    norm_dis : string, optional
        normalization used in discriminator architecture. The default is 'spectral'.
    activ_fct : string or None, optional
        activation function used in generator. The default is None.
    kernel_gen : int, optional
        kernel size used in Convolutions in generator. The default is 3.
    kernel_dis : int, optional
        kernel size used in Convolutions in discriminator. The default is 3.

    Returns
    -------
    params : dictionary
        parameters for training, generator and discriminator.

    """
    params = {
        'train_params': _get_train_params(model_cls=model_cls, dis_cls=dis_cls, gen_cls=gen_cls,
                                          name=name, epochs=epochs, sample_cycle=sample_cycle,
                                          batchsize=batchsize, lr=lr, input_shape=input_shape,
                                          noise=noise, clip_val=clip_val, n_critic=n_critic,
                                          lambda_gp=lambda_gp, optimizer=optimizer),
        'dis_params': _get_discriminator_params(input_shape=input_shape, norm_dis=norm_dis,
                                                kernel_size=kernel_dis, stride=stride_dis,
                                                padding=padding_dis, dilation=dilation_dis,
                                                dropout=dropout_dis, drop_rate=droprate_dis
                                                ),
        'gen_params': _get_generator_params(norm_gen=norm_gen, input_shape=input_shape,
                                            activ_fct=activ_fct, kernel_size=kernel_gen),
        'data_params': _get_data_params(start_date=start_date, end_date=end_date,
                                        window_length=window_length, window_overlap=window_overlap,
                                        asset_ids=asset_ids)
    }

    params['train_params']['ivl'] = params['gen_params']['ivl']
    params['dis_params']['input_shape'] = params['train_params']['input_shape']

    return params
