from contextlib import contextmanager
from typing import List, Optional

import numpy as np
import torch as th
from torch import nn


def load_fake_data(params, size: int = 1000, clip: bool = True):
    """
    Loads specified model, then samples from generator.
    Input:
        params: dictionary of parameters.
        size: number of samples to be generated.
    Returns:
        x_fake: generated torch.tensor of shape [size, feature_dim, seq_len].
    """
    model_cls = params['train_params']['model_cls']
    gen_cls = params['train_params']['gen_cls']
    dis_cls = params['train_params']['dis_cls']
    model = model_cls(params)
    model.build_model(gen_cls, dis_cls)
    [m.eval() for m in model.models.values()]  # evaluation mode
    model.load_model()
    x_fake = model.sample(size=size, clip=clip)
    return x_fake


def load_real_data(dataset, size: int = 1000):
    """
    Loads real data samples of training data.
    Input:
        params: dictionary of parameters (see utils.config)
        size: number of loaded samples.
        shuffle: whether data shall be shuffled before loading.
        dataset: default dataset is LondonDataset containing samples from the London Smart Meter Data.
    Returns:
        x_real: loaded torch.tensor of shape [size, feature_dim, seq_len].
    """
    idx = np.random.permutation(len(dataset))[:size]
    data = dataset.data
    x_real = data[idx]
    return x_real


class SamplePool:
    '''
    Pool of samples that can be querried with given probability.
    '''
    pool: List[th.Tensor]
    max_pool: int

    def __init__(self, max_pool=50, init_pool: Optional[List]=None) -> None:
        '''
        Args:
            max_pool (int): size of the pool
            init_pool (list, optional): Initial state of the pool
        '''
        self.max_pool = max_pool
        if init_pool != None:
            self.pool = init_pool
        else:
            self.pool = []

    def query(self, sample: th.Tensor, num_outputs: int=-1, batched=True) -> th.Tensor:
        ''' Querry the pool for samples. Returns samples based on whether the sample is 
            batched and the number of desired outputs
        Args:
            sample (torch.Tensor): sample to querry
            num_outputs (int): Desired number of outputs. -1 disables this option
            batched (boolean): whether the sample is batched. returns batch size number of 
                outputs. mutually exclusive to num_outputs
        Returns:
            sample querried from given sample and the current pool
        '''
        if num_outputs == -1 and not batched:
            raise RuntimeError('num_outputs must be specified if input is not batched')
        
        result = []
        if batched == True:
            # return same number as in batch
            for b in sample:
                result.append(
                    self._query(b)
                )
        else:
            # return specified number, this may load the same sample into the pool
            for i in num_outputs:
                result.append(
                    self._query(sample)
                )

        return th.stack(result).type_as(sample)


    def _query(self, sample):
        ''' Queries a single sample against the pool. Save it if pool is not full. Otherwise uses 
            sampling strategy.
        '''
        # fill pool if not yet full
        if len(self.pool) < self.max_pool:
            self.pool.append(sample)
            return sample
        # either return the sample itself or return saved sample and keep sample in pool
        decision = th.randint(2, (1,))
        if decision == 0:
            return sample
        else:
            idx = th.randint(self.max_pool, (1,))
            out = self.pool[idx].clone()
            self.pool[idx] = sample
            return out

@contextmanager
def module_freeze(arr: List[nn.Module]):
    ''' Freezes parameters of modules as a context manager, freeing them when context is over.
        Similar to th.no_grad() but directed at modules.
    '''
    try:
        for mod in arr:
            for param in mod.parameters():
                param.requires_grad = False
        yield
    finally:
        for mod in arr:
            for param in mod.parameters():
                param.requires_grad = True

def freeze_module(arr: List[nn.Module]):
    ''' Freeze parameters of given modules.'''
    for mod in arr:
        for param in mod.parameters():
            param.requires_grad = False

def unfreeze_module(arr: List[nn.Module]):
    ''' Unfreeze parameters of given modules.'''
    for mod in arr:
        for param in mod.parameters():
            param.requires_grad = True
