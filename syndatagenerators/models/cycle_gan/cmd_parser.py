import argparse
from typing import Optional


'''
Very generic structure and utility for reading in cmd line arguments
'''
class Arguments:
    '''
    Holds arguments. Use as class in namespace:
    e.g.
    ```
    ns = Arguments()
    parser.parse_args(namespace=ns)
    ```
    '''
    name: str
    
    data_dir: Optional[str]
    batch_size: int
    num_workers: int
    log_dir: str
    epochs: int
    version: Optional[str]
    experiment: Optional[str]

    def __str__(self):
        arr = [
            f'Experiment: {self.experiment} in version {self.version}',
            f'running with data {self.data_dir}',
            'Hyperparameters:',
            f'epochs {self.epochs}',
            f'batch size {self.batch_size}',
        ]

        return "\n".join(arr)



def build_parser() -> argparse.ArgumentParser:
    '''build default parser.'''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='sets batch size of experiment.')
    parser.add_argument('-n', '--name', type=str, default='experiment', help='set name of experiment run.')     ## TODO: review if this is a good arg
    parser.add_argument('-w', '--num-workers', type=int, default=4, help='sets the number of workers for the data loaders.')
    parser.add_argument('-d', '--data-dir', default=None, help='sets the directory for the data. (Uses data module default if not set)')
    parser.add_argument('-l', '--log-dir', default='logs', help='sets the logging directory.')
    parser.add_argument('-e', '--epochs', default=200, type=int, help='total number of epochs to train.')
    parser.add_argument('-v', '--version', default=None, help='Version name to use in Tensorboard logger.')
    parser.add_argument('--experiment', default='default', help='Experiment name for Tensorboard logger.')

    return parser

def run_parser(parser: argparse.ArgumentParser, cls) -> Arguments:
    '''execute parser with given namespace'''
    return parser.parse_args(namespace=cls())

    