import csv

import numpy as np
import datasets
from torch.utils.data import DataLoader

from ..core import BaseDataset


class KlueNli(BaseDataset):
    def __init__(self, config):
        self.config = config
        self.load_data()
        self.serializer()
        self.make_dataloaders()

    def load_data(self):
        self.klue_nli_ds = datasets.load_dataset('klue', 'nli')

    def serializer(self):
        self.dataset = {'train': self.klue_nli_ds['train'],
                        'valid': self.klue_nli_ds['validation']}

    def make_dataloaders(self):
        self.train = DataLoader(self.dataset['train'],
                                batch_size=self.config['batch_size'],
                                shuffle=True, num_workers=4)
        self.valid = DataLoader(self.dataset['valid'],
                                batch_size=self.config['batch_size'],
                                num_workers=4)

