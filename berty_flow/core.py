from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def __init__(self, config, dataset):
        ...

    @abstractmethod
    def preprocess(self, batch:dict):
        ...

    @abstractmethod
    def predict(self):
        ...

    @abstractmethod
    def training_step(self, batch, batch_idx, opt_idx):
        ...

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        ...

    @abstractmethod
    def test_step(self, batch, batch_idx):
        ...

    @abstractmethod
    def configure_optimizers(self):
        ...

    @abstractmethod
    def eval_unseen_data(self, tot_res_dict, epoch):
        # Print evaluation results on validation set
        # Average loss should be returned
        ...

    # TODO need some modification
    def save_user_specific_data(self, save_dict):
        ...


class BaseDataset(ABC):
    @abstractmethod
    def __init__(self, config):
        ...

    @abstractmethod
    def load_data(self, path:str):
        ...

    @abstractmethod
    def serializer(self):
        ...

    @abstractmethod
    def make_dataloaders(self):
        ...
