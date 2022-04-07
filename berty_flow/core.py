from abc import ABC, abstractmethod


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
    def eval_unseen_data(self, tot_res_dict):
        ...

    @abstractmethod
    def configure_optimizers(self):
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
