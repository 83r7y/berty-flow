import numpy as np
import torch
from torch import optim
from transformers import BertTokenizer, BertModel, BertConfig

from ..core import BaseModel


class BertKlueNli(torch.nn.Module, BaseModel):
    def __init__(self, config, dataset):
        super(BertKlueNli, self).__init__()

        self.config = config
        self.dataset = dataset
        self.num_classes = len(set(self.dataset.klue_nli_ds['train']['label']))

        self.bert_cfg = BertConfig.from_pretrained('klue/bert-base')
        self.bert = BertModel.from_pretrained('klue/bert-base')
        self.tk = BertTokenizer.from_pretrained('klue/bert-base', config=self.bert_cfg)

        self.logit_fc = torch.nn.Linear(self.bert_cfg.hidden_size, self.num_classes)
        self.loss = torch.nn.CrossEntropyLoss()

        if 'model_ckpt' in self.config:
            ckpt = torch.load(self.config['model_ckpt'])
            self.load_state_dict(ckpt['state_dict'])
            self.eval()

    def forward(self, batch:dict):
        batch = self.preprocess(batch)
        bert_out = self.bert(input_ids=batch['input_ids'],
                             token_type_ids=batch['token_type_ids'],
                             attention_mask=batch['attention_mask'])
        pooled_out = bert_out['pooler_output']
        logits = self.logit_fc(pooled_out)
        return logits

    def preprocess(self, batch:dict):
        # get device of model
        device = next(self.parameters()).device

        if 'label' in batch:
            batch['label'] = batch['label'].to(device)

        # tokenize sentences
        tk_res = self.tk(text=batch['premise'], text_pair=batch['hypothesis'],
                         padding=True, return_tensors='pt',
                         max_length=self.bert_cfg.max_position_embeddings)

        for key in tk_res:
            batch[key] = tk_res[key].to(device)
        return batch

    def predict(self, summary:str, full:str):
        # make text as batch
        batch = dict(summary=[summary],
                     full=[full])
        logits = self.forward(batch)
        # multilabel intent classification
        return logits

    def training_step(self, batch:dict, batch_idx:int, opt_idx:int):
        logits = self.forward(batch)
        loss = self.loss(logits, batch['label'])
        return loss

    def validation_step(self, batch:dict, batch_idx:int):
        logits = self.forward(batch)
        loss = self.loss(logits, batch['label'])
        return loss

    def test_step(self, batch:dict, batch_idx:int):
        logits = self.forward(batch)
        loss = self.loss(logits, batch['label'])
        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.config['trainer']['learning_rate'])
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 0.9)
        return [opt], [scheduler]



