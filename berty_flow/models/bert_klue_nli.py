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
                         max_length=self.bert_cfg.max_position_embeddings,
                         truncation=True)

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
        res_dict = {'loss': loss}
        return res_dict

    def validation_step(self, batch:dict, batch_idx:int):
        logits = self.forward(batch)
        loss = self.loss(logits, batch['label'])
        res_dict = {'loss': loss, 'gt':batch['label'], 'logits':logits}
        return res_dict

    def test_step(self, batch:dict, batch_idx:int):
        logits = self.forward(batch)
        loss = self.loss(logits, batch['label'])
        res_dict = {'loss': loss, 'gt':batch['label'], 'logits':logits}
        return res_dict

    def eval_unseen_data(self, tot_res_dict, epoch):
        tot_loss = 0.0
        tot_tp = 0
        tot_num_exs = 0
        for batch_res in tot_res_dict:
            batch_size = len(batch_res['gt'])
            tp = (torch.argmax(batch_res['logits'], dim=1) == batch_res['gt']).sum().item()

            tot_loss += batch_size * batch_res['loss'].item()
            tot_tp += tp
            tot_num_exs += batch_size
        avg_loss = tot_loss / tot_num_exs
        acc = tot_tp / tot_num_exs
        print('%dth epoch, average validation loss: %7.4f, accuracy: %7.4f' %\
                (epoch, avg_loss, acc))
        return avg_loss

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.config['trainer']['learning_rate'])
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 0.9)
        return [opt], [scheduler]



