import random
import os
import argparse
import json
from pathlib import Path

from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np

from .core import BaseModel, BaseDataset


def training_prologue():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', required=True, help='path of config file')
    args = parser.parse_args()

    # load config 
    with open(args.cfg_path) as fp:
        str_cfg = fp.read()
        config = json.loads(str_cfg)

    # set seed
    torch.manual_seed(config['trainer']['seed'])
    np.random.seed(config['trainer']['seed'])
    random.seed(config['trainer']['seed'])

    return config, parser

def make_output_dir(config, arg_parser):
    args = arg_parser.parse_args()
    # set output directory
    # check if output dir exists
    dir_idx = 0
    while True:
        try:
            path = Path('%s/output_%d' % (config['output_dir'], dir_idx))
            path.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError as err:
            dir_idx += 1
    config['output_dir'] = '%s/output_%d' % (config['output_dir'], dir_idx)
    # copy config file
    with open('%s/%s' % (config['output_dir'],
                         os.path.basename(args.cfg_path)), 'w') as fp:
        fp.write(json.dumps(config, indent=4, ensure_ascii=False))

class Trainer:
    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        self.dataset = dataset

    def init_process(self, rank, size, fn, backend='nccl'):
        cfg = self.config
        os.environ['MASTER_ADDR'] = cfg['trainer']['master_addr']
        os.environ['MASTER_PORT'] = cfg['trainer']['master_port']
        print('rank: %d, size: %d' % (rank,size))
        fn(rank, size)

    def fit(self):
        cfg = self.config
        if cfg['trainer']['n_gpus'] > 1:
            processes = list()
            # distribute
            for rank in range(cfg['trainer']['n_gpus']):
                p = mp.Process(target=self.init_process,
                               args=(rank,
                                     cfg['trainer']['n_gpus'],
                                     self._run))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        else:
            # single gpu or cpu training
            self._run()

    def average_gradients(self, model):
        world_size = dist.get_world_size()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.reduce_op.SUM)
                param.grad /= world_size

    def _run(self, rank=None, size=None):
        cfg = self.config
        # check if dist training
        dist_train = rank is not None and size is not None
        if dist_train:
            torch.cuda.set_device(rank)
            device = torch.device('cuda', rank)
            dist.init_process_group(backend='nccl',
                                    init_method='env://',
                                    world_size=size,
                                    rank=rank)
        else:
            if cfg['trainer']['n_gpus'] == 0:
                device = torch.device('cpu')
            else:
                # single gpu
                device = torch.device('cuda', 0)

        torch.manual_seed(cfg['trainer']['seed'])
        np.random.seed(cfg['trainer']['seed'])
        random.seed(cfg['trainer']['seed'])

        if dist_train:
            self.model.to(device)
            ddp_model = DDP(self.model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
            model = ddp_model.module
            train_dataset = self.dataset.dataset['train']
            train_sampler = DistributedSampler(train_dataset)
            train_data = DataLoader(train_dataset, sampler=train_sampler, batch_size=cfg['batch_size'])
        else:
            # get optimizer
            self.model.to(device)
            model = self.model
            train_data = self.dataset.train
        valid_data = self.dataset.valid

        optimizers, schedulers = self.model.configure_optimizers()

        # resume checkpoint
        cnt_epoch = 0
        if cfg['trainer']['resume_from_checkpoint']:
            ckpt = torch.load(cfg['trainer']['resume_from_checkpoint'], map_location=device)
            cnt_epoch = ckpt['epoch']
            model.load_state_dict(ckpt['state_dict'])
            for opt, opt_state, sch, sch_state in\
                    zip(optimizers, ckpt['optimizer_states'], schedulers, ckpt['lr_schedulers']):
                opt.load_state_dict(opt_state)
                if sch is not None:
                    sch.load_state_dict(sch_state)

        best_loss = np.inf
        for epoch in range(cnt_epoch, cfg['trainer']['max_epochs']):
            model.train()
            tqdm_ins = tqdm(train_data,
                            disable=(dist_train and rank != 0),
                            ascii=True,
                            desc='epoch: %d' % epoch)
            for batch_idx, batch in enumerate(tqdm_ins):
                for opt_idx, optimizer in zip(range(len(optimizers)), optimizers):
                    optimizer.zero_grad()
                    # training step
                    loss = model.training_step(batch, batch_idx, opt_idx)
                    tqdm_ins.set_postfix({'train_loss': '%7.4f' % loss})
                    loss.backward()
                    if dist_train:
                        self.average_gradients(model)

                    optimizer.step()

                    if dist_train:
                        dist.barrier()

            if schedulers is not None:
                for scheduler in schedulers:
                    scheduler.step()

            model.eval()
            if rank == 0 or not dist_train:
                # compute avg loss
                tot_loss = list()
                for batch_idx, batch in enumerate(tqdm(valid_data, disable=(rank != 0))):
                    loss = model.validation_step(batch, batch_idx)
                    tot_loss.append(loss.item())
                avg_loss = np.mean(np.array(tot_loss))

                # save best checkpoint
                if best_loss > avg_loss:
                    # save
                    save_dict = dict(
                            epoch=epoch,
                            optimizer_states=[optimizer.state_dict() for optimizer in optimizers],
                            state_dict = model.state_dict()
                            )
                    if schedulers is not None:
                        save_dict['lr_schedulers']=[scheduler.state_dict() for scheduler in schedulers]
                    model.save_user_specific_data(save_dict) 
                    torch.save(save_dict, '%s/epoch=%02d-val_loss=%8.6f.ckpt' % (self.config['output_dir'],
                                                                                 epoch, avg_loss))
                    best_loss = avg_loss

                print('%dth epoch, average validation loss: %7.4f, best_loss: %7.4f' %\
                        (epoch, avg_loss, best_loss))

            if dist_train:
                dist.barrier()
