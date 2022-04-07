import argparse
import json
import os
import random

import numpy as np
import torch

from pytorch_berty.trainer import training_prologue, make_output_dir, Trainer
from pytorch_berty.datasets.klue_nli import KlueNli
from pytorch_berty.models.bert_klue_nli import BertKlueNli


def main():
    config, arg_parser = training_prologue()

    # make dataset instance
    dataset = KlueNli(config)

    # make model instance
    model = BertKlueNli(config, dataset)

    # make trainer
    trainer = Trainer(config, model, dataset)

    make_output_dir(config, arg_parser)

    # training
    trainer.fit()


if __name__ == '__main__':
    main()
