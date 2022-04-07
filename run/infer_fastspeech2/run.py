import argparse
import json
import os
import random
import math

import numpy as np
import torch

from pytorch_berty.models.fastspeech2 import FastSpeech2
from pytorch_berty.utils.korean import KOR_SYMBOLS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', required=True, help='path of config file')
    args = parser.parse_args()

    # load config 
    with open(args.cfg_path) as fp:
        str_cfg = fp.read()
        config = json.loads(str_cfg)

    fastspeech2_model = FastSpeech2(config=config, dataset=None, symbols=KOR_SYMBOLS)
    mel = fastspeech2_model.predict('안녕하세요')



if __name__ == '__main__':
    main()
