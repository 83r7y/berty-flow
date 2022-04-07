import argparse
import json
import os
import random
import math

import numpy as np
import torch

from pytorch_berty.models.conv_face_detection import ConvFaceDetection
from pytorch_berty.models.conv_wav2lip import ConvWav2Lip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', required=True, help='path of config file')
    args = parser.parse_args()

    # load config 
    with open(args.cfg_path) as fp:
        str_cfg = fp.read()
        config = json.loads(str_cfg)

    # init face detection model
    # load model of https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
    face_detection_model = ConvFaceDetection(config, None, config['face_detection_ckpt']).to(config['device'])
    wav2lip_model = ConvWav2Lip(config, None, face_detection_model, config['wav2lip_ckpt']).to(config['device'])

    video_bin = wav2lip_model.predict(config['video_path'], config['audio_path'])

    with open(config['output_path'], 'wb') as fp:
        fp.write(video_bin)


if __name__ == '__main__':
    main()
