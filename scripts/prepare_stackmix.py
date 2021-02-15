# -*- coding: utf-8 -*-
import argparse
import sys

import pandas as pd

sys.path.insert(0, '.')

from configs import CONFIGS  # noqa
from src.stackmix import StackMix  # noqa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare StackMix script.')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--image_w', type=int)
    parser.add_argument('--image_h', type=int)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--mwe_tokens_dir', type=str)

    args = parser.parse_args()

    assert args.dataset_name in CONFIGS

    marking = pd.read_csv(f'{args.data_dir}/{args.dataset_name}/marking.csv', index_col='sample_id')
    stackmix = StackMix(
        mwe_tokens_dir=args.mwe_tokens_dir,
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        image_h=args.image_h
    )

    stackmix.prepare_stackmix_dir(marking=marking)
