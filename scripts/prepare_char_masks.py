# -*- coding: utf-8 -*-
import argparse
import sys
import json

import torch
from torch.utils.data import SequentialSampler
import pandas as pd

sys.path.insert(0, '.')

from configs import CONFIGS  # noqa
from src.dataset import DatasetRetriever  # noqa
from src.ctc_labeling import CTCLabeling  # noqa
from src.model import get_ocr_model  # noqa
from src import utils  # noqa
from src.predictor import Predictor  # noqa
from src.metrics import string_accuracy, cer, wer  # noqa
from src.char_masks import CharMasks  # noqa


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare char masks script.')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--image_w', type=int)
    parser.add_argument('--image_h', type=int)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--num_epochs', type=int, default=0)
    parser.add_argument('--experiment_description', type=str, default='Prepare char masks')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--char_masks_add', type=int, default=0)
    parser.add_argument('--char_masks_blank_add', type=int, default=0)

    args = parser.parse_args()

    assert args.dataset_name in CONFIGS
    config = CONFIGS[args.dataset_name](
        data_dir=args.data_dir,
        experiment_name=args.experiment_name,
        experiment_description=args.experiment_description,
        image_w=args.image_w,
        image_h=args.image_h,
        num_epochs=args.num_epochs,
        bs=args.bs,
        num_workers=args.num_workers,
    )

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('DEVICE:', device)
    print('DATASET:', args.dataset_name)

    ctc_labeling = CTCLabeling(config)

    df = pd.read_csv(f'{args.data_dir}/{args.dataset_name}/marking.csv', index_col='sample_id')

    train_dataset = DatasetRetriever(
        df[~df['stage'].isin(['valid', 'test'])],
        config,
        ctc_labeling,
    )

    model = get_ocr_model(config)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['bs'],
        sampler=SequentialSampler(train_dataset),
        pin_memory=False,
        drop_last=False,
        num_workers=config['num_workers'],
        collate_fn=utils.kw_collate_fn
    )

    predictor = Predictor(model, device)
    train_inference = predictor.run_inference(train_loader)

    all_masks, bad = CharMasks(
        config, ctc_labeling,
        add=args.char_masks_add,
        blank_add=args.char_masks_blank_add
    ).run(train_inference)

    with open(f'{config.data_dir}/{config.dataset_name}/all_char_masks.json', 'w') as file:
        json.dump(all_masks, file)

    print('GOOD MASKS:', len(all_masks))
    print('BAD MASKS:', len(bad))
