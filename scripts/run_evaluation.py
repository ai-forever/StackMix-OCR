# -*- coding: utf-8 -*-
import argparse
import sys
import re
from glob import glob
from collections import defaultdict

import torch
import pandas as pd
from torch.utils.data import SequentialSampler


sys.path.insert(0, '.')

from configs import CONFIGS  # noqa
from src.dataset import DatasetRetriever  # noqa
from src.ctc_labeling import CTCLabeling  # noqa
from src.model import get_ocr_model  # noqa
from src import utils  # noqa
from src.predictor import Predictor  # noqa
from src.metrics import string_accuracy, cer, wer  # noqa


if __name__ == '__main__':
    # example  python scripts/run_evaluation.py --experiment_folder ../StackMix-OCR-SAVED_MODELS/saintgall_base- --dataset_name saintgall --image_w 2048 --image_h 128  # noqa
    parser = argparse.ArgumentParser(description='Run train script.')
    parser.add_argument('--experiment_folder', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--image_w', type=int)
    parser.add_argument('--image_h', type=int)
    parser.add_argument('--data_dir', type=str, default='../StackMix-OCR-DATA')
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=6955)

    args = parser.parse_args()

    assert args.dataset_name in CONFIGS

    utils.seed_everything(args.seed)

    config = CONFIGS[args.dataset_name](
        data_dir=args.data_dir,
        image_w=args.image_w,
        image_h=args.image_h,
        bs=args.bs,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('DEVICE:', device)
    print('DATASET:', args.dataset_name)

    ctc_labeling = CTCLabeling(config)

    df = pd.read_csv(f'{args.data_dir}/{args.dataset_name}/marking.csv', index_col='sample_id')

    valid_dataset = DatasetRetriever(df[df['stage'] == 'valid'], config, ctc_labeling)
    test_dataset = DatasetRetriever(df[df['stage'] == 'test'], config, ctc_labeling)

    model = get_ocr_model(config, pretrained=False)
    model = model.to(device)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config['bs'],
        sampler=SequentialSampler(valid_dataset),
        pin_memory=False,
        drop_last=False,
        num_workers=config['num_workers'],
        collate_fn=utils.kw_collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['bs'],
        sampler=SequentialSampler(test_dataset),
        pin_memory=False,
        drop_last=False,
        num_workers=config['num_workers'],
        collate_fn=utils.kw_collate_fn
    )
    result_metrics = []
    for experiment_folder in glob(f'{args.experiment_folder}*'):
        print(experiment_folder)
        exp_metrics = defaultdict(list)
        for checkpoint_path in glob(f'{experiment_folder}/*.pt'):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            predictor = Predictor(model, device)
            valid_predictions = predictor.run_inference(valid_loader)
            test_predictions = predictor.run_inference(test_loader)

            df_valid_pred = pd.DataFrame([{
                'id': prediction['id'],
                'pred_text': ctc_labeling.decode(prediction['raw_output'].argmax(1)),
                'gt_text': prediction['gt_text']
            } for prediction in valid_predictions]).set_index('id')
            df_test_pred = pd.DataFrame([{
                'id': prediction['id'],
                'pred_text': ctc_labeling.decode(prediction['raw_output'].argmax(1)),
                'gt_text': prediction['gt_text']
            } for prediction in test_predictions]).set_index('id')

            exp_metrics['cer_valid'].append(round(cer(df_valid_pred['pred_text'], df_valid_pred['gt_text']), 5))
            exp_metrics['wer_valid'].append(round(wer(df_valid_pred['pred_text'], df_valid_pred['gt_text']), 5))
            exp_metrics['acc_valid'].append(round(
                string_accuracy(df_valid_pred['pred_text'], df_valid_pred['gt_text']), 5))

            exp_metrics['cer_test'].append(round(cer(df_test_pred['pred_text'], df_test_pred['gt_text']), 5))
            exp_metrics['wer_test'].append(round(wer(df_test_pred['pred_text'], df_test_pred['gt_text']), 5))
            exp_metrics['acc_test'].append(round(
                string_accuracy(df_test_pred['pred_text'], df_test_pred['gt_text']), 5))

            def as_asr_text(text):
                """ lowercase, no punct """
                text = re.sub(r'[^\w\s]', '', text)
                return text.lower().strip()

            df_valid_pred['pred_text'] = df_valid_pred['pred_text'].apply(as_asr_text)
            df_valid_pred['gt_text'] = df_valid_pred['gt_text'].apply(as_asr_text)
            df_test_pred['pred_text'] = df_test_pred['pred_text'].apply(as_asr_text)
            df_test_pred['gt_text'] = df_test_pred['gt_text'].apply(as_asr_text)

            exp_metrics['cer_valid_asr'].append(round(cer(df_valid_pred['pred_text'], df_valid_pred['gt_text']), 5))
            exp_metrics['wer_valid_asr'].append(round(wer(df_valid_pred['pred_text'], df_valid_pred['gt_text']), 5))
            exp_metrics['acc_valid_asr'].append(round(
                string_accuracy(df_valid_pred['pred_text'], df_valid_pred['gt_text']), 5))

            exp_metrics['cer_test_asr'].append(round(cer(df_test_pred['pred_text'], df_test_pred['gt_text']), 5))
            exp_metrics['wer_test_asr'].append(round(wer(df_test_pred['pred_text'], df_test_pred['gt_text']), 5))
            exp_metrics['acc_test_asr'].append(round(
                string_accuracy(df_test_pred['pred_text'], df_test_pred['gt_text']), 5))

        result_metrics.append({
            'cer_valid': min(exp_metrics['cer_valid']),
            'wer_valid': min(exp_metrics['wer_valid']),
            'acc_valid': max(exp_metrics['acc_valid']),
            'cer_test': min(exp_metrics['cer_test']),
            'wer_test': min(exp_metrics['wer_test']),
            'acc_test': max(exp_metrics['acc_test']),
            'cer_valid_asr': min(exp_metrics['cer_valid_asr']),
            'wer_valid_asr': min(exp_metrics['wer_valid_asr']),
            'acc_valid_asr': max(exp_metrics['acc_valid_asr']),
            'cer_test_asr': min(exp_metrics['cer_test_asr']),
            'wer_test_asr': min(exp_metrics['wer_test_asr']),
            'acc_test_asr': max(exp_metrics['acc_test_asr']),
        })

        print('---- VALID ----')
        print('CER:', min(exp_metrics['cer_valid']))
        print('WER:', min(exp_metrics['wer_valid']))
        print('ACC:', max(exp_metrics['acc_valid']))
        print('---- TEST -----')
        print('CER:', min(exp_metrics['cer_test']))
        print('WER:', min(exp_metrics['wer_test']))
        print('ACC:', max(exp_metrics['acc_test']))
        print('---- VALID as ASR ----')
        print('CER:', min(exp_metrics['cer_valid_asr']))
        print('WER:', min(exp_metrics['wer_valid_asr']))
        print('ACC:', max(exp_metrics['acc_valid_asr']))
        print('---- TEST as ASR -----')
        print('CER:', min(exp_metrics['cer_test_asr']))
        print('WER:', min(exp_metrics['wer_test_asr']))
        print('ACC:', max(exp_metrics['acc_test_asr']))

    result_metrics = pd.DataFrame(result_metrics)

    def mean_std(key, ndigits=4):
        mean = round(result_metrics[key].mean(), ndigits=ndigits)
        std = round(result_metrics[key].std(), ndigits=ndigits)
        return f'{mean} ± {std} [{round(std / mean * 100, 1)}%]'

    print('---- ----- ----')
    print('---- VALID ----')
    print('CER:', mean_std('cer_valid'))
    print('WER:', mean_std('wer_valid'))
    print('ACC:', mean_std('acc_valid'))
    print('---- TEST -----')
    print('CER:', mean_std('cer_test'))
    print('WER:', mean_std('wer_test'))
    print('ACC:', mean_std('acc_test'))
    print('---- VALID as ASR ----')
    print('CER:', mean_std('cer_valid_asr'))
    print('WER:', mean_std('wer_valid_asr'))
    print('ACC:', mean_std('acc_valid_asr'))
    print('---- TEST as ASR -----')
    print('CER:', mean_std('cer_test_asr'))
    print('WER:', mean_std('wer_test_asr'))
    print('ACC:', mean_std('acc_test_asr'))
