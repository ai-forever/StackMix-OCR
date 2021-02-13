# -*- coding: utf-8 -*-
import argparse
import sys
import time
from datetime import datetime

import neptune
import torch
import numpy as np
from torch.utils.data import SequentialSampler, RandomSampler
import pandas as pd

sys.path.insert(0, '.')

from configs import CONFIGS  # noqa
from src.dataset import DatasetRetriever  # noqa
from src.ctc_labeling import CTCLabeling  # noqa
from src.model import get_ocr_model  # noqa
from src.experiment import OCRExperiment  # noqa
from src import utils  # noqa
from src.predictor import Predictor  # noqa
from src.blot import get_train_transforms  # noqa
from src.metrics import string_accuracy, cer, wer  # noqa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--use_blot', type=int)
    parser.add_argument('--neptune_project', type=str)
    parser.add_argument('--neptune_token', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--experiment_description', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--image_w', type=int)
    parser.add_argument('--image_h', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--seed', type=int, default=6955)
    parser.add_argument('--use_progress_bar', type=int, default=0)

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
        df[df['stage'] == 'train'],
        config,
        ctc_labeling,
        transforms=get_train_transforms(config),
    )
    valid_dataset = DatasetRetriever(df[df['stage'] == 'valid'], config, ctc_labeling)
    test_dataset = DatasetRetriever(df[df['stage'] == 'test'], config, ctc_labeling)

    model = get_ocr_model(config)

    model = model.to(device)
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **config['optimizer']['params'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['bs'],
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=config['num_workers'],
        collate_fn=utils.kw_collate_fn
    )
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
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader),
        **config['scheduler']['params'],
    )

    neptune_kwargs = {}
    if args.neptune_project:
        neptune.init(
            project_qualified_name=args.neptune_project,
            api_token=args.neptune_token,
        )
        neptune_kwargs = dict(
            neptune=neptune,
            neptune_params={
                'description': config['experiment_description'],
                'params': config.params,
            }
        )

    if not args.checkpoint_path:
        experiment = OCRExperiment(
            experiment_name=config['experiment_name'],
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            base_dir=args.output_dir,
            best_saving={'cer': 'min', 'wer': 'min', 'acc': 'max'},
            last_saving=True,
            low_memory=True,
            verbose_step=10**5,
            seed=args.seed,
            use_progress_bar=bool(args.use_progress_bar),
            **neptune_kwargs,
            ctc_labeling=ctc_labeling,
        )
        experiment.fit(train_loader, valid_loader, config['num_epochs'])
    else:
        print('RESUMED FROM:', args.checkpoint_path)
        experiment = OCRExperiment.resume(
            checkpoint_path=args.checkpoint_path,
            train_loader=train_loader,
            valid_loader=valid_loader,
            n_epochs=config['num_epochs'],
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            seed=round(datetime.utcnow().timestamp()) % 10000,  # warning! in resume need change seed
            neptune=neptune_kwargs.get('neptune'),
            ctc_labeling=ctc_labeling,
        )

    time_inference = []
    for best_metric in ['best_cer', 'best_wer', 'best_acc', 'last']:
        experiment.load(f'{experiment.experiment_dir}/{best_metric}.pt')
        experiment.model.eval()
        predictor = Predictor(experiment.model, device)
        time_a = time.time()
        predictions = predictor.run_inference(test_loader)
        time_b = time.time()
        time_inference.append(time_b - time_a)
        df_pred = pd.DataFrame([{
            'id': prediction['id'],
            'pred_text': ctc_labeling.decode(prediction['raw_output'].argmax(1)),
            'gt_text': prediction['gt_text']
        } for prediction in predictions]).set_index('id')

        cer_metric = round(cer(df_pred['pred_text'], df_pred['gt_text']), 5)
        wer_metric = round(wer(df_pred['pred_text'], df_pred['gt_text']), 5)
        acc_metric = round(string_accuracy(df_pred['pred_text'], df_pred['gt_text']), 5)

        experiment.neptune.log_metric(f'cer_test__{best_metric}', cer_metric)
        experiment.neptune.log_metric(f'wer_test__{best_metric}', wer_metric)
        experiment.neptune.log_metric(f'acc_test__{best_metric}', acc_metric)

        mistakes = df_pred[df_pred['pred_text'] != df_pred['gt_text']]
        experiment.neptune.log_metric(f'mistakes__{best_metric}', mistakes.shape[0])

        df_pred.to_csv(f'{experiment.experiment_dir}/pred__{best_metric}.csv')
        experiment.neptune.log_artifact(f'{experiment.experiment_dir}/pred__{best_metric}.csv')

        experiment._log(  # noqa
            f'Results for {best_metric}.pt.',
            cer=cer_metric,
            wer=wer_metric,
            acc=acc_metric,
            speed_inference=len(test_dataset) / (time_b - time_a),
        )

    experiment.neptune.log_metric('time_inference', np.mean(time_inference))
    experiment.neptune.log_metric('speed_inference', len(test_dataset) / np.mean(time_inference))  # sample / sec

    experiment.destroy()
