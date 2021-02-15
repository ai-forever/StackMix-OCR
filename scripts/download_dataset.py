# -*- coding: utf-8 -*-
import os
import tarfile
import argparse

import gdown


DATASETS_ID = {
    'bentham': '13S6lwxuFoM1vOBlofK8VUnQ3RKC2iFI0',
    'peter': '1DWQS-7RNJ1AM02lTDBAqM94tvuVRYFx_',
    'corpora': '1C9hSJ2R72dgZIdrwIRW0GGjVxiAlWo3G',
    'iam': '16yz_b-tfBXc3meaUorbeeIXn9SEQrwYY',
}


def extract_archive(archive_path, extract_path):
    if archive_path.endswith('tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar_file:
            tar_file.extractall(path=extract_path)


def download_and_extract(data_dir, dataset_name):
    archive_path = f'{data_dir}/{dataset_name}.tar.gz'
    if not os.path.exists(archive_path):
        gdown.download(f'https://drive.google.com/uc?id={DATASETS_ID[dataset_name]}', archive_path, quiet=False)

    if not os.path.exists(f'{data_dir}/{dataset_name}'):
        extract_archive(archive_path, data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download dataset script.')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--data_dir', type=str, default='../StackMix-OCR-DATA')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    download_and_extract(args.data_dir, args.dataset_name)
