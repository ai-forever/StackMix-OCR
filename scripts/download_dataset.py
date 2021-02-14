# -*- coding: utf-8 -*-
import os
import tarfile
import argparse

import gdown


DATASETS_INFO = {
    'data_root': '../StackMix-OCR-DATA',
    'bentham': {
        'dataset_name': 'bentham',
        'archive_name': 'bentham.tar.gz',
        'file_id': '13S6lwxuFoM1vOBlofK8VUnQ3RKC2iFI0',
    }
}


def extract_archive(archive_path, extrach_path):
    if archive_path.endswith('tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar_file:
            tar_file.extractall(path=extrach_path)


def download_and_extract(info):
    archive_path = f'{DATASETS_INFO["data_root"]}/{info["archive_name"]}'
    if not os.path.exists(archive_path):
        gdown.download(f'https://drive.google.com/uc?id={info["file_id"]}', archive_path, quiet=False)

    if not os.path.exists(f'{DATASETS_INFO["data_root"]}/{info["dataset_name"]}'):
        extract_archive(archive_path, DATASETS_INFO['data_root'])


if __name__ == '__main__':
    os.makedirs(DATASETS_INFO['data_root'], exist_ok=True)

    parser = argparse.ArgumentParser(description='Download dataset script.')
    parser.add_argument('--name', type=str)
    args = parser.parse_args()

    download_and_extract(DATASETS_INFO[args.name])
