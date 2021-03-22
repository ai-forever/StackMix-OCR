# -*- coding: utf-8 -*-
import os
import tarfile
import argparse

import gdown


DATASETS_ID = {
    'bentham': '19cFZkq4Fd0KHKxW9J0npTEhDkMyvpRAR',
    'peter': '1zjloEBbIBZkw4zrpYn3YrGuPHXep-xOk',
    'corpora': '1C9hSJ2R72dgZIdrwIRW0GGjVxiAlWo3G',
    'iam': '18EdxUOSUdVyXQoZxS-oHDbD5_pM7te-U',
    'iam_tbluche': '1rz-l__zjAQjiKDlipo08Zxat-FqPv5p8',
    'hkr': '18bS3xJinoKzVJc_p9bEWEXAwavy8WnEz',
    'saintgall': '1bJgrxZyjcdmYpnBD6z2CQoMue--DASJK',
    'washington': '1CUPav7swZfdqYnqfREgHEnzS8mo0I9wD',
    'schwerin': '1WRFsWopEbccbjUWJ4D7Cj9u4MtaHxHVE',
    'konzil': '16EGucsb41gErbAToW0Fv2WjN8pKqVbUt',
    'patzig': '1LnbXBhmwciGSMmRTBNWZwxom_5E9DTOz',
    'ricordi': '1bo6i2hs3rYiebjnVbdheadiPpipFYiIj',
    'schiller': '1xbf5HviEx8KXHd3s5qVSsRIHyHC4941B',
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
