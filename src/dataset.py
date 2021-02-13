# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


def resize_if_need(image, max_h, max_w):
    img = image.copy()
    img_h, img_w, img_c = img.shape
    coef = 1 if img_h <= max_h and img_w <= max_w else max(
        img_h / max_h, img_w / max_w)
    h = int(img_h / coef)
    w = int(img_w / coef)
    img = cv2.resize(img, (w, h))
    return img


def make_img_padding(image, max_h, max_w):
    img = image.copy()
    img_h, img_w, img_c = img.shape
    bg = np.zeros((max_h, max_w, img_c), dtype=np.uint8)
    x1 = 0
    y1 = (max_h - img_h) // 2
    x2 = x1 + img_w
    y2 = y1 + img_h
    bg[y1:y2, x1:x2, :] = img.copy()
    return bg


class DatasetRetriever(Dataset):

    def __init__(self, df, config, ctc_labeling, transforms=None):
        self.config = config
        self.ctc_labeling = ctc_labeling
        self.image_ids = df.index.values
        self.texts = df['text'].values
        self.paths = df['path'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image = cv2.imread(f'{self.config.data_dir}/{self.paths[idx]}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_if_need(
            image, self.config['image_h'], self.config['image_w'])
        image = make_img_padding(
            image, self.config['image_h'], self.config['image_w'])

        text = self.texts[idx]

        encoded = self.ctc_labeling.encode(text)
        if self.transforms:
            image = self.transforms(image=image)['image']

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        return {
            'id': image_id,
            'image': image,
            'text': text,
            'encoded': torch.tensor(encoded, dtype=torch.int32),
        }
