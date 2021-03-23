# -*- coding: utf-8 -*-
import random

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from .utils import resize_if_need, make_img_padding


class DatasetRetriever(Dataset):

    def __init__(self, df, config, ctc_labeling, transforms=None, stackmix=None):
        self.config = config
        self.ctc_labeling = ctc_labeling
        self.image_ids = df.index.values
        self.texts = df['text'].values
        self.paths = df['path'].values
        self.transforms = transforms
        self.stackmix = stackmix

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        if self.stackmix and random.random() < 0.8:
            for _ in range(10):
                gt_text, image = self.stackmix.run_corpus_stackmix()
                if image is not None:
                    image, coef = self.resize_image(image)
                    break
            else:
                image, coef = self.load_image(idx)
        else:
            image, coef = self.load_image(idx)
            gt_text = self.texts[idx]

        encoded = self.ctc_labeling.encode(gt_text)
        if self.transforms:
            image = self.transforms(image=image)['image']

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        return {
            'id': image_id,
            'image': image,
            'gt_text': gt_text,
            'coef': coef,
            'encoded': torch.tensor(encoded, dtype=torch.int32),
        }

    def load_image(self, idx):
        image = cv2.imread(f'{self.config.data_dir}/{self.paths[idx]}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.resize_image(image)

    def resize_image(self, image):
        image, coef = resize_if_need(image, self.config['image_h'], self.config['image_w'])
        image = make_img_padding(image, self.config['image_h'], self.config['image_w'])
        return image, coef
