# -*- coding: utf-8 -*-
import random
import os

import numpy as np
import cv2
import torch
from torch.nn.utils.rnn import pad_sequence


def kw_collate_fn(batch):
    """ key-word collate_fn """
    result = {}
    paddings = {}
    for key, value in batch[0].items():
        result[key] = []
        paddings[key] = isinstance(value, torch.Tensor)

    for i, sample in enumerate(batch):
        for key, value in sample.items():
            result[key].append(value)

    lengths = {}
    for key, values in result.items():
        if paddings[key]:
            result[key] = pad_sequence(values, batch_first=True)
            lengths[f'{key}_length'] = torch.tensor(
                [value.shape[0] for value in values])
    result.update(lengths)
    return result


def resize_if_need(image, max_h, max_w):
    img = image.copy()
    img_h, img_w, img_c = img.shape
    coef = 1 if img_h <= max_h and img_w <= max_w else max(img_h / max_h, img_w / max_w)
    h = int(img_h / coef)
    w = int(img_w / coef)
    img = cv2.resize(img, (w, h))
    return img, coef


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


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
