# -*- coding: utf-8 -*-
import shutil
import os
import json
import random
from glob import glob
from collections import defaultdict

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk import tokenize
from albumentations import DualTransform


class StackMix:

    def __init__(self, mwe_tokens_dir, data_dir, dataset_name, image_h, p_background_smoothing=0.1):
        self.image_h = image_h
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.mwe_tokens_dir = mwe_tokens_dir
        self.stackmix_dir = f'{self.mwe_tokens_dir}/{self.dataset_name}/{self.dataset_name}'
        self.max_token_size = 8  # TODO work with other tokens
        self.angle = 0  # TODO work with other angles
        #
        self.stackmix_data = None
        self.token2path = None
        self.path2leftx = None
        self.all_mwe_tokens = None
        self.corpus = []
        self.tokenizers = {}
        #
        self.background_smooth = BackgroundSmoothing(p=p_background_smoothing)

    def prepare_stackmix_dir(self, marking):
        train_ids = marking[~marking['stage'].isin(['valid', 'test'])].index.values
        all_masks = {}
        for masks in json.load(open(f'{self.data_dir}/{self.dataset_name}/all_char_masks.json', 'rb')):
            if masks['id'] not in train_ids:
                continue
            mask = []
            for _mask in masks['mask']:
                mask.append({
                    'label': _mask['label'],
                    'x1': _mask['x1'],
                    'x2': _mask['x2'],
                })
            all_masks[masks['id']] = mask

        try:
            self.rm_stackmix_dir()
        except:  # noqa
            pass
        os.makedirs(self.stackmix_dir, exist_ok=True)

        image_ids = [_id for _id, masks in tqdm(all_masks.items())]
        stackmix_ids = set()
        for ids in marking.loc[image_ids].reset_index()[
            ['sample_id', 'text']
        ].groupby('text').agg(lambda x: list(x)[:50])['sample_id'].values:
            stackmix_ids.update(ids)

        mwe_tokens_count = defaultdict(int)
        stackmix_data = []
        for _id, masks in tqdm(all_masks.items()):
            if _id not in stackmix_ids:
                continue
            image = self.load_image(marking.loc[_id]['path'])
            text = ''.join([mask['label'] for mask in masks])
            for n in range(1, self.max_token_size + 1):
                for i in range(len(text) - n):
                    span_a, span_b = i, i + n
                    cut_image, left_x = self.cut_symbol(image, masks[span_a]['x1'], masks[span_b - 1]['x2'], self.angle)
                    h, w, c = cut_image.shape
                    if h < 10 or w < 10:
                        continue

                    cut_image, coef = self.resize_to_h_if_need(cut_image, self.image_h)
                    cut_image = self.make_img_padding_to_h(cut_image, self.image_h)
                    cut_image = self.replace_center(cut_image)
                    left_x = int(round(left_x / coef))

                    mwe_token = text[i:i + n]
                    os.makedirs(f'{self.stackmix_dir}/{mwe_token}', exist_ok=True)

                    count = mwe_tokens_count[mwe_token]
                    if count > 100:
                        continue

                    path = f'{self.stackmix_dir}/{mwe_token}/{count}.png'
                    cv2.imwrite(path, cut_image)

                    mwe_tokens_count[mwe_token] += 1

                    stackmix_data.append({
                        'text': text[i:i + n],
                        'path': path,
                        'left_x': left_x,
                    })

        stackmix_data = pd.DataFrame(stackmix_data)
        stackmix_data.to_csv(f'{self.mwe_tokens_dir}/{self.dataset_name}/stackmix.csv', index=False)

        self.load()

    def run_corpus_stackmix(self, tokenizer=None):
        if not self.corpus:
            raise
        text = random.choice(self.corpus)
        image = self.run_stackmix(text, tokenizer=tokenizer)
        return text, image

    def run_stackmix(self, text, tokenizer=None):
        if tokenizer is None:
            tokenizer = random.choices(
                population=[
                    self.tokenizers[3],
                    self.tokenizers[4],
                    self.tokenizers[5],
                    self.tokenizers[6],
                    self.tokenizers[7],
                    self.tokenizers[8],
                ],
                weights=[0.05, 0.15, 0.20, 0.20, 0.20, 0.20],
                k=1
            )[0]

        image, left_x = None, None

        spans = self.get_spans(text, tokenizer)
        if not self.check_spans(spans):
            return

        for span in spans:
            word_token, (span_a, span_b), token = span
            if image is None:
                path = random.choice(self.token2path.loc[word_token]['path'])
                image = cv2.imread(path)
                left_x = self.path2leftx.loc[path]['left_x']  # noqa
            else:
                path = random.choice(self.token2path.loc[word_token]['path'])
                stack_image = cv2.imread(path)
                stack_left_x = self.path2leftx.loc[path]['left_x']
                image = self.stack_images(image, stack_image, self.angle, stack_left_x)
                left_x = stack_left_x  # noqa

        if image is not None:
            image = self.background_smooth(image=image)['image']
        return image

    def load(self):
        self.stackmix_data = pd.read_csv(f'{self.mwe_tokens_dir}/{self.dataset_name}/stackmix.csv')
        self.stackmix_data['text'] = self.stackmix_data['text'].astype(str)
        for max_len in range(3, self.max_token_size+1):
            mwes = sorted([mwe for mwe in self.stackmix_data['text'].unique() if len(mwe) <= max_len])
            self.tokenizers[max_len] = tokenize.MWETokenizer(mwes=mwes, separator='')

        self.token2path = self.stackmix_data[['text', 'path']].groupby('text').agg(list)
        self.path2leftx = self.stackmix_data[['path', 'left_x']].set_index('path')
        self.all_mwe_tokens = self.stackmix_data['text'].unique()

    def load_corpus(self, ctc_labeling, corpus_path):
        mwe_chars = ''.join(sorted([mwe for mwe in list(self.all_mwe_tokens) if len(mwe) <= 1]))
        lines = open(corpus_path, 'r').readlines()
        for line in tqdm(lines):
            line = line.strip()
            line = ctc_labeling.preprocess(line)
            if len(set(line).intersection(mwe_chars)) != len(set(line)) or len(line) > 110:
                continue
            self.corpus.append(line)

    def rm_stackmix_dir(self):
        for i in range(5):
            for path in glob(f'{self.stackmix_dir}/*'):
                try:
                    shutil.rmtree(path)
                except OSError:
                    for subpath in glob(f'{path}/*'):
                        shutil.rmtree(subpath)
        shutil.rmtree(f'{self.mwe_tokens_dir}/{self.dataset_name}')

    @staticmethod
    def cut_symbol(image, x1, x2, angle):
        img = image.copy()
        h, w, c = img.shape
        d_w = int(round(np.tan(angle * np.pi / 180) * (h // 2)))
        # TODO FIX WITH ANGLES!!!!
        # left_mask = np.array([[x1 - d_w, 0], [x1 - d_w, h], [x1 + d_w, 0]])
        # right_mask = np.array([[x2 + d_w, 0], [x2 + d_w, h], [x2 - d_w, h]])
        # cv2.fillPoly(img, pts =[left_mask, right_mask], color=(0, 0, 0))
        #####
        return img[:, max(x1 - d_w, 0): min(x2 + d_w, w), :], x1 - d_w

    @staticmethod
    def get_spans(text, tokenizer):
        word_tokens = tokenizer.tokenize(text)
        spans = []
        current_idx = 0
        for i, word_token in enumerate(word_tokens):
            span_a = current_idx
            current_idx += len(word_token)
            span_b = current_idx
            token = text[span_a:span_b]
            # current_idx += 1
            spans.append(
                (word_token, (span_a, span_b), token)
            )
        return spans

    @staticmethod
    def stack_images(img_1, img_2, angle, left_x):
        d_w = int(round(np.tan(angle * np.pi / 180) * img_1.shape[0])) + min(left_x, 0)
        result_w = img_1.shape[1] + img_2.shape[1] - 1
        if result_w - d_w > img_1.shape[1] and result_w - d_w > img_2.shape[1]:
            result_w -= d_w
        result_h = img_1.shape[0]
        result_img = np.ones((result_h, result_w, 3), dtype=np.uint16) * 255
        result_img[:, :img_1.shape[1], :] = img_1
        result_img[:, -img_2.shape[1]:, :] = img_2
        result_img = np.clip(result_img, 0, 255)
        result_img = result_img.astype(np.uint8)
        return result_img

    @staticmethod
    def replace_center(image):
        h, _, _ = image.shape
        mean_h = image.mean(axis=1).mean(axis=1)
        center = np.argwhere(mean_h < mean_h.min() * 1.25).mean()
        if str(center) == 'nan':
            center = h // 2
        else:
            center = int(center)
        empty_img = np.ones(image.shape, dtype=np.uint8) * 255
        empty_img[max(h//2 - center, 0):h//2] = image[max(center - h//2, 0):center]
        empty_img[h//2:h//2 + h - center] = image[center:min(center + h//2, h)]
        return empty_img

    @staticmethod
    def resize_to_h_if_need(image, max_h):
        img = image.copy()
        img_h, img_w, img_c = img.shape
        coef = 1 if img_h < max_h else img_h / max_h
        h = int(img_h / coef)
        w = int(img_w / coef)
        img = cv2.resize(img, (w, h))
        return img, coef

    @staticmethod
    def make_img_padding_to_h(image, max_h):
        img = image.copy()
        img_h, img_w, img_c = img.shape
        bg = np.ones((max_h, img_w, img_c), dtype=np.uint8) * 255
        y1 = (max_h - img_h) // 2
        y2 = y1 + img_h
        bg[y1:y2, :, :] = img.copy()
        return bg

    def load_image(self, path):
        image = cv2.imread(f'{self.data_dir}/{path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def check_spans(self, spans):
        for span in spans:
            word_token, (span_a, span_b), token = span
            if word_token not in self.all_mwe_tokens:
                print(f'WARNING! Used unknown token "{word_token}".')
                return False
        return True


class BackgroundSmoothing(DualTransform):
    def __init__(self, window_width=32, always_apply=False, p=0.5):
        super(BackgroundSmoothing, self).__init__(always_apply, p)
        self.window_width = window_width

    def apply(self, image, **params):
        img = self.smash_color(image)
        return img

    def calc_mask_mean(self, images, index, window):
        img = np.zeros(images[0].shape, dtype=np.float32)
        img_count = 0
        for i in range(index - window, index):
            img += images[i]
            img_count += 1
        img = img / img_count
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        return img

    def smash_color(self, image):
        out_img = image.copy()
        COLOR_MIN = np.array([2, 2, 160], np.uint8)
        COLOR_MAX = np.array([64, 255, 255], np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(out_img, COLOR_MIN, COLOR_MAX)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_HSV2RGB)
        bool_mask = mask.astype(bool)
        white_bg = np.ones(out_img.shape, dtype=np.uint8) * 255
        white_bg[bool_mask] = out_img[bool_mask]
        w = self.window_width
        steps = out_img.shape[1] // w
        micro_step = out_img.shape[1] % w
        img_parts = [white_bg[:, i * w:i * w + w] for i in range(steps)]
        background = np.zeros(out_img.shape, dtype=np.uint8)
        for i in range(steps):
            background[:, i * w:i * w + w] = self.calc_mask_mean(img_parts, i, steps)
        background[:, -1 * micro_step:, :] = white_bg[:, -1 * micro_step:, :]
        out_img[bool_mask] = background[bool_mask]
        return out_img
