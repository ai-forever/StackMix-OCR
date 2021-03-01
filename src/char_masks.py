# -*- coding: utf-8 -*-
from copy import deepcopy

from tqdm import tqdm

from .metrics import cer


class CharMasks:

    def __init__(self, config, ctc_labeling, add=0, blank_add=0):
        self.ctc_labeling = ctc_labeling
        self.config = config
        self.time_feature_count = self.config['model']['params']['time_feature_count']
        self.image_w = self.config['image_w']
        self.add = add
        self.blank_add = blank_add

    def run(self, train_inference):
        bad = []
        all_masks = []
        for i, sample in tqdm(enumerate(train_inference), total=len(train_inference)):
            gt_text = self.ctc_labeling.preprocess(sample['gt_text'])
            raw_output = sample['raw_output']
            encoded_chars = raw_output.argmax(1).numpy()
            cer_value = cer([self.ctc_labeling.decode(encoded_chars)], [gt_text if gt_text else ' '])
            if cer_value == 0:
                try:
                    masks = self.get_masks(encoded_chars, gt_text, sample['coef'])
                except Exception as e:
                    print(f'Warning! Exception during get char masks: {e}, {type(e)}')
                    masks = []
                if not masks:
                    continue
                all_masks.append({
                    'id': sample['id'],
                    'mask': masks
                })
            else:
                bad.append(i)
        return all_masks, bad

    def decode_plus(self, output):
        chars, plus_chars = [], []
        last_output, last_index, last_count = None, None, 0
        for i, char_output in enumerate(output.softmax(1)):
            index = char_output.argmax()
            if last_index is None:
                last_index = index
                last_output = char_output
                last_count = 1
                continue

            if last_index == index:
                last_output += char_output
                last_count += 1
            else:
                probas, indexes = last_output.sort(descending=True)

                chars.append(self.ctc_labeling.chars[last_index])
                plus_chars.append([
                    [self.ctc_labeling.chars[indexes[0]], probas[0].cpu().item() / last_count],
                    [self.ctc_labeling.chars[indexes[1]], probas[1].cpu().item() / last_count],
                    [self.ctc_labeling.chars[indexes[2]], probas[2].cpu().item() / last_count],
                ])

                last_index = index
                last_output = char_output
                last_count = 1

        probas, indexes = last_output.sort(descending=True)

        chars.append(self.ctc_labeling.chars[last_index])
        plus_chars.append([
            [self.ctc_labeling.chars[indexes[0]], probas[0].cpu().item() / last_count],
            [self.ctc_labeling.chars[indexes[1]], probas[1].cpu().item() / last_count],
            [self.ctc_labeling.chars[indexes[2]], probas[2].cpu().item() / last_count],
        ])

        pred_text = ''.join(chars).strip().replace(self.ctc_labeling.blank, '')
        pred_text = self.ctc_labeling.postprocess(pred_text)
        return pred_text, plus_chars

    @staticmethod
    def get_coords(encoded_chars_raw, mode='left'):
        encoded_chars = deepcopy(encoded_chars_raw)
        if mode == 'right':
            encoded_chars = encoded_chars[::-1]

        coords = []
        last_end = 0
        for i, (encoded_char) in enumerate(encoded_chars):
            if encoded_char == 0:
                continue

            if encoded_char == encoded_chars[i - 1]:
                last_end += 1
                continue

            coord = (last_end + i - 1) / 2
            if mode == 'right':
                coord = 256 - coord

            coords.append(coord)
            last_end = i + 1

        if mode == 'right':
            return coords[::-1]
        return coords

    def get_masks(self, encoded_chars, text, coef):
        left_coords = self.get_coords(encoded_chars, 'left')
        right_coords = self.get_coords(encoded_chars, 'right')
        coords = [left_coords[0]]
        for left_coord, right_coord, char in zip(left_coords[1:], right_coords[:-1], text):
            add = self.add
            if char == ' ':
                add = self.blank_add
            coords.append((left_coord + right_coord + add) / 2)
        coords.append(right_coords[-1])
        masks = []
        if len(text) != len(coords) - 1:
            return masks
        for i, (char, coord) in enumerate(zip(text, coords)):
            x1 = int(round(coord / self.time_feature_count * self.image_w * coef))
            x2 = int(round(coords[i + 1] / self.time_feature_count * self.image_w * coef))
            masks.append({
                'label': char,
                'x1': x1,
                'x2': x2,
            })
        return masks
