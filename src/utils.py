# -*- coding: utf-8 -*-
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
