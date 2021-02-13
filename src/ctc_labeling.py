# -*- coding: utf-8 -*-
import re


class CTCLabeling:

    def __init__(self, config):
        self.config = config
        self.blank = config['blank']
        self.chars = [self.blank] + sorted(list(config['chars']))
        self.char2ind = {c: i for i, c in enumerate(self.chars)}

    def encode(self, text):
        text = self.preprocess(text)
        return [self.char2ind[char] for char in text]

    def decode(self, indexes):
        chars = []
        for i, index in enumerate(indexes):
            if index == self.padding_value:
                continue
            if i == 0:
                chars.append(self.chars[index])
                continue
            if indexes[i - 1] != index:
                chars.append(self.chars[index])
                continue
        text = ''.join(chars).strip()
        text = self.postprocess(text)
        return text

    def preprocess(self, text):
        """ Метод чистки текста перед self.encode  """
        text = self.config.preprocess(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def postprocess(self, text):
        text = self.config.postprocess(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @property
    def padding_value(self):
        return self.char2ind[self.blank]

    def __len__(self):
        return len(self.chars)
