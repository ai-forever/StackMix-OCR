# -*- coding: utf-8 -*-
import re

from .base import BaseConfig


class PeterConfig(BaseConfig):

    def __init__(
            self,
            data_dir,
            image_w=2048,
            image_h=128,
            dataset_name='peter',
            chars=' #()+0123456789[]bdfghilmnrstwабвгдежзийклмнопрстуфхцчшщъыьэюяѣ⊕⊗',
            corpus_name='old_russian.txt',
            blank='ß',
            **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            dataset_name=dataset_name,
            image_w=image_w,
            image_h=image_h,
            chars=chars,
            blank=blank,
            corpus_name=corpus_name,
            **kwargs,
        )

    @staticmethod
    def preprocess(text):
        """ Метод чистки текста перед self.encode  """
        eng2rus = {
            'o': 'о',
            'a': 'а',
            'c': 'с',
            'e': 'е',
            'p': 'р',
            '×': 'х',
            '/': '',
            '…': '',
            '|': '',
            '–': '',
            'ǂ': '',
            'u': 'и',
            'k': 'к',
            'і': 'i',
        }
        text = text.strip()
        text = ''.join([eng2rus.get(char, char) for char in text])
        text = re.sub(r'\b[pр]s\b', 'р s', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def postprocess(text):
        """ Метод чистки текста после self.decode  """
        text = text.strip()
        text = text.replace('і', 'i')
        text = text.replace('рit', 'pit')
        text = text.replace('pitеr', 'piter')
        text = text.replace('irе', 'ire')
        text = text.replace('hеr', 'her')
        text = text.replace('mоn', 'mon')
        text = text.replace('siе', 'sie')
        text = text.replace('иr', 'ur')
        text = re.sub(r'точки а\b', 'точки a', text)
        text = re.sub(r'точки е\b', 'точки e', text)
        text = re.sub(r'точки с\b', 'точки c', text)
        text = re.sub(r'точка а\b', 'точка a', text)
        text = re.sub(r'точка е\b', 'точка e', text)
        text = re.sub(r'точка с\b', 'точка c', text)
        text = re.sub(r'разстоянием а\b', 'разстоянием a', text)
        text = re.sub(r'разстоянием е\b', 'разстоянием e', text)
        text = re.sub(r'разстоянием с\b', 'разстоянием c', text)
        text = re.sub(r'разстояние а\b', 'разстояние a', text)
        text = re.sub(r'разстояние е\b', 'разстояние e', text)
        text = re.sub(r'разстояние с\b', 'разстояние c', text)
        text = re.sub(r'линѣи а\b', 'линѣи a', text)
        text = re.sub(r'линѣи е\b', 'линѣи e', text)
        text = re.sub(r'линѣи с\b', 'линѣи c', text)
        text = re.sub(r'линѣi а\b', 'линѣi a', text)
        text = re.sub(r'линѣi е\b', 'линѣi e', text)
        text = re.sub(r'линѣi с\b', 'линѣi c', text)
        text = re.sub(r'линѣя а\b', 'линѣя a', text)
        text = re.sub(r'линѣя е\b', 'линѣя e', text)
        text = re.sub(r'линѣя с\b', 'линѣя c', text)
        text = re.sub(r'линiи а\b', 'линiи a', text)
        text = re.sub(r'линiи е\b', 'линiи e', text)
        text = re.sub(r'линiи с\b', 'линiи c', text)
        text = re.sub(r'линii а\b', 'линii a', text)
        text = re.sub(r'линii е\b', 'линii e', text)
        text = re.sub(r'линii с\b', 'линii c', text)
        text = re.sub(r'линiя а\b', 'линiя a', text)
        text = re.sub(r'линiя е\b', 'линiя e', text)
        text = re.sub(r'линiя с\b', 'линiя c', text)
        text = text.replace('р s', 'p s')
        text = text.replace('рs', 'p s')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
