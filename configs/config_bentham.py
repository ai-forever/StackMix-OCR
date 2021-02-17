# -*- coding: utf-8 -*-
import re

from .base import BaseConfig


class BenthamConfig(BaseConfig):

    def __init__(
            self,
            data_dir,
            image_w=2048,
            image_h=128,
            dataset_name='bentham',
            chars=' !"#&\'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXY[]_abcdefghijklmnopqrstuvwxyz|£§',
            corpus_name='jigsaw_corpus.txt',
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
    def clean_punct(text):
        text = re.sub(r'(\w)\s([,\.\!\:;\?])', '\g<1>\g<2>', text)  # noqa
        text = re.sub(r'(["\'\(\[<])\s(\w+)', '\g<1>\g<2>', text)  # noqa
        text = re.sub(r'(\w+)\s([\)\]>])', '\g<1>\g<2>', text)  # noqa
        return text

    def preprocess(self, text):
        """ preprocess only train text """
        text = text.strip()
        text = text.replace('é', 'e')
        text = text.replace('ê', 'e')
        text = text.replace('è', 'e')
        text = text.replace('à', 'a')
        text = text.replace('â', 'a')
        text = text.replace('⊥', '')
        text = self.clean_punct(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def postprocess(self, text):
        """ postprocess output text """
        text = text.strip()
        text = self.clean_punct(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
