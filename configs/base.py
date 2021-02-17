# -*- coding: utf-8 -*-
class BaseConfig:

    def __init__(
            self,
            data_dir,
            dataset_name,
            image_w,
            image_h,
            chars,
            corpus_name,
            blank='ß',
            **kwargs,
    ):
        self.corpus_name = corpus_name
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.blank = blank
        self.chars = chars
        assert self.blank not in self.chars
        self.params = {
            'blank': self.blank,
            'chars': chars,
            'dataset_name': dataset_name,
            'image_w': image_w,
            'image_h': image_h,
            'optimizer': {
                'name': 'AdamW',
                'params': {
                    'lr': 0.0002,
                    'weight_decay': 1e-2,
                }
            },
            'scheduler': {
                'name': 'OneCycleLR',
                'params': {
                    'max_lr': 0.001,
                    'pct_start': 0.1,
                    'anneal_strategy': 'cos',
                    'final_div_factor': 10 ** 5,
                }
            },
            'model': {
                'name': 'resnet34',
                'params': {
                    'time_feature_count': 256,
                    'lstm_hidden': 256,
                    'lstm_len': 3,
                    'n_class': len(self.blank + self.chars),
                },
            },
            'blot': {
                'params': {
                    'min_h': 50,
                    'min_w': 10 * 2,
                    'max_h': 100,
                    'max_w': 50 * 2,
                    'min_shift': 10,
                    'max_shift': 50,
                }
            },
            **kwargs,
        }

    def preprocess(self, text):
        """ preprocess only train text """
        return text

    def postprocess(self, text):
        """ postprocess output text """
        return text

    def __getitem__(self, key):
        return self.params[key]
