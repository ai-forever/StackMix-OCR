# -*- coding: utf-8 -*-
from .bentham import BenthamConfig
from .peter import PeterConfig
from .iam import IAMConfig
from .hkr import HKRConfig
from .saintgall import SaintGallConfig

CONFIGS = {
    'bentham': BenthamConfig,
    'peter': PeterConfig,
    'iam': IAMConfig,
    'hkr': HKRConfig,
    'saintgall': SaintGallConfig,
}
