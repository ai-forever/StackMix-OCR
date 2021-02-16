# -*- coding: utf-8 -*-
from .config_bentham import BenthamConfig
from .config_peter import PeterConfig
from .config_iam import IAMConfig
from .config_hkr import HKRConfig
from .config_saintgall import SaintGallConfig

CONFIGS = {
    'bentham': BenthamConfig,
    'peter': PeterConfig,
    'iam': IAMConfig,
    'hkr': HKRConfig,
    'saintgall': SaintGallConfig,
}
