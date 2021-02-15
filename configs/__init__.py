# -*- coding: utf-8 -*-
from .config_bentham import BenthamConfig
from .config_peter import PeterConfig
from .config_iam import IAMConfig


CONFIGS = {
    'bentham': BenthamConfig,
    'peter': PeterConfig,
    'iam': IAMConfig,
}
