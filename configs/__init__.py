# -*- coding: utf-8 -*-
from .bentham import BenthamConfig
from .peter import PeterConfig
from .iam import IAMConfig
from .iam_tbluche import TblucheIAMConfig
from .hkr import HKRConfig
from .saintgall import SaintGallConfig
from .washington import WashingtonConfig
from .schwerin import SchwerinConfig
from .konzil import KonzilConfig
from .patzig import PatzigConfig
from .ricordi import RicordiConfig
from .schiller import SchillerConfig

CONFIGS = {
    'bentham': BenthamConfig,
    'peter': PeterConfig,
    'iam': IAMConfig,
    'iam_tbluche': TblucheIAMConfig,
    'hkr': HKRConfig,
    'saintgall': SaintGallConfig,
    'washington': WashingtonConfig,
    'schwerin': SchwerinConfig,
    'konzil': KonzilConfig,
    'patzig': PatzigConfig,
    'ricordi': RicordiConfig,
    'schiller': SchillerConfig,
}
