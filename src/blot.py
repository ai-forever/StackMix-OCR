# -*- coding: utf-8 -*-
import albumentations as A
from augmixations import HandWrittenBlot


class AlbuHandWrittenBlot(A.DualTransform):
    def __init__(self, hwb, always_apply=False, p=0.5):
        super(AlbuHandWrittenBlot, self).__init__(always_apply, p)
        self.hwb = hwb

    def apply(self, image, **params):
        return self.hwb(image)


def get_blot_transforms(config):
    bp = config['blot']['params']
    return A.OneOf([
        AlbuHandWrittenBlot(HandWrittenBlot(
            {
                'x': (None, None),
                'y': (None, None),
                'h': (bp['min_h'], bp['max_h']),
                'w': (bp['min_w'], bp['max_w']),
            }, {
                'incline': (bp['min_shift'], bp['max_shift']),
                'intensivity': (0.75, 0.75),
                'transparency': (0.05, 0.4),
                'count': i,
            }), p=1) for i in range(1, 11)
    ], p=0.5)
