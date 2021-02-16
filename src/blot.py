# -*- coding: utf-8 -*-
import albumentations as A
import numpy as np
import cv2
import bezier


class HandWrittenBlot(A.DualTransform):
    def __init__(self, min_h, min_w, max_h, max_w, min_shift, max_shift, count, always_apply=False, p=0.5):
        super(HandWrittenBlot, self).__init__(always_apply, p)
        self.min_h = min_h
        self.min_w = min_w
        self.max_h = max_h
        self.max_w = max_w
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.count = count

    def generate_points(self, mask_x, mask_y, mask_w, mask_h, intensivity, shift):
        points_count = int(intensivity * 20)

        point_prer_pixel = points_count / mask_h
        step_size = int(1 / point_prer_pixel) if 1 / \
            point_prer_pixel > 1 else 1

        lp_min, lp_max = int(mask_w * 0.01), int(mask_w * 0.20) + 1
        rp_min, rp_max = int(mask_w * 0.8), int(mask_w * 0.99) + 1

        points = [[], []]

        for i, step in enumerate(range(0, points_count * step_size, step_size)):

            if i < points_count // 10:
                x = np.random.randint(mask_x, mask_x + mask_w)
                y = np.random.randint(mask_y, mask_y + mask_h)
                points[0].append(x)
                points[1].append(y)
            else:
                l_att = np.random.randint(10, 80)
                r_att = 80 - l_att

                if i % 2 == 0:
                    for _ in range(l_att):
                        x = np.random.randint(lp_min, lp_max)
                        y = np.random.randint(step, step + step_size + shift)

                        points[0].append(x + mask_x)
                        points[1].append(y + mask_y)
                else:
                    for _ in range(r_att):
                        x = np.random.randint(rp_min, rp_max)
                        y = np.random.randint(step, step + step_size)

                        points[0].append(x + mask_x)
                        points[1].append(y + mask_y)

        return points

    def draw_bezier_curve(self, image, points):
        img = image.copy()

        curve = bezier.Curve(points, degree=len(points[0]) - 1)

        x, y = curve.evaluate(0.01)
        x1, y1 = int(x), int(y)

        for point in np.arange(0.01, 0.99, 0.01):
            x, y = curve.evaluate(point)
            x2, y2 = int(x), int(y)
            img = cv2.line(img, (x1, y1), (x2, y2),
                           (0, 0, 0), np.random.randint(1, 5))
            x1, y1 = x2, y2

        return img

    def make_handwriting(self, image, configs):
        bg_img = image.copy()
        fg_img = image.copy()

        for config in configs:
            for _ in range(config['repeat']):
                points = self.generate_points(
                    config['x'],
                    config['y'],
                    config['w'],
                    config['h'],
                    config['points_intensivity'],
                    config['shift'],
                )
                fg_img = self.draw_bezier_curve(fg_img, points)

            bg_img = cv2.addWeighted(
                bg_img, config['transparency'], fg_img, 1 - config['transparency'], 0)

        return bg_img

    def generate_configs(self, img_h, img_w):

        min_h = 0 if self.min_h is None or self.min_h < 0 else self.min_h
        min_w = 0 if self.min_w is None or self.min_w < 0 else self.min_w

        max_h = self.img_h if self.max_h is None or self.max_h > img_h else self.max_h
        max_w = self.img_w if self.max_w is None or self.max_w > img_w else self.max_w

        configs = []

        for _ in range(self.count):
            h = np.random.randint(min_h, max_h)
            w = np.random.randint(min_w, max_w)

            x = np.random.randint(0, img_w - w)
            y = np.random.randint(0, img_h - h)

            configs.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'points_intensivity': .75,
                'repeat': np.random.randint(1, 5),
                'transparency': np.random.randint(5, 40) / 100,
                'shift': np.random.randint(self.min_shift, self.max_shift),
            })

        return configs

    def apply(self, image, **params):
        img_h, img_w, _ = image.shape
        shades_configs = self.generate_configs(img_h, img_w)
        img = self.make_handwriting(image, shades_configs)
        return img

# TODO переделать через augmixations


def get_train_transforms(config):
    return A.OneOf([
        HandWrittenBlot(**config['blot']['params'], count=i, p=1.0) for i in range(1, 11)
    ], p=0.5)
