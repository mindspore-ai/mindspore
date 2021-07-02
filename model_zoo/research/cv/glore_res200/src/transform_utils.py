# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
augment operation
"""
import math
import random
import re
import numpy as np
import PIL
from PIL import Image, ImageEnhance, ImageOps

_PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])
_FILL = (128, 128, 128)
_MAX_LEVEL = 10.
_HPARAMS_DEFAULT = dict(translate_const=250, img_mean=_FILL)
_RAND_TRANSFORMS = [
    'Distort',
    'Zoom',
    'Blur',
    'Skew',
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'PosterizeTpu',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
]
_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)
_RAND_CHOICE_WEIGHTS_0 = {
    'Rotate': 0.3,
    'ShearX': 0.2,
    'ShearY': 0.2,
    'TranslateXRel': 0.1,
    'TranslateYRel': 0.1,
    'Color': .025,
    'Sharpness': 0.025,
    'AutoContrast': 0.025,
    'Solarize': .005,
    'SolarizeAdd': .005,
    'Contrast': .005,
    'Brightness': .005,
    'Equalize': .005,
    'PosterizeTpu': 0,
    'Invert': 0,
    'Distort': 0,
    'Zoom': 0,
    'Blur': 0,
    'Skew': 0,
}


def _interpolation(kwargs):
    interpolation = kwargs.pop('resample', Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    return interpolation


def _check_args_tf(kwargs):
    if 'fillcolor' in kwargs and _PIL_VER < (5, 0):
        kwargs.pop('fillcolor')
    kwargs['resample'] = _interpolation(kwargs)

# define all kinds of functions


def _randomly_negate(v):
    return -v if random.random() > 0.5 else v


def shear_x(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def translate_x_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def rotate(img, degrees, **kwargs):
    """
    rotate operation
    """
    kwargs_new = kwargs
    kwargs_new.pop('resample')
    kwargs_new['resample'] = Image.BICUBIC
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs_new)
    if _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs_new)
    return img.rotate(degrees, resample=kwargs['resample'])


def auto_contrast(img, **__):
    return ImageOps.autocontrast(img)


def invert(img, **__):
    return ImageOps.invert(img)


def equalize(img, **__):
    return ImageOps.equalize(img)


def solarize(img, thresh, **__):
    return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    """
    add solarize operation
    """
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    return img


def posterize(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **__):
    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    return ImageEnhance.Sharpness(img).enhance(factor)


def _rotate_level_to_arg(level, _hparams):
    # range [-30, 30]
    level = (level / _MAX_LEVEL) * 30.
    level = _randomly_negate(level)
    return (level,)


def _enhance_level_to_arg(level, _hparams):
    # range [0.1, 1.9]
    return ((level / _MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level, _hparams):
    # range [-0.3, 0.3]
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate(level)
    return (level,)


def _translate_abs_level_to_arg(level, hparams):
    translate_const = hparams['translate_const']
    level = (level / _MAX_LEVEL) * float(translate_const)
    level = _randomly_negate(level)
    return (level,)


def _translate_rel_level_to_arg(level, _hparams):
    # range [-0.45, 0.45]
    level = (level / _MAX_LEVEL) * 0.45
    level = _randomly_negate(level)
    return (level,)


def _posterize_original_level_to_arg(level, _hparams):
    # As per original AutoAugment paper description
    # range [4, 8], 'keep 4 up to 8 MSB of image'
    return (int((level / _MAX_LEVEL) * 4) + 4,)


def _posterize_research_level_to_arg(level, _hparams):
    # As per Tensorflow models research and UDA impl
    # range [4, 0], 'keep 4 down to 0 MSB of original image'
    return (4 - int((level / _MAX_LEVEL) * 4),)


def _posterize_tpu_level_to_arg(level, _hparams):
    # As per Tensorflow TPU EfficientNet impl
    # range [0, 4], 'keep 0 up to 4 MSB of original image'
    return (int((level / _MAX_LEVEL) * 4),)


def _solarize_level_to_arg(level, _hparams):
    # range [0, 256]
    return (int((level / _MAX_LEVEL) * 256),)


def _solarize_add_level_to_arg(level, _hparams):
    # range [0, 110]
    return (int((level / _MAX_LEVEL) * 110),)


def _distort_level_to_arg(level, _hparams):
    return (int((level / _MAX_LEVEL) * 10 + 10),)


def _zoom_level_to_arg(level, _hparams):
    return ((level / _MAX_LEVEL) * 0.4,)


def _blur_level_to_arg(level, _hparams):
    level = (level / _MAX_LEVEL) * 0.5
    level = _randomly_negate(level)
    return (level,)


def _skew_level_to_arg(level, _hparams):
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate(level)
    return (level,)


def distort(img, v, **__):
    """
    distort operation
    """
    w, h = img.size
    horizontal_tiles = int(0.1 * v)
    vertical_tiles = int(0.1 * v)

    width_of_square = int(math.floor(w / float(horizontal_tiles)))
    height_of_square = int(math.floor(h / float(vertical_tiles)))
    width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
    height_of_last_square = h - (height_of_square * (vertical_tiles - 1))
    dimensions = []

    for vertical_tile in range(vertical_tiles):
        for horizontal_tile in range(horizontal_tiles):
            if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                   vertical_tile * height_of_square,
                                   width_of_last_square + (horizontal_tile * width_of_square),
                                   height_of_last_square + (height_of_square * vertical_tile)])
            elif vertical_tile == (vertical_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                   vertical_tile * height_of_square,
                                   width_of_square + (horizontal_tile * width_of_square),
                                   height_of_last_square + (height_of_square * vertical_tile)])
            elif horizontal_tile == (horizontal_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                   vertical_tile * height_of_square,
                                   width_of_last_square + (horizontal_tile * width_of_square),
                                   height_of_square + (height_of_square * vertical_tile)])
            else:
                dimensions.append([horizontal_tile * width_of_square,
                                   vertical_tile * height_of_square,
                                   width_of_square + (horizontal_tile * width_of_square),
                                   height_of_square + (height_of_square * vertical_tile)])
    last_column = []
    for i in range(vertical_tiles):
        last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

    last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

    polygons = []
    for x1, y1, x2, y2 in dimensions:
        polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

    polygon_indices = []
    for i in range((vertical_tiles * horizontal_tiles) - 1):
        if i not in last_row and i not in last_column:
            polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

    for a, b, c, d in polygon_indices:
        dx = v
        dy = v

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
        polygons[a] = [x1, y1,
                       x2, y2,
                       x3 + dx, y3 + dy,
                       x4, y4]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
        polygons[b] = [x1, y1,
                       x2 + dx, y2 + dy,
                       x3, y3,
                       x4, y4]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
        polygons[c] = [x1, y1,
                       x2, y2,
                       x3, y3,
                       x4 + dx, y4 + dy]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
        polygons[d] = [x1 + dx, y1 + dy,
                       x2, y2,
                       x3, y3,
                       x4, y4]

    generated_mesh = []
    for idx, i in enumerate(dimensions):
        generated_mesh.append([dimensions[idx], polygons[idx]])
    return img.transform(img.size, PIL.Image.MESH, generated_mesh, resample=PIL.Image.BICUBIC)


def zoom(img, v, **__):
    #assert 0.1 <= v <= 2
    w, h = img.size
    image_zoomed = img.resize((int(round(img.size[0] * v)),
                               int(round(img.size[1] * v))),
                              resample=PIL.Image.BICUBIC)
    w_zoomed, h_zoomed = image_zoomed.size

    return image_zoomed.crop((math.floor((float(w_zoomed) / 2) - (float(w) / 2)),
                              math.floor((float(h_zoomed) / 2) - (float(h) / 2)),
                              math.floor((float(w_zoomed) / 2) + (float(w) / 2)),
                              math.floor((float(h_zoomed) / 2) + (float(h) / 2))))


def erase(img, v, **__):
    """
    distort operation
    """
    #assert 0.1<= v <= 1
    w, h = img.size
    w_occlusion = int(w * v)
    h_occlusion = int(h * v)
    if len(img.getbands()) == 1:
        rectangle = PIL.Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion) * 255))
    else:
        rectangle = PIL.Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion, len(img.getbands())) * 255))

    random_position_x = random.randint(0, w - w_occlusion)
    random_position_y = random.randint(0, h - h_occlusion)
    img.paste(rectangle, (random_position_x, random_position_y))
    return img


def skew(img, v, **__):
    """
    skew operation
    """
    #assert -1 <= v <= 1
    w, h = img.size
    x1 = 0
    x2 = h
    y1 = 0
    y2 = w
    original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]
    max_skew_amount = max(w, h)
    max_skew_amount = int(math.ceil(max_skew_amount * v))
    skew_amount = max_skew_amount
    new_plane = [(y1 - skew_amount, x1),  # Top Left
                 (y2, x1 - skew_amount),  # Top Right
                 (y2 + skew_amount, x2),  # Bottom Right
                 (y1, x2 + skew_amount)]
    matrix = []
    for p1, p2 in zip(new_plane, original_plane):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(original_plane).reshape(8)
    perspective_skew_coefficients_matrix = np.dot(np.linalg.pinv(A), B)
    perspective_skew_coefficients_matrix = np.array(perspective_skew_coefficients_matrix).reshape(8)

    return img.transform(img.size, PIL.Image.PERSPECTIVE, perspective_skew_coefficients_matrix,
                         resample=PIL.Image.BICUBIC)


def blur(img, v, **__):
    #assert -3 <= v <= 3
    return img.filter(PIL.ImageFilter.GaussianBlur(v))


def rand_augment_ops(magnitude=10, hparams=None, transforms=None):
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
    return [AutoAugmentOp(name, prob=0.5, magnitude=magnitude, hparams=hparams) for name in transforms]


def _select_rand_weights(weight_idx=0, transforms=None):
    transforms = transforms or _RAND_TRANSFORMS
    assert weight_idx == 0  # only one set of weights currently
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    probs = [rand_weights[k] for k in transforms]
    probs /= np.sum(probs)
    return probs


def rand_augment_transform(config_str, hparams):
    """
    rand selcet transform operation
    """
    magnitude = _MAX_LEVEL  # default to _MAX_LEVEL for magnitude (currently 10)
    num_layers = 2  # default to 2 ops per image
    weight_idx = None  # default to no probability weights for op choice
    config = config_str.split('-')
    assert config[0] == 'rand'
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            # noise param injected via hparams for now
            hparams.setdefault('magnitude_std', float(val))
        elif key == 'm':
            magnitude = int(val)
        elif key == 'n':
            num_layers = int(val)
        elif key == 'w':
            weight_idx = int(val)
        else:
            assert False, 'Unknown RandAugment config section'
    ra_ops = rand_augment_ops(magnitude=magnitude, hparams=hparams)
    choice_weights = None if weight_idx is None else _select_rand_weights(weight_idx)

    final_result = RandAugment(ra_ops, num_layers, choice_weights=choice_weights)
    return final_result


LEVEL_TO_ARG = {
    'Distort': _distort_level_to_arg,
    'Zoom': _zoom_level_to_arg,
    'Blur': _blur_level_to_arg,
    'Skew': _skew_level_to_arg,
    'AutoContrast': None,
    'Equalize': None,
    'Invert': None,
    'Rotate': _rotate_level_to_arg,
    'PosterizeOriginal': _posterize_original_level_to_arg,
    'PosterizeResearch': _posterize_research_level_to_arg,
    'PosterizeTpu': _posterize_tpu_level_to_arg,
    'Solarize': _solarize_level_to_arg,
    'SolarizeAdd': _solarize_add_level_to_arg,
    'Color': _enhance_level_to_arg,
    'Contrast': _enhance_level_to_arg,
    'Brightness': _enhance_level_to_arg,
    'Sharpness': _enhance_level_to_arg,
    'ShearX': _shear_level_to_arg,
    'ShearY': _shear_level_to_arg,
    'TranslateX': _translate_abs_level_to_arg,
    'TranslateY': _translate_abs_level_to_arg,
    'TranslateXRel': _translate_rel_level_to_arg,
    'TranslateYRel': _translate_rel_level_to_arg,
}

NAME_TO_OP = {
    'Distort': distort,
    'Zoom': zoom,
    'Blur': blur,
    'Skew': skew,
    'AutoContrast': auto_contrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'PosterizeOriginal': posterize,
    'PosterizeResearch': posterize,
    'PosterizeTpu': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x_abs,
    'TranslateY': translate_y_abs,
    'TranslateXRel': translate_x_rel,
    'TranslateYRel': translate_y_rel,
}


class AutoAugmentOp:
    """
    AutoAugmentOp class
    """
    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fillcolor=hparams['img_mean'] if 'img_mean' in hparams else _FILL,
            resample=hparams['interpolation'] if 'interpolation' in hparams else _RANDOM_INTERPOLATION,
        )
        self.magnitude_std = self.hparams.get('magnitude_std', 0)

    def __call__(self, img):
        if random.random() > self.prob:
            return img
        magnitude = self.magnitude
        if self.magnitude_std and self.magnitude_std > 0:
            magnitude = random.gauss(magnitude, self.magnitude_std)
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else tuple()
        return self.aug_fn(img, *level_args, **self.kwargs)


class RandAugment:
    """
    rand augment class
    """
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img):
        ops = np.random.choice(
            self.ops, self.num_layers, replace=self.choice_weights is None, p=self.choice_weights)
        for op in ops:
            img = op(img)
        return img
