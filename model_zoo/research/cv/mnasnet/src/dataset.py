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
Data operations, will be used in train.py and eval.py
"""
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C


def create_dataset(dataset_path, do_train, batch_size=16, device_num=1, rank=0):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        batch_size(int): the batch size of dataset. Default: 16.
        device_num (int): Number of shards that the dataset should be divided into (default=1).
        rank (int): The shard ID within num_shards (default=0).

    Returns:
        dataset
    """
    if device_num == 1:
        ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=32, shuffle=True)
    else:
        ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=32, shuffle=True,
                                   num_shards=device_num, shard_id=rank)
    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(224),
            C.RandomHorizontalFlip(prob=0.5),
            C.RandomSelectSubpolicy(imagenet_policy)
            # C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(255),
            C.CenterCrop(224)
        ]
    trans += [
        C.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
        # C.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255]),
        C.HWC2CHW(),
        C2.TypeCast(mstype.float32)
    ]

    type_cast_op = C2.TypeCast(mstype.int32)
    ds = ds.map(input_columns="image", operations=trans, num_parallel_workers=8)
    ds = ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=8)
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


# define Auto Augmentation operators
PARAMETER_MAX = 10


def float_parameter(level, maxval):
    """augment function"""
    return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
    """augment function"""
    return int(level * maxval / PARAMETER_MAX)


def shear_x(level):
    """augment function"""
    v = float_parameter(level, 0.3)
    return C2.RandomChoice([C.RandomAffine(degrees=0, shear=(-v, -v)),
                            C.RandomAffine(degrees=0, shear=(v, v))])


def shear_y(level):
    """augment function"""
    v = float_parameter(level, 0.3)
    return C2.RandomChoice(
        [C.RandomAffine(degrees=0, shear=(0, 0, -v, -v)),
         C.RandomAffine(degrees=0, shear=(0, 0, v, v))])


def translate_x(level):
    """augment function"""
    v = float_parameter(level, 150 / 331)
    return C2.RandomChoice([C.RandomAffine(degrees=0, translate=(-v, -v)),
                            C.RandomAffine(degrees=0, translate=(v, v))])


def translate_y(level):
    """augment function"""
    v = float_parameter(level, 150 / 331)
    return C2.RandomChoice(
        [C.RandomAffine(degrees=0, translate=(0, 0, -v, -v)),
         C.RandomAffine(degrees=0, translate=(0, 0, v, v))])


def color_impl(level):
    """augment function"""
    v = float_parameter(level, 1.8) + 0.1
    return C.RandomColor(degrees=(v, v))


def rotate_impl(level):
    """augment function"""
    v = int_parameter(level, 30)
    return C2.RandomChoice([C.RandomRotation(degrees=(-v, -v)),
                            C.RandomRotation(degrees=(v, v))])


def solarize_impl(level):
    """augment function"""
    level = int_parameter(level, 256)
    v = 256 - level
    return C.RandomSolarize(threshold=(0, v))


def posterize_impl(level):
    """augment function"""
    level = int_parameter(level, 4)
    v = 4 - level
    return C.RandomPosterize(bits=(v, v))


def contrast_impl(level):
    """augment function"""
    v = float_parameter(level, 1.8) + 0.1
    return C.RandomColorAdjust(contrast=(v, v))


def autocontrast_impl(level):
    """augment function"""
    return C.AutoContrast()


def sharpness_impl(level):
    """augment function"""
    v = float_parameter(level, 1.8) + 0.1
    return C.RandomSharpness(degrees=(v, v))


def brightness_impl(level):
    """augment function"""
    v = float_parameter(level, 1.8) + 0.1
    return C.RandomColorAdjust(brightness=(v, v))


# define the Auto Augmentation policy
imagenet_policy = [
    [(posterize_impl(8), 0.4), (rotate_impl(9), 0.6)],
    [(solarize_impl(5), 0.6), (autocontrast_impl(5), 0.6)],
    [(C.Equalize(), 0.8), (C.Equalize(), 0.6)],
    [(posterize_impl(7), 0.6), (posterize_impl(6), 0.6)],
    [(C.Equalize(), 0.4), (solarize_impl(4), 0.2)],

    [(C.Equalize(), 0.4), (rotate_impl(8), 0.8)],
    [(solarize_impl(3), 0.6), (C.Equalize(), 0.6)],
    [(posterize_impl(5), 0.8), (C.Equalize(), 1.0)],
    [(rotate_impl(3), 0.2), (solarize_impl(8), 0.6)],
    [(C.Equalize(), 0.6), (posterize_impl(6), 0.4)],

    [(rotate_impl(8), 0.8), (color_impl(0), 0.4)],
    [(rotate_impl(9), 0.4), (C.Equalize(), 0.6)],
    [(C.Equalize(), 0.0), (C.Equalize(), 0.8)],
    [(C.Invert(), 0.6), (C.Equalize(), 1.0)],
    [(color_impl(4), 0.6), (contrast_impl(8), 1.0)],

    [(rotate_impl(8), 0.8), (color_impl(2), 1.0)],
    [(color_impl(8), 0.8), (solarize_impl(7), 0.8)],
    [(sharpness_impl(7), 0.4), (C.Invert(), 0.6)],
    [(shear_x(5), 0.6), (C.Equalize(), 1.0)],
    [(color_impl(0), 0.4), (C.Equalize(), 0.6)],

    [(C.Equalize(), 0.4), (solarize_impl(4), 0.2)],
    [(solarize_impl(5), 0.6), (autocontrast_impl(5), 0.6)],
    [(C.Invert(), 0.6), (C.Equalize(), 1.0)],
    [(color_impl(4), 0.6), (contrast_impl(8), 1.0)],
    [(C.Equalize(), 0.8), (C.Equalize(), 0.6)],
]
