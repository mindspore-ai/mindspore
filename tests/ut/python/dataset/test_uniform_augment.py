# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
Testing UniformAugment in DE
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.vision.py_transforms as F
from mindspore import log as logger
from util import visualize_list, diff_mse

DATA_DIR = "../data/dataset/testImageNetData/train/"


def test_uniform_augment_callable(num_ops=2):
    """
    Test UniformAugment is callable
    """
    logger.info("test_uniform_augment_callable")
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    decode_op = C.Decode()
    img = decode_op(img)
    assert img.shape == (2268, 4032, 3)

    transforms_ua = [C.RandomCrop(size=[400, 400], padding=[32, 32, 32, 32]),
                     C.RandomCrop(size=[400, 400], padding=[32, 32, 32, 32])]
    uni_aug = C.UniformAugment(transforms=transforms_ua, num_ops=num_ops)
    img = uni_aug(img)
    assert img.shape == (2268, 4032, 3) or img.shape == (400, 400, 3)


def test_uniform_augment(plot=False, num_ops=2):
    """
    Test UniformAugment
    """
    logger.info("Test UniformAugment")

    # Original Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = mindspore.dataset.transforms.py_transforms.Compose([F.Decode(),
                                                                              F.Resize((224, 224)),
                                                                              F.ToTensor()])

    ds_original = data_set.map(operations=transforms_original, input_columns="image")

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = np.transpose(image.asnumpy(), (0, 2, 3, 1))
        else:
            images_original = np.append(images_original,
                                        np.transpose(image.asnumpy(), (0, 2, 3, 1)),
                                        axis=0)

            # UniformAugment Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transform_list = [F.RandomRotation(45),
                      F.RandomColor(),
                      F.RandomSharpness(),
                      F.Invert(),
                      F.AutoContrast(),
                      F.Equalize()]

    transforms_ua = \
        mindspore.dataset.transforms.py_transforms.Compose([F.Decode(),
                                                            F.Resize((224, 224)),
                                                            F.UniformAugment(transforms=transform_list,
                                                                             num_ops=num_ops),
                                                            F.ToTensor()])

    ds_ua = data_set.map(operations=transforms_ua, input_columns="image")

    ds_ua = ds_ua.batch(512)

    for idx, (image, _) in enumerate(ds_ua):
        if idx == 0:
            images_ua = np.transpose(image.asnumpy(), (0, 2, 3, 1))
        else:
            images_ua = np.append(images_ua,
                                  np.transpose(image.asnumpy(), (0, 2, 3, 1)),
                                  axis=0)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_ua[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))

    if plot:
        visualize_list(images_original, images_ua)


def test_cpp_uniform_augment(plot=False, num_ops=2):
    """
    Test UniformAugment
    """
    logger.info("Test CPP UniformAugment")

    # Original Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = [C.Decode(), C.Resize(size=[224, 224]),
                           F.ToTensor()]

    ds_original = data_set.map(operations=transforms_original, input_columns="image")

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = np.transpose(image.asnumpy(), (0, 2, 3, 1))
        else:
            images_original = np.append(images_original,
                                        np.transpose(image.asnumpy(), (0, 2, 3, 1)),
                                        axis=0)

    # UniformAugment Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    transforms_ua = [C.RandomCrop(size=[224, 224], padding=[32, 32, 32, 32]),
                     C.RandomHorizontalFlip(),
                     C.RandomVerticalFlip(),
                     C.RandomColorAdjust(),
                     C.RandomRotation(degrees=45)]

    uni_aug = C.UniformAugment(transforms=transforms_ua, num_ops=num_ops)

    transforms_all = [C.Decode(), C.Resize(size=[224, 224]),
                      uni_aug,
                      F.ToTensor()]

    ds_ua = data_set.map(operations=transforms_all, input_columns="image", num_parallel_workers=1)

    ds_ua = ds_ua.batch(512)

    for idx, (image, _) in enumerate(ds_ua):
        if idx == 0:
            images_ua = np.transpose(image.asnumpy(), (0, 2, 3, 1))
        else:
            images_ua = np.append(images_ua,
                                  np.transpose(image.asnumpy(), (0, 2, 3, 1)),
                                  axis=0)
    if plot:
        visualize_list(images_original, images_ua)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_ua[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_cpp_uniform_augment_exception_pyops(num_ops=2):
    """
    Test UniformAugment invalid op in operations
    """
    logger.info("Test CPP UniformAugment invalid OP exception")

    transforms_ua = [C.RandomCrop(size=[224, 224], padding=[32, 32, 32, 32]),
                     C.RandomHorizontalFlip(),
                     C.RandomVerticalFlip(),
                     C.RandomColorAdjust(),
                     C.RandomRotation(degrees=45),
                     F.Invert()]

    with pytest.raises(TypeError) as e:
        C.UniformAugment(transforms=transforms_ua, num_ops=num_ops)

    logger.info("Got an exception in DE: {}".format(str(e)))
    assert "Type of Transforms[5] must be c_transform" in str(e.value)


def test_cpp_uniform_augment_exception_large_numops(num_ops=6):
    """
    Test UniformAugment invalid large number of ops
    """
    logger.info("Test CPP UniformAugment invalid large num_ops exception")

    transforms_ua = [C.RandomCrop(size=[224, 224], padding=[32, 32, 32, 32]),
                     C.RandomHorizontalFlip(),
                     C.RandomVerticalFlip(),
                     C.RandomColorAdjust(),
                     C.RandomRotation(degrees=45)]

    try:
        _ = C.UniformAugment(transforms=transforms_ua, num_ops=num_ops)

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "num_ops" in str(e)


def test_cpp_uniform_augment_exception_nonpositive_numops(num_ops=0):
    """
    Test UniformAugment invalid non-positive number of ops
    """
    logger.info("Test CPP UniformAugment invalid non-positive num_ops exception")

    transforms_ua = [C.RandomCrop(size=[224, 224], padding=[32, 32, 32, 32]),
                     C.RandomHorizontalFlip(),
                     C.RandomVerticalFlip(),
                     C.RandomColorAdjust(),
                     C.RandomRotation(degrees=45)]

    try:
        _ = C.UniformAugment(transforms=transforms_ua, num_ops=num_ops)

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input num_ops must be greater than 0" in str(e)


def test_cpp_uniform_augment_exception_float_numops(num_ops=2.5):
    """
    Test UniformAugment invalid float number of ops
    """
    logger.info("Test CPP UniformAugment invalid float num_ops exception")

    transforms_ua = [C.RandomCrop(size=[224, 224], padding=[32, 32, 32, 32]),
                     C.RandomHorizontalFlip(),
                     C.RandomVerticalFlip(),
                     C.RandomColorAdjust(),
                     C.RandomRotation(degrees=45)]

    try:
        _ = C.UniformAugment(transforms=transforms_ua, num_ops=num_ops)

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Argument num_ops with value 2.5 is not of type (<class 'int'>,)" in str(e)


def test_cpp_uniform_augment_random_crop_badinput(num_ops=1):
    """
    Test UniformAugment with greater crop size
    """
    logger.info("Test CPP UniformAugment with random_crop bad input")
    batch_size = 2
    cifar10_dir = "../data/dataset/testCifar10Data"
    ds1 = ds.Cifar10Dataset(cifar10_dir, shuffle=False)  # shape = [32,32,3]

    transforms_ua = [
        # Note: crop size [224, 224] > image size [32, 32]
        C.RandomCrop(size=[224, 224]),
        C.RandomHorizontalFlip()
    ]
    uni_aug = C.UniformAugment(transforms=transforms_ua, num_ops=num_ops)
    ds1 = ds1.map(operations=uni_aug, input_columns="image")

    # apply DatasetOps
    ds1 = ds1.batch(batch_size, drop_remainder=True, num_parallel_workers=1)
    num_batches = 0
    try:
        for _ in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):
            num_batches += 1
    except Exception as e:
        assert "crop size" in str(e)


if __name__ == "__main__":
    test_uniform_augment_callable(num_ops=2)
    test_uniform_augment(num_ops=1, plot=True)
    test_cpp_uniform_augment(num_ops=1, plot=True)
    test_cpp_uniform_augment_exception_pyops(num_ops=1)
    test_cpp_uniform_augment_exception_large_numops(num_ops=6)
    test_cpp_uniform_augment_exception_nonpositive_numops(num_ops=0)
    test_cpp_uniform_augment_exception_float_numops(num_ops=2.5)
    test_cpp_uniform_augment_random_crop_badinput(num_ops=1)
