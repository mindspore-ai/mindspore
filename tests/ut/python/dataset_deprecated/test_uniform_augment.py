# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
Test UniformAugment op in Dataset
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms as PT
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.vision.py_transforms as F
from mindspore import log as logger
from ..dataset.util import visualize_list, diff_mse

DATA_DIR = "../data/dataset/testImageNetData/train/"


def test_cpp_uniform_augment_callable(num_ops=2):
    """
    Feature: UniformAugment
    Description: Test UniformAugment C++ op under under execute mode. Use list for transforms list argument.
    Expectation: Output's shape is the same as expected output's shape
    """
    logger.info("test_cpp_uniform_augment_callable")
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    decode_op = C.Decode()
    img = decode_op(img)
    assert img.shape == (2268, 4032, 3)

    transforms_ua = [C.RandomCrop(size=[200, 400], padding=[32, 32, 32, 32]),
                     C.RandomCrop(size=[200, 400], padding=[32, 32, 32, 32])]
    uni_aug = C.UniformAugment(transforms=transforms_ua, num_ops=num_ops)
    img = uni_aug(img)
    assert img.shape == (2268, 4032, 3) or img.shape == (200, 400, 3)


def test_cpp_uniform_augment_callable_tuple(num_ops=2):
    """
    Feature: UniformAugment
    Description: Test UniformAugment C++ op under under execute mode. Use tuple for transforms list argument.
    Expectation: Output's shape is the same as expected output's shape
    """
    logger.info("test_cpp_uniform_augment_callable")
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    decode_op = C.Decode()
    img = decode_op(img)
    assert img.shape == (2268, 4032, 3)

    transforms_ua = (C.RandomCrop(size=[200, 400], padding=[32, 32, 32, 32]),
                     C.RandomCrop(size=[200, 400], padding=[32, 32, 32, 32]))
    uni_aug = C.UniformAugment(transforms=transforms_ua, num_ops=num_ops)
    img = uni_aug(img)
    assert img.shape == (2268, 4032, 3) or img.shape == (200, 400, 3)


def test_py_uniform_augment_callable(num_ops=2):
    """
    Feature: UniformAugment
    Description: Test UniformAugment Python op under under execute mode. Use list for transforms list argument.
    Expectation: Output's shape is the same as expected output's shape
    """
    logger.info("test_cpp_uniform_augment_callable")
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    decode_op = F.Decode()
    img = decode_op(img)
    assert img.size == (4032, 2268)

    transforms_ua = [F.RandomCrop(size=[200, 400], padding=[32, 32, 32, 32]),
                     F.RandomCrop(size=[200, 400], padding=[32, 32, 32, 32])]
    uni_aug = F.UniformAugment(transforms=transforms_ua, num_ops=num_ops)
    img = uni_aug(img)
    assert img.size == (4032, 2268) or img.size == (400, 200)


def test_py_uniform_augment_pyfunc(plot=False, num_ops=2):
    """
    Feature: UniformAugment Op
    Description: Test Python op with valid Python function in transforms list.  Include pyfunc in transforms list.
    Expectation: Pipeline is successfully executed
    """
    logger.info("Test UniformAugment")

    # Original Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = PT.Compose([F.Decode(),
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
                      lambda x: x,
                      F.AutoContrast(),
                      F.Equalize()]

    transforms_ua = PT.Compose([F.Decode(),
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


def test_cpp_uniform_augment_exception_pyops(num_ops=2):
    """
    Feature: UniformAugment Op
    Description: Test C++ op with invalid Python op in transforms list
    Expectation: Invalid input is detected
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


def test_cpp_uniform_augment_exception_pyfunc():
    """
    Feature: UniformAugment
    Description: Test C++ op with pyfunc in transforms list
    Expectation: Exception is raised as expected
    """
    pyfunc = lambda x: x
    transforms_list = [C.RandomVerticalFlip(), pyfunc]
    with pytest.raises(TypeError) as error_info:
        _ = C.UniformAugment(transforms_list, 1)
    error_msg = "Type of Transforms[1] must be c_transform, but got <class 'function'>"
    assert error_msg in str(error_info.value)


def test_c_uniform_augment_exception_num_ops():
    """
    Feature: UniformAugment
    Description: Test C++ op with more ops than number of ops in transforms list
    Expectation: Exception is raised as expected
    """
    transforms_list = [C.RandomVerticalFlip()]
    with pytest.raises(ValueError) as error_info:
        _ = C.UniformAugment(transforms_list, 3)
    error_msg = "num_ops is greater than transforms list size"
    assert error_msg in str(error_info.value)


def test_py_uniform_augment_exception_num_ops():
    """
    Feature: UniformAugment
    Description: Test Python op with more ops than number of ops in transforms list
    Expectation: Exception is raised as expected
    """
    pyfunc = lambda x: x
    transforms_list = [F.RandomVerticalFlip(), pyfunc]
    with pytest.raises(ValueError) as error_info:
        _ = F.UniformAugment(transforms_list, 9)
    error_msg = "num_ops cannot be greater than the length of transforms list."
    assert error_msg in str(error_info.value)


def test_py_uniform_augment_exception_tuple1():
    """
    Feature: UniformAugment
    Description: Test Python op with transforms argument as tuple
    Expectation: Exception is raised as expected
    """
    transforms_list = (F.RandomVerticalFlip())
    with pytest.raises(TypeError) as error_info:
        _ = F.UniformAugment(transforms_list, 1)
    error_msg = "not of type [<class 'list'>], but got"
    assert error_msg in str(error_info.value)


def test_py_uniform_augment_exception_tuple2():
    """
    Feature: UniformAugment
    Description: Test Python op with transforms argument as tuple
    Expectation: Exception is raised as expected
    """
    transforms_list = (F.RandomHorizontalFlip(), F.RandomVerticalFlip())
    with pytest.raises(TypeError) as error_info:
        _ = F.UniformAugment(transforms_list, 1)
    error_msg = "not of type [<class 'list'>], but got <class 'tuple'>."
    assert error_msg in str(error_info.value)


if __name__ == "__main__":
    test_cpp_uniform_augment_callable()
    test_cpp_uniform_augment_callable_tuple()
    test_py_uniform_augment_callable()
    test_py_uniform_augment_pyfunc(plot=True, num_ops=7)
    test_cpp_uniform_augment_exception_pyops(num_ops=1)
    test_cpp_uniform_augment_exception_pyfunc()
    test_c_uniform_augment_exception_num_ops()
    test_py_uniform_augment_exception_num_ops()
    test_py_uniform_augment_exception_tuple1()
    test_py_uniform_augment_exception_tuple2()
