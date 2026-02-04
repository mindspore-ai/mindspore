# Copyright 2025 Huawei Technologies Co., Ltd
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
import os
import platform
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import visualize_list, diff_mse, config_get_set_seed

DATA_DIR = "../data/dataset/testImageNetData/train/"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")


def py_func(x):
    '''Python custom methods'''
    return x


def test_uniform_augment_callable(num_ops=2):
    """
    Feature: UniformAugment
    Description: Test UniformAugment under execute mode
    Expectation: Output's shape is the same as expected output's shape
    """
    logger.info("test_uniform_augment_callable")
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    decode_op = vision.Decode()
    img = decode_op(img)
    assert img.shape == (2268, 4032, 3)

    transforms_ua = [vision.RandomCrop(size=[200, 400], padding=[32, 32, 32, 32]),
                     vision.RandomCrop(size=[200, 400], padding=[32, 32, 32, 32])]
    uni_aug = vision.UniformAugment(transforms=transforms_ua, num_ops=num_ops)
    img = uni_aug(img)
    assert img.shape == (2268, 4032, 3) or img.shape == (200, 400, 3)


def test_uniform_augment_callable_pil(num_ops=2):
    """
    Feature: UniformAugment
    Description: Test UniformAugment under execute mode, with PIL input.
    Expectation: Output's shape is the same as expected output's shape
    """
    logger.info("test_uniform_augment_callable")
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    decode_op = vision.Decode(to_pil=True)
    img = decode_op(img)
    assert img.size == (4032, 2268)

    transforms_ua = [vision.RandomCrop(size=[200, 400], padding=[32, 32, 32, 32]),
                     vision.RandomCrop(size=[200, 400], padding=[32, 32, 32, 32])]
    uni_aug = vision.UniformAugment(transforms=transforms_ua, num_ops=num_ops)
    img = uni_aug(img)
    assert img.size == (4032, 2268) or img.size == (400, 200)


def test_uniform_augment_callable_pil_pyfunc(num_ops=3):
    """
    Feature: UniformAugment
    Description: Test UniformAugment under execute mode, with PIL input. Include pyfunc in transforms list.
    Expectation: Output's shape is the same as expected output's shape
    """
    logger.info("test_uniform_augment_callable")
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    decode_op = vision.Decode(to_pil=True)
    img = decode_op(img)
    assert img.size == (4032, 2268)

    transforms_ua = [vision.RandomCrop(size=[200, 400], padding=[32, 32, 32, 32]),
                     lambda x: x,
                     vision.RandomCrop(size=[200, 400], padding=[32, 32, 32, 32])]
    uni_aug = vision.UniformAugment(transforms=transforms_ua, num_ops=num_ops)
    img = uni_aug(img)
    assert img.size == (4032, 2268) or img.size == (400, 200)


def test_uniform_augment_callable_tuple(num_ops=2):
    """
    Feature: UniformAugment
    Description: Test UniformAugment under execute mode. Use tuple for transforms list argument.
    Expectation: Output's shape is the same as expected output's shape
    """
    logger.info("test_uniform_augment_callable")
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    decode_op = vision.Decode()
    img = decode_op(img)
    assert img.shape == (2268, 4032, 3)

    transforms_ua = (vision.RandomCrop(size=[200, 400], padding=[32, 32, 32, 32]),
                     vision.RandomCrop(size=[200, 400], padding=[32, 32, 32, 32]))
    uni_aug = vision.UniformAugment(transforms=transforms_ua, num_ops=num_ops)
    img = uni_aug(img)
    assert img.shape == (2268, 4032, 3) or img.shape == (200, 400, 3)


def test_uniform_augment(plot=False, num_ops=2):
    """
    Feature: UniformAugment
    Description: Test UniformAugment using Python implementation
    Expectation: Output is the same as expected output
    """
    logger.info("Test UniformAugment")

    # Original Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = mindspore.dataset.transforms.Compose([vision.Decode(True),
                                                                vision.Resize((224, 224)),
                                                                vision.ToTensor()])

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

    transform_list = [vision.RandomRotation(45),
                      vision.RandomColor(),
                      vision.RandomSharpness(),
                      vision.Invert(),
                      vision.AutoContrast(),
                      vision.Equalize()]

    transforms_ua = \
        mindspore.dataset.transforms.Compose([vision.Decode(True),
                                              vision.Resize((224, 224)),
                                              vision.UniformAugment(transforms=transform_list,
                                                                    num_ops=num_ops),
                                              vision.ToTensor()])

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


def test_uniform_augment_pyfunc(num_ops=2, my_seed=1):
    """
    Feature: UniformAugment
    Description: Test UniformAugment using Python implementation.   Include pyfunc in transforms list.
    Expectation: Output is the same as expected output
    """
    logger.info("Test UniformAugment with pyfunc")
    original_seed = config_get_set_seed(my_seed)
    logger.info("my_seed= {}".format(str(my_seed)))

    # Original Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = mindspore.dataset.transforms.Compose([vision.Decode(True)])

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

    transform_list = [vision.Invert(),
                      lambda x: x,
                      vision.AutoContrast(),
                      vision.Equalize()
                      ]

    transforms_ua = \
        mindspore.dataset.transforms.Compose([vision.Decode(True),
                                              vision.UniformAugment(transforms=transform_list, num_ops=num_ops)])

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

    # Restore configuration
    ds.config.set_seed(original_seed)


def test_cpp_uniform_augment(plot=False, num_ops=2):
    """
    Feature: UniformAugment
    Description: Test UniformAugment using Cpp implementation
    Expectation: Output is the same as expected output
    """
    logger.info("Test CPP UniformAugment")

    # Original Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = [vision.Decode(), vision.Resize(size=[224, 224]),
                           vision.ToTensor()]

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
    transforms_ua = [vision.RandomCrop(size=[224, 224], padding=[32, 32, 32, 32]),
                     vision.RandomHorizontalFlip(),
                     vision.RandomVerticalFlip(),
                     vision.RandomColorAdjust(),
                     vision.RandomRotation(degrees=45)]

    uni_aug = vision.UniformAugment(transforms=transforms_ua, num_ops=num_ops)

    transforms_all = [vision.Decode(), vision.Resize(size=[224, 224]),
                      uni_aug,
                      vision.ToTensor()]

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


def test_cpp_uniform_augment_exception_large_numops(num_ops=6):
    """
    Feature: UniformAugment
    Description: Test UniformAugment using invalid large number of ops
    Expectation: Exception is raised as expected
    """
    logger.info("Test CPP UniformAugment invalid large num_ops exception")

    transforms_ua = [vision.RandomCrop(size=[224, 224], padding=[32, 32, 32, 32]),
                     vision.RandomHorizontalFlip(),
                     vision.RandomVerticalFlip(),
                     vision.RandomColorAdjust(),
                     vision.RandomRotation(degrees=45)]

    with pytest.raises(ValueError) as error_info:
        _ = vision.UniformAugment(transforms=transforms_ua, num_ops=num_ops)
    logger.info("Got an exception in DE: {}".format(str(error_info)))
    assert "num_ops is greater than transforms list size" in str(error_info)


def test_cpp_uniform_augment_exception_nonpositive_numops(num_ops=0):
    """
    Feature: UniformAugment
    Description: Test UniformAugment using invalid non-positive num_ops
    Expectation: Exception is raised as expected
    """
    logger.info("Test UniformAugment invalid non-positive num_ops exception")

    transforms_ua = [vision.RandomCrop(size=[224, 224], padding=[32, 32, 32, 32]),
                     vision.RandomHorizontalFlip(),
                     vision.RandomVerticalFlip(),
                     vision.RandomColorAdjust(),
                     vision.RandomRotation(degrees=45)]

    with pytest.raises(ValueError) as error_info:
        _ = vision.UniformAugment(transforms=transforms_ua, num_ops=num_ops)
    logger.info("Got an exception in DE: {}".format(str(error_info)))
    assert "Input num_ops must be greater than 0" in str(error_info)


def test_cpp_uniform_augment_exception_float_numops(num_ops=2.5):
    """
    Feature: UniformAugment
    Description: Test UniformAugment using invalid float num_ops
    Expectation: Exception is raised as expected
    """
    logger.info("Test UniformAugment invalid float num_ops exception")

    transforms_ua = [vision.RandomCrop(size=[224, 224], padding=[32, 32, 32, 32]),
                     vision.RandomHorizontalFlip(),
                     vision.RandomVerticalFlip(),
                     vision.RandomColorAdjust(),
                     vision.RandomRotation(degrees=45)]

    with pytest.raises(TypeError) as error_info:
        _ = vision.UniformAugment(transforms=transforms_ua, num_ops=num_ops)
    logger.info("Got an exception in DE: {}".format(str(error_info)))
    assert "Argument num_ops with value 2.5 is not of type [<class 'int'>]" in str(error_info)


def test_cpp_uniform_augment_random_crop_badinput(num_ops=1):
    """
    Feature: UniformAugment
    Description: Test UniformAugment with greater crop size
    Expectation: Exception is raised as expected
    """
    logger.info("Test UniformAugment with random_crop bad input")
    batch_size = 2
    cifar10_dir = "../data/dataset/testCifar10Data"
    ds1 = ds.Cifar10Dataset(cifar10_dir, shuffle=False)  # shape = [32,32,3]

    transforms_ua = [
        # Note: crop size [224, 224] > image size [32, 32]
        vision.RandomCrop(size=[224, 224]),
        vision.RandomHorizontalFlip()
    ]
    uni_aug = vision.UniformAugment(transforms=transforms_ua, num_ops=num_ops)
    ds1 = ds1.map(operations=uni_aug, input_columns="image")

    # apply DatasetOps
    ds1 = ds1.batch(batch_size, drop_remainder=True, num_parallel_workers=1)
    num_batches = 0
    with pytest.raises(RuntimeError) as error_info:
        for _ in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):
            num_batches += 1
    assert "map operation: [UniformAugment] failed." in str(error_info.value)


def test_uniform_augment_operation_01():
    """
    Feature: UniformAugment operation
    Description: Testing the normal functionality of the UniformAugment operator
    Expectation: The Output is equal to the expected output
    """
    # UniformAugment: Handling Operator Redundancy
    dataset = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    transforms = [vision.RandomHorizontalFlip(),
                  vision.RandomColorAdjust(),
                  vision.RandomColorAdjust(),
                  vision.RandomRotation(degrees=45),
                  vision.RandomRotation(degrees=45)]
    ua_op = vision.UniformAugment(transforms=transforms, num_ops=2)
    dataset = dataset.map(input_columns=["image"], operations=ua_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # UniformAugment: num_ops equals the number of operators processed
    dataset = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    transforms = [vision.RandomHorizontalFlip(),
                  vision.RandomColorAdjust(),
                  vision.RandomColorAdjust(),
                  vision.RandomRotation(degrees=45)]
    ua_op = vision.UniformAugment(transforms=transforms, num_ops=4)
    dataset = dataset.map(input_columns=["image"], operations=ua_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # UniformAugment: Processing operators are Python-side operators.
    dataset = ds.ImageFolderDataset(DATA_DIR, shuffle=False)
    transforms = [vision.RandomHorizontalFlip(0.5),
                  vision.RandomHorizontalFlip(0.5)]
    ua_op = vision.UniformAugment(transforms=transforms, num_ops=2)
    dataset = dataset.map(input_columns=["image"], operations=vision.Decode(to_pil=True))
    dataset = dataset.map(input_columns=["image"], operations=ua_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # UniformAugment: Eager mode, input is Pillow
    image = Image.open(image_jpg)
    transforms = [vision.RandomRotation((45, 45)), vision.RandomVerticalFlip(), vision.RandomColorAdjust(),
                  vision.RandomRotation((90, 90))]
    num_ops = 3
    ua_op = vision.UniformAugment(transforms, num_ops)
    _ = ua_op(image)

    # UniformAugment: Eager mode, input is a NumPy array
    image = np.random.randint(0, 255, (464, 464, 3)).astype(np.uint8)
    transforms = (vision.RandomVerticalFlip(),)
    num_ops = 1
    ua_op = vision.UniformAugment(transforms, num_ops)
    ua_op(image)


def test_uniform_augment_exception_01():
    """
    Feature: UniformAugment operation
    Description: Testing the UniformAugment Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # UniformAugment: Abnormal Testing, transforms are empty
    transforms = []
    with pytest.raises(ValueError, match="num_ops is greater than transforms list size."):
        vision.UniformAugment(transforms=transforms, num_ops=2)

    # UniformAugment: Abnormal Testing, The transforms parameter is a string.
    transforms = "12345"
    with pytest.raises(TypeError, match="Argument transforms list with value 12345 is not of type \\[<class"
                                        " 'list'>, <class 'tuple'>\\], but got <class 'str'>."):
        vision.UniformAugment(transforms=transforms, num_ops=2)

    # UniformAugment: Abnormal Testing, num_ops is greater than the length of transforms
    transforms = [vision.RandomHorizontalFlip(),
                  vision.RandomColorAdjust(),
                  vision.RandomColorAdjust(),
                  vision.RandomRotation(degrees=45)]
    with pytest.raises(ValueError, match="num_ops is greater than transforms list size."):
        vision.UniformAugment(transforms=transforms, num_ops=5)

    # UniformAugment: Abnormal Testing, num_ops equals 0
    transforms = [vision.RandomHorizontalFlip(),
                  vision.RandomColorAdjust(),
                  vision.RandomColorAdjust(),
                  vision.RandomRotation(degrees=45)]
    with pytest.raises(ValueError, match="Input num_ops must be greater than 0."):
        vision.UniformAugment(transforms=transforms, num_ops=0)

    # UniformAugment: Abnormal Testing, num_ops equals 2.1
    transforms = [vision.RandomHorizontalFlip(),
                  vision.RandomColorAdjust(),
                  vision.RandomColorAdjust(),
                  vision.RandomRotation(degrees=45)]
    with pytest.raises(TypeError, match="Argument num_ops with value 2.1 is not of type \\[<class"
                                        " 'int'>\\], but got <class 'float'>."):
        vision.UniformAugment(transforms=transforms, num_ops=2.1)

    # UniformAugment: Abnormal Testing, num_ops equals ""
    transforms = [vision.RandomHorizontalFlip(),
                  vision.RandomColorAdjust(),
                  vision.RandomColorAdjust(),
                  vision.RandomRotation(degrees=45)]
    with pytest.raises(TypeError, match="is not of type \\[<class 'int'>\\], but got <class 'str'>."):
        vision.UniformAugment(transforms=transforms, num_ops="")

    # UniformAugment: Abnormal Testing, transforms equals int
    image = np.random.randint(0, 255, (464, 464, 3)).astype(np.uint8)
    with pytest.raises(TypeError, match="object of type 'int' has no len\\(\\)"):
        ua_op = vision.UniformAugment(1)
        ua_op(image)

    # UniformAugment: Abnormal Testing, Multi-parameter
    transforms = [vision.RandomHorizontalFlip(),
                  vision.RandomColorAdjust(),
                  vision.RandomColorAdjust(),
                  vision.RandomRotation(degrees=45)]
    num_ops = 1
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        vision.UniformAugment(transforms, num_ops, more_para)

    # UniformAugment: Abnormal Testing, input equals list
    ds.config.set_seed(10)
    image = 1
    transforms = [vision.RandomColorAdjust((1, 10.4), 0.2, (0, 10.8), (-0.5, 0.5))]
    num_ops = 1
    ua_op = vision.UniformAugment(transforms, num_ops)
    if platform.system() == "Linux":
        with pytest.raises(RuntimeError):
            ua_op(image)
    else:
        ua_op(image)

    # UniformAugment: Abnormal Testing, Transforms are two-dimensional.
    transforms = [[vision.AutoContrast()],
                  [vision.Invert()]]
    num_ops = 2
    with pytest.raises(TypeError, match="transforms\\[0\\] is neither a transforms"
                                        " op \\(TensorOperation\\) nor a callable pyfunc."):
        vision.UniformAugment(transforms, num_ops)

    # UniformAugment: Abnormal Testing, Transforms are one-dimensional.
    transforms = [vision.AutoContrast(), vision.Invert(), 1]
    num_ops = 1
    with pytest.raises(TypeError, match="transforms\\[2\\] is neither a transforms"
                                        " op \\(TensorOperation\\) nor a callable pyfunc."):
        vision.UniformAugment(transforms, num_ops)

    # UniformAugment: Abnormal Testing, num_ops equals list
    transforms = [vision.AutoContrast(), vision.Invert()]
    num_ops = [1]
    with pytest.raises(TypeError, match="Argument num_ops with value \\[1\\] is not of type \\[<class 'int'>\\]."):
        vision.UniformAugment(transforms, num_ops)

    # UniformAugment: Abnormal Testing, transforms equals operator
    transforms = vision.AutoContrast()
    num_ops = 1
    with pytest.raises(TypeError, match="object of type 'AutoContrast' has no len\\(\\)"):
        vision.UniformAugment(transforms, num_ops)

    # UniformAugment: Abnormal Testing, transforms equals custom operators
    transforms = [vision.RandomVerticalFlip(), py_func]
    vision.UniformAugment(transforms, 1)


if __name__ == "__main__":
    test_uniform_augment_callable()
    test_uniform_augment_callable_pil()
    test_uniform_augment_callable_pil_pyfunc()
    test_uniform_augment_callable_tuple()
    test_uniform_augment(num_ops=6, plot=True)
    test_uniform_augment_pyfunc(num_ops=2, my_seed=1)
    test_cpp_uniform_augment(num_ops=1, plot=True)
    test_cpp_uniform_augment_exception_large_numops(num_ops=6)
    test_cpp_uniform_augment_exception_nonpositive_numops(num_ops=0)
    test_cpp_uniform_augment_exception_float_numops(num_ops=2.5)
    test_cpp_uniform_augment_random_crop_badinput(num_ops=1)
    test_uniform_augment_operation_01()
    test_uniform_augment_exception_01()
