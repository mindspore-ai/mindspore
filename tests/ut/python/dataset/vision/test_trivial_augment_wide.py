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
Testing TrivialAugmentWide in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore.dataset.vision.transforms import Decode, TrivialAugmentWide, Resize
from mindspore.dataset.vision.utils import Inter
from mindspore import log as logger
from util import visualize_image, visualize_list, diff_mse

TEST_DATA_DATASET_FUNC ="../data/dataset/"


def test_trivial_augment_wide_pipeline(plot=False):
    """
    Feature: TrivialAugmentWide
    Description: test TrivialAugmentWide pipeline
    Expectation: pass without error
    """
    logger.info("Test TrivialAugmentWide pipeline")
    data_dir = "../data/dataset/testImageNetData/train/"

    # Original Images
    data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    transforms_original = [Decode(), Resize(size=[224, 224])]
    ds_original = data_set.map(operations=transforms_original, input_columns="image")
    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image.asnumpy()
        else:
            images_original = np.append(images_original,
                                        image.asnumpy(),
                                        axis=0)

    # Trivial Augment Wided Images with ImageNet num_magnitude_bins
    data_set1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    trivial_augment_wide_op = TrivialAugmentWide(31, Inter.BICUBIC, 20)
    transforms = [Decode(), Resize(size=[224, 224]), trivial_augment_wide_op]
    ds_trivial_augment_wide = data_set1.map(operations=transforms, input_columns="image")
    ds_trivial_augment_wide = ds_trivial_augment_wide.batch(512)

    for idx, (image, _) in enumerate(ds_trivial_augment_wide):
        if idx == 0:
            images_trivial_augment_wide = image.asnumpy()
        else:
            images_trivial_augment_wide = np.append(images_trivial_augment_wide,
                                                    image.asnumpy(), axis=0)
    assert images_original.shape[0] == images_trivial_augment_wide.shape[0]
    if plot:
        visualize_list(images_original, images_trivial_augment_wide)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_trivial_augment_wide[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))

    # Trivial Augment Wided Images with Cifar10 num_magnitude_bins
    data_set2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    trivial_augment_wide_op = TrivialAugmentWide(31, Inter.BILINEAR, 20)
    transforms = [Decode(), Resize(size=[224, 224]), trivial_augment_wide_op]
    ds_trivial_augment_wide = data_set2.map(operations=transforms, input_columns="image")
    ds_trivial_augment_wide = ds_trivial_augment_wide.batch(512)
    for idx, (image, _) in enumerate(ds_trivial_augment_wide):
        if idx == 0:
            images_trivial_augment_wide = image.asnumpy()
        else:
            images_trivial_augment_wide = np.append(images_trivial_augment_wide,
                                                    image.asnumpy(), axis=0)
    assert images_original.shape[0] == images_trivial_augment_wide.shape[0]
    if plot:
        visualize_list(images_original, images_trivial_augment_wide)

    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_trivial_augment_wide[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))

    # Trivial Augment Wide Images with SVHN num_magnitude_bins
    data_set3 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    trivial_augment_wide_op = TrivialAugmentWide(31, Inter.NEAREST, 20)
    transforms = [Decode(), Resize(size=[224, 224]), trivial_augment_wide_op]
    ds_trivial_augment_wide = data_set3.map(operations=transforms, input_columns="image")
    ds_trivial_augment_wide = ds_trivial_augment_wide.batch(512)
    for idx, (image, _) in enumerate(ds_trivial_augment_wide):
        if idx == 0:
            images_trivial_augment_wide = image.asnumpy()
        else:
            images_trivial_augment_wide = np.append(images_trivial_augment_wide,
                                                    image.asnumpy(), axis=0)
    assert images_original.shape[0] == images_trivial_augment_wide.shape[0]
    if plot:
        visualize_list(images_original, images_trivial_augment_wide)

    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_trivial_augment_wide[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_trivial_augment_wide_eager(plot=False):
    """
    Feature: TrivialAugmentWide
    Description: test TrivialAugmentWide eager
    Expectation: pass without error
    """
    image_file = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
    img = np.fromfile(image_file, dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = Decode()(img)
    img_trivial_augment_wided = TrivialAugmentWide(63)(img)

    if plot:
        visualize_image(img, img_trivial_augment_wided)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_trivial_augment_wided),
                                                         img_trivial_augment_wided.shape))
    mse = diff_mse(img_trivial_augment_wided, img)
    logger.info("MSE= {}".format(str(mse)))


def test_trivial_augment_wide_invalid_input():
    """
    Feature: TrivialAugmentWide
    Description: test TrivialAugmentWide with invalid input
    Expectation: throw TypeError
    """
    try:
        image = np.random.randint(0, 256, (300, 300, 3)).astype(np.uint32)
        TrivialAugmentWide()(image)
    except RuntimeError as e:
        assert "TrivialAugmentWide: the data type of image tensor does not match the requirement of operator." in str(e)

    try:
        image = np.random.randint(0, 256, (300, 300, 1)).astype(np.uint8)
        TrivialAugmentWide()(image)
    except RuntimeError as e:
        assert "TrivialAugmentWide: the channel of image tensor does not match the requirement of operator" in str(e)

    try:
        image = np.random.randint(0, 256, (300, 300)).astype(np.uint8)
        TrivialAugmentWide()(image)
    except RuntimeError as e:
        assert "TrivialAugmentWide: the dimension of image tensor does not match the requirement of operator" in str(e)


def test_trivial_augment_wide_invalid_num_magnitude_bins():
    """
    Feature: TrivialAugmentWide
    Description: test TrivialAugmentWide with invalid num_magnitude_bins
    Expectation: throw TypeError
    """
    logger.info("test_trivial_augment_wide_invalid_num_magnitude_bins")
    data_dir = "../data/dataset/testImageNetData/train/"
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        trivial_augment_wide_op = TrivialAugmentWide(num_magnitude_bins=-1)
        dataset.map(operations=trivial_augment_wide_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input num_magnitude_bins is not within the required interval of [2, 16777216]." in str(e)


def test_trivial_augment_wide_invalid_interpolation():
    """
    Feature: TrivialAugmentWide
    Description: test TrivialAugmentWide with invalid interpolation
    Expectation: throw TypeError
    """
    logger.info("test_trivial_augment_wide_invalid_interpolation")
    data_dir = "../data/dataset/testImageNetData/train/"
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        trivial_augment_wide_op = TrivialAugmentWide(interpolation="invalid")
        dataset.map(operations=trivial_augment_wide_op, input_columns=['image'])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Argument interpolation with value invalid is not of type [<enum 'Inter'>]" in str(e)


def test_trivial_augment_wide_invalid_fill_value():
    """
    Feature: TrivialAugmentWide
    Description: test TrivialAugmentWide with invalid fill_value
    Expectation: throw TypeError or ValueError
    """
    logger.info("test_trivial_augment_wide_invalid_fill_value")
    data_dir = "../data/dataset/testImageNetData/train/"
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        trivial_augment_wide_op = TrivialAugmentWide(fill_value=(10, 10))
        dataset.map(operations=trivial_augment_wide_op, input_columns=['image'])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "fill_value should be a single integer or a 3-tuple." in str(e)
    try:
        trivial_augment_wide_op = TrivialAugmentWide(fill_value=300)
        dataset.map(operations=trivial_augment_wide_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "is not within the required interval of [0, 255]." in str(e)


def test_trivialaugmentwide_operation_01():
    """
    Feature: TrivialAugmentWide operation
    Description: Testing the normal functionality of the TrivialAugmentWide operator
    Expectation: The Output is equal to the expected output
    """
    # TrivialAugmentWide operator, normal testing, Pipeline mode, input data is numpy
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(dataset_dir, decode=True, shuffle=False)
    op = vision.TrivialAugmentWide(num_magnitude_bins=2, interpolation=Inter.NEAREST, fill_value=0)
    dataset2 = dataset2.map(operations=op, input_columns=["image"])
    for _ in dataset2.create_dict_iterator(output_numpy=True, num_epochs=1):
        pass

    # TrivialAugmentWide operator, normal testing, Pipeline mode, input data is PIL
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(dataset_dir, decode=False, shuffle=False)
    op = vision.TrivialAugmentWide(num_magnitude_bins=3, interpolation=Inter.BILINEAR, fill_value=1)
    dataset2 = dataset2.map(operations=[vision.Decode(to_pil=True), op], input_columns=["image"])
    for _ in dataset2.create_dict_iterator(output_numpy=True, num_epochs=1):
        pass

    # TrivialAugmentWide operator, normal testing, Pipeline mode, default parameters
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(dataset_dir, decode=True, shuffle=False)
    op = vision.TrivialAugmentWide()
    dataset2 = dataset2.map(operations=op, input_columns=["image"])
    for _ in dataset2.create_dict_iterator(output_numpy=True, num_epochs=1):
        pass

    # TrivialAugmentWide operator, normal testing, eager mode, Input data is numpy
    image = np.random.randint(0, 256, (30, 30, 3)).astype(np.uint8)
    op = vision.TrivialAugmentWide(num_magnitude_bins=10, interpolation=Inter.BICUBIC, fill_value=2)
    _ = op(image)

    # TrivialAugmentWide operator, normal testing, eager mode, Input data is a JPG image
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_path) as image:
        op = vision.TrivialAugmentWide(num_magnitude_bins=20, interpolation=Inter.AREA, fill_value=255)
        _ = op(image)

    # TrivialAugmentWide operator, normal testing, eager mode, Input data is a BMP image.
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    with Image.open(image_path) as image:
        op = vision.TrivialAugmentWide(num_magnitude_bins=100, interpolation=Inter.AREA, fill_value=254)
        _ = op(image)

    # TrivialAugmentWide operator, normal testing, eager mode, Input data is a GIF image in RGB mode
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_path).convert("RGB") as image:
        op = vision.TrivialAugmentWide(num_magnitude_bins=16777216, interpolation=Inter.AREA, fill_value=(0, 0, 1))
        _ = op(image)

    # TrivialAugmentWide operator, normal testing, eager mode, Input data is a PNG image in RGB mode
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with Image.open(image_path).convert("RGB") as image:
        op = vision.TrivialAugmentWide(num_magnitude_bins=100, interpolation=Inter.AREA, fill_value=(255, 255, 254))
        _ = op(image)

    # TrivialAugmentWide operator, normal testing, eager mode, Input data is single-channel data
    image = np.random.randint(0, 256, (300, 300, 1)).astype(np.uint8)
    op = vision.TrivialAugmentWide(num_magnitude_bins=100, interpolation=Inter.AREA, fill_value=(255, 255, 254))
    with pytest.raises(RuntimeError, match=r"TrivialAugmentWide: the channel of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in channel "
                                           r"of \(3\). But got channel 1."):
        op(image)


def test_trivialaugmentwide_exception_01():
    """
    Feature: TrivialAugmentWide operation
    Description: Testing the TrivialAugmentWide Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # TrivialAugmentWide operator, anomaly testing, input data is float32
    image = np.random.randint(0, 256, (30, 30, 3)).astype(np.float32)
    op = vision.TrivialAugmentWide()
    with pytest.raises(RuntimeError, match=r"TrivialAugmentWide: the data type of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in type "
                                           r"of \(uint8\). But got type float32."):
        op(image)

    # TrivialAugmentWide operator, anomaly testing, input data is float64
    image = np.random.randint(0, 256, (30, 30, 3)).astype(np.float64)
    op = vision.TrivialAugmentWide()
    with pytest.raises(RuntimeError, match=r"TrivialAugmentWide: the data type of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in type "
                                           r"of \(uint8\). But got type float64."):
        op(image)

    # TrivialAugmentWide operator, anomaly testing, input data is float16
    image = np.random.randint(0, 256, (30, 30, 3)).astype(np.float16)
    op = vision.TrivialAugmentWide()
    with pytest.raises(RuntimeError, match=r"TrivialAugmentWide: the data type of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in type "
                                           r"of \(uint8\). But got type float16."):
        op(image)

    # TrivialAugmentWide operator, anomaly testing, input data is int8
    image = np.random.randint(0, 256, (30, 30, 3)).astype(np.int8)
    op = vision.TrivialAugmentWide()
    with pytest.raises(RuntimeError, match=r"TrivialAugmentWide: the data type of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in type "
                                           r"of \(uint8\). But got type int8."):
        op(image)

    # TrivialAugmentWide operator, anomaly testing, input data is int16
    image = np.random.randint(0, 256, (30, 30, 3)).astype(np.int16)
    op = vision.TrivialAugmentWide()
    with pytest.raises(RuntimeError, match=r"TrivialAugmentWide: the data type of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in type "
                                           r"of \(uint8\). But got type int16."):
        op(image)

    # TrivialAugmentWide operator, anomaly testing, input data is int32
    image = np.random.randint(0, 256, (30, 30, 3)).astype(np.int32)
    op = vision.TrivialAugmentWide()
    with pytest.raises(RuntimeError, match=r"TrivialAugmentWide: the data type of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in type "
                                           r"of \(uint8\). But got type int32."):
        op(image)

    # TrivialAugmentWide operator, anomaly testing, input data is bytes
    image = np.random.randint(0, 256, (30, 30, 3)).astype("S")
    op = vision.TrivialAugmentWide()
    with pytest.raises(RuntimeError, match=r"TrivialAugmentWide: the data type of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in type "
                                           r"of \(uint8\). But got type bytes."):
        op(image)

    # TrivialAugmentWide operator, anomaly testing, The number of input data channels is 2
    image = np.random.randint(0, 256, (30, 30, 2)).astype(np.uint8)
    op = vision.TrivialAugmentWide()
    with pytest.raises(RuntimeError, match=r"TrivialAugmentWide: the channel of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in channel "
                                           r"of \(3\). But got channel 2."):
        op(image)

    # TrivialAugmentWide operator, anomaly testing, The number of input data channels is 4
    image = np.random.randint(0, 256, (30, 30, 4)).astype(np.uint8)
    op = vision.TrivialAugmentWide()
    with pytest.raises(RuntimeError, match=r"TrivialAugmentWide: the channel of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in channel "
                                           r"of \(3\). But got channel 4."):
        op(image)

    # TrivialAugmentWide operator, anomaly testing, The input data has two dimensions
    image = np.random.randint(0, 256, (30, 30)).astype(np.uint8)
    op = vision.TrivialAugmentWide()
    with pytest.raises(RuntimeError, match=r"TrivialAugmentWide: the dimension of image tensor does"
                                           r" not match the requirement of operator. Expecting tensor in dimension "
                                           r"of \(3\), in shape of <H, W, C>. But got dimension 2."):
        op(image)

    # TrivialAugmentWide operator, anomaly testing, The input data has four dimensions
    image = np.random.randint(0, 256, (30, 30, 3, 3)).astype(np.uint8)
    op = vision.TrivialAugmentWide()
    with pytest.raises(RuntimeError, match=r"TrivialAugmentWide: the dimension of image tensor does"
                                           r" not match the requirement of operator. Expecting tensor in dimension "
                                           r"of \(3\), in shape of <H, W, C>. But got dimension 4."):
        op(image)

    # TrivialAugmentWide operator, anomaly testing, The input data has one dimensions
    image = np.random.randint(0, 256, (30,)).astype(np.uint8)
    op = vision.TrivialAugmentWide()
    with pytest.raises(RuntimeError, match=r"TrivialAugmentWide: the dimension of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in dimension "
                                           r"of \(3\), in shape of <H, W, C>. But got dimension 1. You may "
                                           r"need to perform Decode first."):
        op(image)


def test_trivialaugmentwide_exception_02():
    """
    Feature: TrivialAugmentWide operation
    Description: Testing the TrivialAugmentWide Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # TrivialAugmentWide operator, anomaly testing, input data is int
    image = 2
    op = vision.TrivialAugmentWide()
    with pytest.raises(TypeError, match=r"Input should be NumPy or PIL image, got <class 'int'>."):
        op(image)

    # TrivialAugmentWide operator, anomaly testing, input data is bool
    image = True
    op = vision.TrivialAugmentWide()
    with pytest.raises(TypeError, match=r"Input should be NumPy or PIL image, got <class 'bool'>."):
        op(image)

    # TrivialAugmentWide operator, anomaly testing, input data is str
    image = "abc"
    op = vision.TrivialAugmentWide()
    with pytest.raises(TypeError, match=r"Input should be NumPy or PIL image, got <class 'str'>."):
        op(image)

    # TrivialAugmentWide operator, anomaly testing, num_magnitude_bins type error
    with pytest.raises(TypeError, match=r"Argument num_magnitude_bins with value 1 is not of "
                                        r"type \[<class 'int'>\], but got <class 'str'>."):
        vision.TrivialAugmentWide(num_magnitude_bins="1")

    # TrivialAugmentWide operator, anomaly testing, num_magnitude_bins is 1
    with pytest.raises(ValueError, match=r"Input num_magnitude_bins is not within the required "
                                         r"interval of \[2, 16777216\]."):
        vision.TrivialAugmentWide(num_magnitude_bins=1)

    # TrivialAugmentWide operator, anomaly testing, num_magnitude_bins is 16777217
    with pytest.raises(ValueError, match=r"Input num_magnitude_bins is not within the required "
                                         r"interval of \[2, 16777216\]."):
        vision.TrivialAugmentWide(num_magnitude_bins=16777217)

    # TrivialAugmentWide operator, anomaly testing, Interpolation type error
    with pytest.raises(TypeError, match=r"Argument interpolation with value 1 is not of "
                                        r"type \[<enum 'Inter'>\], but got <class 'int'>."):
        vision.TrivialAugmentWide(interpolation=1)

    # TrivialAugmentWide operator, anomaly testing, fill_value type error
    with pytest.raises(TypeError, match=r"fill_value should be a single integer or a 3-tuple."):
        vision.TrivialAugmentWide(fill_value=1.0)

    # TrivialAugmentWide operator, anomaly testing, fill_value is -1
    with pytest.raises(ValueError, match=r"Input fill_value is not within the required interval of \[0, 255\]."):
        vision.TrivialAugmentWide(fill_value=-1)

    # TrivialAugmentWide operator, anomaly testing, fill_value is 256
    with pytest.raises(ValueError, match=r"Input fill_value is not within the required interval of \[0, 255\]."):
        vision.TrivialAugmentWide(fill_value=256)

    # TrivialAugmentWide operator, anomaly testing, fill_value=(-1, 2, 3)
    with pytest.raises(ValueError, match=r"Input fill_value\[0\] is not within the required interval of \[0, 255\]."):
        vision.TrivialAugmentWide(fill_value=(-1, 2, 3))

    # TrivialAugmentWide operator, anomaly testing, fill_value=(256, 2, 3)
    with pytest.raises(ValueError, match=r"Input fill_value\[0\] is not within the required interval of \[0, 255\]."):
        vision.TrivialAugmentWide(fill_value=(256, 2, 3))

    # TrivialAugmentWide operator, anomaly testing, fill_value type error
    with pytest.raises(TypeError, match=r"Argument fill_value\[2\] with value 1.0 is not of "
                                        r"type \[<class 'int'>\], but got <class 'float'>."):
        vision.TrivialAugmentWide(fill_value=(1, 2, 1.0))

    # TrivialAugmentWide operator, anomaly testing, The length of fill_value is 2
    with pytest.raises(TypeError, match=r"fill_value should be a single integer or a 3-tuple."):
        vision.TrivialAugmentWide(fill_value=(2, 3))

    # TrivialAugmentWide operator, anomaly testing, The length of fill_value is 1
    with pytest.raises(TypeError, match=r"fill_value should be a single integer or a 3-tuple."):
        vision.TrivialAugmentWide(fill_value=(1,))

    # TrivialAugmentWide operator, anomaly testing, The length of fill_value is 4
    with pytest.raises(TypeError, match=r"fill_value should be a single integer or a 3-tuple."):
        vision.TrivialAugmentWide(fill_value=(1, 2, 10, 4))

    # TrivialAugmentWide operator, anomaly testing, The length of fill_value is 0
    with pytest.raises(TypeError, match=r"fill_value should be a single integer or a 3-tuple."):
        vision.TrivialAugmentWide(fill_value=())


if __name__ == "__main__":
    test_trivial_augment_wide_pipeline(plot=True)
    test_trivial_augment_wide_eager(plot=True)
    test_trivial_augment_wide_invalid_input()
    test_trivial_augment_wide_invalid_num_magnitude_bins()
    test_trivial_augment_wide_invalid_interpolation()
    test_trivial_augment_wide_invalid_fill_value()
    test_trivialaugmentwide_operation_01()
    test_trivialaugmentwide_exception_01()
    test_trivialaugmentwide_exception_02()
