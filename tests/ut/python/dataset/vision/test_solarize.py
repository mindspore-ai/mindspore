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
Testing Solarize op in DE
"""
import numpy as np
import os
import pytest
from PIL import Image, ImageOps

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import visualize_list, config_get_set_seed, config_get_set_num_parallel_workers, \
    visualize_one_channel_dataset, visualize_image, diff_mse

GENERATE_GOLDEN = False

MNIST_DATA_DIR = "../data/dataset/testMnistData"
DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


def solarize(threshold, plot=False):
    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    solarize_op = vision.Solarize(threshold)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=solarize_op, input_columns=["image"])
    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])
    num_iter = 0
    for dat1, dat2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                          data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        solarize_ms = dat1["image"]
        original = dat2["image"]
        original = Image.fromarray(original.astype('uint8')).convert('RGB')
        solarize_cv = ImageOps.solarize(original, threshold)
        solarize_ms = np.array(solarize_ms)
        solarize_cv = np.array(solarize_cv)
        mse = diff_mse(solarize_ms, solarize_cv)
        logger.info("rotate_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1
        if plot:
            visualize_image(original, solarize_ms, mse, solarize_cv)

    image_solarized = []
    image = []

    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_solarized.append(item1["image"].copy())
        image.append(item2["image"].copy())
    if plot:
        visualize_list(image, image_solarized)


def test_solarize_basic(plot=False):
    """
    Feature: Solarize
    Description: Test Solarize op basic usage
    Expectation: The dataset is processed as expected
    """
    solarize(150.1, plot)
    solarize(120, plot)
    solarize(115, plot)


def test_solarize_mnist(plot=False):
    """
    Feature: Solarize op
    Description: Test Solarize op with MNIST dataset (Grayscale images)
    Expectation: The dataset is processed as expected
    """
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    mnist_1 = ds.MnistDataset(dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)
    mnist_2 = ds.MnistDataset(dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)
    mnist_2 = mnist_2.map(operations=vision.Solarize((1.0, 255.0)), input_columns="image")

    images = []
    images_trans = []
    labels = []

    for _, (data_orig, data_trans) in enumerate(zip(mnist_1, mnist_2)):
        image_orig, label_orig = data_orig
        image_trans, _ = data_trans
        images.append(image_orig.asnumpy())
        labels.append(label_orig.asnumpy())
        images_trans.append(image_trans.asnumpy())

    if plot:
        visualize_one_channel_dataset(images, images_trans, labels)

    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_solarize_errors():
    """
    Feature: Solarize op
    Description: Test that Solarize errors with bad input
    Expectation: Passes the error check test
    """
    with pytest.raises(ValueError) as error_info:
        vision.Solarize((12, 1))
    assert "threshold must be in order of (min, max)." in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        vision.Solarize((-1, 200))
    assert "Input threshold[0] is not within the required interval of [0, 255]." in str(error_info.value)

    try:
        vision.Solarize(("122.1", "140"))
    except TypeError as e:
        assert "Argument threshold[0] with value 122.1 is not of type [<class 'float'>, <class 'int'>]" in str(e)

    try:
        vision.Solarize((122, 100, 30))
    except TypeError as e:
        assert "threshold must be a single number or sequence of two numbers." in str(e)

    try:
        vision.Solarize((120,))
    except TypeError as e:
        assert "threshold must be a single number or sequence of two numbers." in str(e)


def test_input_shape_errors():
    """
    Feature: Solarize op
    Description: Test that Solarize errors with bad input shape
    Expectation: Passes the error check test
    """
    try:
        image = np.random.randint(0, 256, (300, 300, 3, 3)).astype(np.uint8)
        vision.Solarize(5)(image)
    except RuntimeError as e:
        assert "Solarize: the dimension of image tensor does not match the requirement of operator" in str(e)

    try:
        image = np.random.randint(0, 256, (4, 300, 300)).astype(np.uint8)
        vision.Solarize(5)(image)
    except RuntimeError as e:
        assert "Solarize: the channel of image tensor does not match the requirement of operator" in str(e)

    try:
        image = np.random.randint(0, 256, (3, 300, 300)).astype(np.uint8)
        vision.Solarize(5)(image)
    except RuntimeError as e:
        assert "Solarize: the channel of image tensor does not match the requirement of operator" in str(e)


def test_input_type_errors():
    """
    Feature: Solarize op
    Description: Test that Solarize errors with bad input type
    Expectation: Passes the error check test
    """
    try:
        image = np.random.randint(0, 256, (300, 300, 3)).astype(np.uint32)
        vision.Solarize(5)(image)
    except RuntimeError as e:
        assert "Solarize: the data type of image tensor does not match the requirement of operator." in str(e)

    try:
        image = np.random.randint(0, 256, (300, 300, 3)).astype(np.uint64)
        vision.Solarize(5)(image)
    except RuntimeError as e:
        assert "Solarize: the data type of image tensor does not match the requirement of operator." in str(e)

    try:
        image = np.random.randint(0, 256, (300, 300, 3)).astype(np.float16)
        vision.Solarize(5)(image)
    except RuntimeError as e:
        assert "Solarize: the data type of image tensor does not match the requirement of operator." in str(e)

    try:
        image = np.random.randint(0, 256, (300, 300, 3)).astype(np.float64)
        vision.Solarize(5)(image)
    except RuntimeError as e:
        assert "Solarize: the data type of image tensor does not match the requirement of operator." in str(e)


def test_solarize_operation_01():
    """
    Feature: Solarize operation
    Description: Testing the normal functionality of the Solarize operator
    Expectation: The Output is equal to the expected output
    """
    # Solarize operator: normal testing, threshold=200, input image is numpy
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    ds1 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=True)
    ds2 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=True)
    ds2 = ds2.map(operations=vision.Solarize(200), input_columns=["image"])
    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True, num_epochs=1),
                            ds2.create_dict_iterator(output_numpy=True, num_epochs=1)):
        out_exp = abs((data1["image"] >= 200).astype(np.int32) * 255 - data1["image"])
        assert (out_exp == data2["image"]).all()

    # Solarize operator: normal testing, threshold=120.0, input image is numpy
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")

    ds1 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=False)
    op_list = [vision.Decode(to_pil=False)]
    ds1 = ds1.map(operations=op_list, input_columns=["image"])

    ds2 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=False)
    op_list = [vision.Decode(to_pil=False), vision.Solarize(120.0)]
    ds2 = ds2.map(operations=op_list, input_columns=["image"])

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True, num_epochs=1),
                            ds2.create_dict_iterator(output_numpy=True, num_epochs=1)):
        out_exp = abs((data1["image"] >= 120.0).astype(np.int32) * 255 - data1["image"])
        assert (out_exp == data2["image"]).all()

    # Solarize operator: normal testing, threshold=(100, 200), input image is PIL
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")

    ds1 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=False)
    ds1 = ds1.map(operations=vision.Decode(to_pil=True), input_columns=["image"])

    ds2 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=False)
    op_list = [vision.Decode(to_pil=True), vision.Solarize((100, 200))]
    ds2 = ds2.map(operations=op_list, input_columns=["image"])

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True, num_epochs=1),
                            ds2.create_dict_iterator(output_numpy=True, num_epochs=1)):
        out_exp_1 = (data1["image"] >= 100).astype(np.int32)
        out_exp_2 = (data1["image"] <= 200).astype(np.int32)
        out_exp = abs((out_exp_1 * out_exp_2).astype(np.int32) * 255 - data1["image"])
        assert (out_exp == data2["image"]).all()

    # Solarize operator: normal testing, threshold=(0, 255)
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")

    ds1 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=False)
    ds1 = ds1.map(operations=vision.Decode(to_pil=True), input_columns=["image"])

    ds2 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=False)
    op_list = [vision.Decode(to_pil=True), vision.Solarize(threshold=(0, 255))]
    ds2 = ds2.map(operations=op_list, input_columns=["image"])

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True, num_epochs=1),
                            ds2.create_dict_iterator(output_numpy=True, num_epochs=1)):
        out_exp_1 = (data1["image"] >= 0).astype(np.int32)
        out_exp_2 = (data1["image"] <= 255).astype(np.int32)
        out_exp = abs((out_exp_1 * out_exp_2).astype(np.int32) * 255 - data1["image"])
        assert (out_exp == data2["image"]).all()

    # Solarize operator: normal testing, input image is numpy <H,W,3>, threshold=0
    image = np.random.randint(0, 256, (20, 20, 3)).astype(np.uint8)
    op = vision.Solarize(0)
    out = op(image)
    assert out.shape == (20, 20, 3)
    assert out.dtype == np.uint8
    out_exp = abs((image >= 0).astype(np.int32) * 255 - image)
    assert (out_exp == out).all()

    # Solarize operator: normal testing, input image is numpy of type float32, threshold=(0.1, 0.2)
    image = (np.random.randint(0, 256, (20, 20, 3)) / 255.0).astype(np.float32)
    op = vision.Solarize((0.1, 0.2))
    out = op(image)
    assert out.shape == (20, 20, 3)
    assert out.dtype == np.float32
    out_exp_1 = image >= 0.1
    out_exp_2 = image <= 0.2
    out_exp = abs((out_exp_1 * out_exp_2).astype(np.float32) * 255.0 - image)
    assert (out == out_exp).all()

    # Solarize operator: normal testing, input image is numpy of type float32, threshold=(1, 2)
    image = (np.random.randint(0, 256, (20, 20, 3)) / 255.0).astype(np.float32)
    op = vision.Solarize((1, 2))
    out = op(image)
    assert out.shape == (20, 20, 3)
    assert out.dtype == np.float32
    out_exp_1 = image >= 1
    out_exp_2 = image <= 2
    out_exp = abs((out_exp_1 * out_exp_2).astype(np.float32) * 255.0 - image)
    assert (out == out_exp).all()


def test_solarize_operation_02():
    """
    Feature: Solarize operation
    Description: Testing the normal functionality of the Solarize operator
    Expectation: The Output is equal to the expected output
    """
    # Solarize operator: normal testing, input image is numpy of type float32, threshold=0.2666
    image = np.random.randint(0, 256, (20, 20, 3)) / 255.0
    image = image.astype(np.float32)
    op = vision.Solarize(0.2666)
    out = op(image)
    assert out.shape == (20, 20, 3)
    assert out.dtype == np.float32
    out_exp = abs((image >= 0.2666).astype(np.float32) * 255.0 - image)
    assert (out_exp == out).all()

    # Solarize operator: normal testing, input image is numpy of type float32, threshold=1.00001
    image = np.random.randint(0, 256, (20, 20, 3)) / 255.0
    image = image.astype(np.float32)
    op = vision.Solarize(1.00001)
    out = op(image)
    assert (out == image).all()

    # Solarize operator: normal testing, input image is numpy of type float64, threshold=255
    image = np.random.randint(0, 256, (20, 20, 3)) / 255.0
    image = image.astype(np.float64)
    op = vision.Solarize(255)
    out = op(image)
    assert (out == image).all()

    # Solarize operator: normal testing, input image is numpy of type float32, threshold=100
    image = np.random.randint(0, 256, (20, 20, 3)).astype(np.int32)
    op = vision.Solarize(100)
    out = op(image)
    out_exp = abs((image >= 100).astype(np.float32) * 255.0 - image)
    assert (out_exp == out).all()

    # Solarize operator: normal testing, input image is uint16, threshold=10.2
    image = np.random.randint(0, 256, (20, 20, 3)).astype(np.uint16)
    op = vision.Solarize(10.2)
    out = op(image)
    assert isinstance(out, np.ndarray)
    assert out.shape == (20, 20, 3)
    image = np.array(image)
    out_exp = abs((image >= 10.2).astype(np.int32) * 255 - image)
    assert (out_exp == out).all()

    # Solarize operator: normal testing, input image is PIL, threshold=10
    image = np.random.randint(0, 256, (20, 20, 3)).astype(np.uint8)
    image = vision.ToPIL()(image)
    op = vision.Solarize(10)
    out = op(image)
    assert isinstance(out, np.ndarray)
    assert out.shape == (20, 20, 3)
    image = np.array(image)
    out_exp = abs((image >= 10).astype(np.int32) * 255 - image)
    assert (out_exp == out).all()

    # Solarize operator: normal testing, input image is bmp image
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    with Image.open(image_path) as image:
        op = vision.Solarize(2)
        out = op(image)
        assert np.array(out).shape == np.array(image).shape
        image = np.array(image)
        out_exp = abs((image >= 2).astype(np.int32) * 255 - image)
        assert (out_exp == out).all()

    # Solarize operator: normal testing, input image is jpg image
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_path) as image:
        op = vision.Solarize(2)
        out = op(image)
        assert np.array(out).shape == np.array(image).shape
        image = np.array(image)
        out_exp = abs((image >= 2).astype(np.int32) * 255 - image)
        assert (out_exp == out).all()

    # Solarize operator: normal testing, input image is PNG image, mode=RGB
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with Image.open(image_path).convert("RGB") as image:
        op = vision.Solarize(2)
        out = op(image)
        assert np.array(out).shape == np.array(image).shape
        image = np.array(image)
        out_exp = abs((image >= 2).astype(np.int32) * 255 - image)
        assert (out_exp == out).all()

    # Solarize operator: normal testing, input image is gif image
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_path) as image:
        op = vision.Solarize(2)
        out = op(image)
        assert np.array(out).shape == np.array(image).shape
        image = np.array(image)
        out_exp = abs((image >= 2).astype(np.int32) * 255 - image)
        assert (out_exp == out).all()


def test_solarize_operation_03():
    """
    Feature: Solarize operation
    Description: Testing the normal functionality of the Solarize operator
    Expectation: The Output is equal to the expected output
    """
    # Solarize operator: normal testing, input image is gif image, mode=RGB
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_path).convert("RGB") as image:
        op = vision.Solarize(2)
        out = op(image)
        assert np.array(out).shape == np.array(image).shape
        image = np.array(image)
        out_exp = abs((image >= 2).astype(np.int32) * 255 - image)
        assert (out_exp == out).all()

    # Solarize operator: normal testing, input image is numpy <H,W,1>
    image = np.random.randint(0, 256, (20, 20, 1)).astype(np.uint8)
    op = vision.Solarize(2)
    out = op(image)
    assert np.array(out).shape == image.shape
    out_exp = abs((image >= 2).astype(np.int32) * 255 - image)
    assert (out_exp == out).all()

    # Solarize operator: normal testing, input image is a two-dimensional numpy array
    image = np.random.randint(0, 256, (20, 20)).astype(np.uint8)
    op = vision.Solarize(2)
    out = op(image)
    assert np.array(out).shape == image.shape
    out_exp = abs((image >= 2).astype(np.int32) * 255 - image)
    assert (out_exp == out).all()

    # Solarize operator: normal testing, threshold is list
    image = np.random.randint(0, 256, (20, 20)).astype(np.uint8)
    op = vision.Solarize(threshold=[1, 30])
    out = op(image)
    assert np.array(out).shape == image.shape
    out_exp_1 = image >= 1
    out_exp_2 = image <= 30
    out_exp = abs((out_exp_1 * out_exp_2).astype(np.float32) * 255.0 - image)
    assert (out_exp == out).all()


def test_solarize_exception_01():
    """
    Feature: Solarize operation
    Description: Testing the Solarize Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Solarize operator: exception testing, input image is PNG image, mode=RGBA
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with Image.open(image_path) as image:
        op = vision.Solarize(2)
        with pytest.raises(RuntimeError, match=r"Solarize: the channel of image tensor does "
                                               r"not match the requirement of operator. Expecting tensor in "
                                               r"channel of \(1, 3\). But got channel 4."):
            op(image)

    # Solarize operator: exception testing, input image is numpy <H,W,2>
    image = np.random.randint(0, 256, (20, 20, 2)).astype(np.uint8)
    op = vision.Solarize(2)
    with pytest.raises(RuntimeError, match=r"Solarize: the channel of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in "
                                           r"channel of \(1, 3\). But got channel 2."):
        op(image)

    # Solarize operator: exception testing, input image is numpy <H,W,4>
    image = np.random.randint(0, 256, (20, 20, 4)).astype(np.uint8)
    op = vision.Solarize(2)
    with pytest.raises(RuntimeError, match=r"Solarize: the channel of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in "
                                           r"channel of \(1, 3\). But got channel 4."):
        op(image)

    # Solarize operator: exception testing, input image is numpy <C,H,W>
    image = np.random.randint(0, 256, (3, 20, 20)).astype(np.uint8)
    op = vision.Solarize(2)
    with pytest.raises(RuntimeError, match=r"Solarize: the channel of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in "
                                           r"channel of \(1, 3\). But got channel 20."):
        op(image)

    # Solarize operator: exception testing, input image is a one-dimensional numpy array
    image = np.random.randint(0, 256, (20,)).astype(np.uint8)
    op = vision.Solarize(2)
    with pytest.raises(RuntimeError, match=r"Solarize: the dimension of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in dimension"
                                           r" of \(2, 3\), in shape of <H, W> or <H, W, C>. But got dimension 1. "
                                           r"You may need to perform Decode first."):
        op(image)

    # Solarize operator: exception testing, input image is a four-dimensional numpy array
    image = np.random.randint(0, 256, (20, 20, 3, 5)).astype(np.uint8)
    op = vision.Solarize(2)
    with pytest.raises(RuntimeError, match=r"Solarize: the dimension of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in dimension"
                                           r" of \(2, 3\), in shape of <H, W> or <H, W, C>. But got dimension 4."):
        op(image)

    # Solarize operator: exception testing, the input image is numpy bytes data
    image = np.random.randint(0, 256, (20, 20, 3)).astype("S")
    op = vision.Solarize(2)
    with pytest.raises(RuntimeError, match=r"Solarize: the data type of image tensor does not "
                                           r"match the requirement of operator. Expecting tensor in type "
                                           r"of \(bool, int8, uint8, int16, uint16, int32, float32, float64\). "
                                           r"But got type bytes."):
        op(image)

    # Solarize operator: exception testing, threshold is str
    with pytest.raises(TypeError, match=r"Argument threshold with value 1 is not of type \[<class 'float'>, "
                                        r"<class 'int'>, <class 'list'>, <class 'tuple'>\], but got <class 'str'>."):
        vision.Solarize(threshold='1')

    # Solarize operator: exception testing, threshold is None
    with pytest.raises(TypeError, match=r"Argument threshold with value None is not of type \[<class 'float'>, "
                                        r"<class 'int'>, <class 'list'>, <class 'tuple'>\], but "
                                        r"got <class 'NoneType'>."):
        vision.Solarize(threshold=None)

    # Solarize operator: exception testing, threshold is -0.00001
    with pytest.raises(ValueError, match=r"Input threshold\[0\] is not within the required interval of \[0, 255\]."):
        vision.Solarize(threshold=-0.00001)

    # Solarize operator: exception testing, threshold is 255.00001
    with pytest.raises(ValueError, match=r"Input threshold\[0\] is not within the required interval of \[0, 255\]."):
        vision.Solarize(threshold=255.00001)

    # Solarize operator: exception testing, the threshold tuple element value is -0.00001
    with pytest.raises(ValueError, match=r"Input threshold\[0\] is not within the required interval of \[0, 255\]."):
        vision.Solarize(threshold=(-0.00001, 255))

    # Solarize operator: exception testing, the threshold tuple element value is 255.00001
    with pytest.raises(ValueError, match=r"Input threshold\[1\] is not within the required interval of \[0, 255\]."):
        vision.Solarize(threshold=(2, 255.00001))

    # Solarize operator: exception testing, threshold tuple element type error
    with pytest.raises(TypeError, match=r"Argument threshold\[0\] with value True is not of type \(<class 'float'>, "
                                        r"<class 'int'>\), but got <class 'bool'>."):
        vision.Solarize(threshold=(True, 255))

    # Solarize operator: exception testing, the length of the Threshold parameter is 2
    with pytest.raises(TypeError, match=r"threshold must be a single number or sequence of two numbers."):
        vision.Solarize(threshold=(0.2, 2, 255))

    # Solarize operator: exception testing, the length of the Threshold parameter is 1
    with pytest.raises(TypeError, match=r"threshold must be a single number or sequence of two numbers."):
        vision.Solarize(threshold=(255,))


def test_solarize_exception_02():
    """
    Feature: Solarize operation
    Description: Testing the Solarize Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Solarize operator: exception testing, the length of the Threshold parameter is 0
    with pytest.raises(TypeError, match=r"threshold must be a single number or sequence of two numbers."):
        vision.Solarize(threshold=())

    # Solarize operator: exception testing, threshold=(200, 2)
    with pytest.raises(ValueError, match=r"threshold must be in order of \(min, max\)."):
        vision.Solarize(threshold=(200, 2))

    # Solarize operator: exception testing, input image is int
    image = 100
    op = vision.Solarize(2)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'int'>."):
        op(image)

    # Solarize operator: exception testing, input image is str
    image = '100'
    op = vision.Solarize(2)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'str'>."):
        op(image)

    # Solarize operator: exception testing, input image is float
    image = 100.0
    op = vision.Solarize(2)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'float'>."):
        op(image)

    # Solarize operator: exception testing, input image is list
    image = np.random.randint(0, 256, (10, 10, 3)).tolist()
    op = vision.Solarize(2)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        op(image)

    # Solarize operator: exception testing, input image is tuple
    image = (1, 2, 3)
    op = vision.Solarize(2)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>."):
        op(image)

    # Solarize operator: exception testing, input image is ms.Tensor
    image = ms.Tensor(np.random.randint(0, 256, (10, 10, 3)))
    op = vision.Solarize(2)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, "
                                        "got <class 'mindspore.common.tensor.Tensor'>."):
        op(image)


if __name__ == "__main__":
    test_solarize_basic()
    test_solarize_mnist(plot=False)
    test_solarize_errors()
    test_input_shape_errors()
    test_input_type_errors()
    test_solarize_operation_01()
    test_solarize_operation_02()
    test_solarize_operation_03()
    test_solarize_exception_01()
    test_solarize_exception_02()
