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
Testing Equalize op in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import visualize_list, visualize_one_channel_dataset, diff_mse, save_and_check_md5

DATA_DIR = "../data/dataset/testImageNetData/train/"
MNIST_DATA_DIR = "../data/dataset/testMnistData"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

GENERATE_GOLDEN = False


def test_equalize_callable():
    """
    Feature: Equalize Op
    Description: Test op in eager mode
    Expectation: Output image shape from op is verified
    """
    logger.info("Test Equalize is callable")
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = vision.Decode()(img)
    img = vision.Equalize()(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    assert img.shape == (2268, 4032, 3)




def test_equalize_c(plot=False):
    """
    Feature: Equalize Op
    Description: Test C++ implementation
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    logger.info("Test Equalize C++ implementation")

    # Original Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = [vision.Decode(), vision.Resize(size=[224, 224])]

    ds_original = data_set.map(operations=transforms_original, input_columns="image")

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image.asnumpy()
        else:
            images_original = np.append(images_original,
                                        image.asnumpy(),
                                        axis=0)

    # Equalize Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transform_equalize = [vision.Decode(), vision.Resize(size=[224, 224]),
                          vision.Equalize()]

    ds_equalize = data_set.map(operations=transform_equalize, input_columns="image")

    ds_equalize = ds_equalize.batch(512)

    for idx, (image, _) in enumerate(ds_equalize):
        if idx == 0:
            images_equalize = image.asnumpy()
        else:
            images_equalize = np.append(images_equalize,
                                        image.asnumpy(),
                                        axis=0)
    if plot:
        visualize_list(images_original, images_equalize)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_equalize[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))




def test_equalize_one_channel():
    """
    Feature: Equalize Op
    Description: Test Equalize C++ implementation with one channel images
    Expectation: Invalid input is detected
    """
    logger.info("Test Equalize C++ implementation with One Channel Images")

    c_op = vision.Equalize()

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[vision.Decode(), vision.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])

        data_set.map(operations=c_op, input_columns="image")

    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "The shape" in str(e)


def test_equalize_mnist_c(plot=False):
    """
    Feature: Equalize Op
    Description: Test Equalize C++ implementation with MNIST dataset (Grayscale images)
    Expectation: Dataset pipeline runs successfully and md5 results are verified
    """
    logger.info("Test Equalize C++ implementation with MNIST Images")
    data_set = ds.MnistDataset(dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)
    ds_equalize_c = data_set.map(operations=vision.Equalize(), input_columns="image")
    ds_orig = ds.MnistDataset(dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)

    images = []
    images_trans = []
    labels = []
    for _, (data_orig, data_trans) in enumerate(zip(ds_orig, ds_equalize_c)):
        image_orig, label_orig = data_orig
        image_trans, _ = data_trans
        images.append(image_orig.asnumpy())
        labels.append(label_orig.asnumpy())
        images_trans.append(image_trans.asnumpy())

    # Compare with expected md5 from images
    filename = "equalize_mnist_result_c.npz"
    save_and_check_md5(ds_equalize_c, filename, generate_golden=GENERATE_GOLDEN)

    if plot:
        visualize_one_channel_dataset(images, images_trans, labels)




def test_equalize_md5_c():
    """
    Feature: Equalize Op
    Description: Test C++ implementation with md5 check
    Expectation: Dataset pipeline runs successfully and md5 results are verified
    """
    logger.info("Test Equalize C++ implementation with md5 check")

    # Generate dataset
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_equalize = [vision.Decode(),
                           vision.Resize(size=[224, 224]),
                           vision.Equalize(),
                           vision.ToTensor()]

    data = data_set.map(operations=transforms_equalize, input_columns="image")
    # Compare with expected md5 from images
    filename = "equalize_01_result_c_unified.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)




def test_equalize_exception_01():
    """
    Feature: Equalize operation
    Description: Testing the Equalize Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Equalize operator: Test input is 3d list
    image = list(np.random.randn(128, 128, 3).astype(np.uint8))
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>"):
        equalize_op = vision.Equalize()
        _ = equalize_op(image)

    # Equalize operator: Test input is 2d tuple
    image = tuple(np.random.randint(0, 255, (20, 10)).astype(np.uint8))
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>"):
        equalize_op = vision.Equalize()
        _ = equalize_op(image)

    # Equalize operator: Test input is 4-d image
    image = np.random.randn(10, 468, 368, 3).astype(np.uint8)
    equalize_op = vision.Equalize()
    with pytest.raises(RuntimeError, match="Equalize: image rank should be 2 or 3,  but got: 4"):
        equalize_op(image)

    # Equalize operator: Test image is png
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with Image.open(image_png) as image:
        equalize_op = vision.Equalize()
        with pytest.raises(OSError, match="not supported for .*"):
            equalize_op(image)

    # Equalize operator: Test input is 1-d numpy data
    image = np.random.randn(200,).astype(np.uint8)
    equalize_op = vision.Equalize()
    with pytest.raises(RuntimeError, match="Equalize: image rank should be 2 or 3,  but got: 1"):
        equalize_op(image)

    # Equalize operator: Test input is 4 channel numpy array
    image = np.random.randn(128, 128, 4).astype(np.uint8)
    equalize_op = vision.Equalize()
    with pytest.raises(RuntimeError,
                       match="Equalize: channel of input image should be 1 or 3, but got: 4"):
        equalize_op(image)

    # Equalize operator: Test input is 3d numpy list
    image = np.random.randn(128, 128, 3).astype(np.uint8).tolist()
    equalize_op = vision.Equalize()
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>"):
        equalize_op(image)

    # Equalize operator: Test input is int
    image = 10
    equalize_op = vision.Equalize()
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'int'>"):
        equalize_op(image)

    # Equalize operator: Test input is tuple
    image = (10,)
    equalize_op = vision.Equalize()
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>"):
        equalize_op(image)

    # Equalize operator: Test no image is transferred
    equalize_op = vision.Equalize()
    with pytest.raises(RuntimeError, match="Input Tensor is not valid."):
        equalize_op()

    # Equalize operator: Test one more parameter
    with pytest.raises(TypeError, match="takes 1 positional argument but 2 were given"):
        vision.Equalize(1)
