# Copyright 2022 Huawei Technologies Co., Ltd
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
from PIL import Image, ImageOps
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import visualize_list, config_get_set_seed, config_get_set_num_parallel_workers, \
    visualize_one_channel_dataset, visualize_image, diff_mse

GENERATE_GOLDEN = False

MNIST_DATA_DIR = "../data/dataset/testMnistData"
DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


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


if __name__ == "__main__":
    test_solarize_basic()
    test_solarize_mnist(plot=False)
    test_solarize_errors()
    test_input_shape_errors()
    test_input_type_errors()
