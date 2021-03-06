# Copyright 2020 Huawei Technologies Co., Ltd
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
Testing RandomColor op in DE
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.c_transforms as vision
import mindspore.dataset.vision.py_transforms as F
from mindspore import log as logger
from util import visualize_list, diff_mse, save_and_check_md5, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = "../data/dataset/testImageNetData/train/"

C_DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
C_SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

MNIST_DATA_DIR = "../data/dataset/testMnistData"

GENERATE_GOLDEN = False


def test_random_color_py(degrees=(0.1, 1.9), plot=False):
    """
    Test Python RandomColor
    """
    logger.info("Test RandomColor")

    # Original Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = mindspore.dataset.transforms.py_transforms.Compose([F.Decode(),
                                                                              F.Resize((224, 224)),
                                                                              F.ToTensor()])

    ds_original = data.map(operations=transforms_original, input_columns="image")

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = np.transpose(image.asnumpy(), (0, 2, 3, 1))
        else:
            images_original = np.append(images_original,
                                        np.transpose(image.asnumpy(), (0, 2, 3, 1)),
                                        axis=0)

            # Random Color Adjusted Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_random_color = mindspore.dataset.transforms.py_transforms.Compose([F.Decode(),
                                                                                  F.Resize((224, 224)),
                                                                                  F.RandomColor(degrees=degrees),
                                                                                  F.ToTensor()])

    ds_random_color = data.map(operations=transforms_random_color, input_columns="image")

    ds_random_color = ds_random_color.batch(512)

    for idx, (image, _) in enumerate(ds_random_color):
        if idx == 0:
            images_random_color = np.transpose(image.asnumpy(), (0, 2, 3, 1))
        else:
            images_random_color = np.append(images_random_color,
                                            np.transpose(image.asnumpy(), (0, 2, 3, 1)),
                                            axis=0)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_color[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))

    if plot:
        visualize_list(images_original, images_random_color)


def test_random_color_c(degrees=(0.1, 1.9), plot=False, run_golden=True):
    """
    Test Cpp RandomColor
    """
    logger.info("test_random_color_op")

    original_seed = config_get_set_seed(10)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Decode with rgb format set to True
    data1 = ds.TFRecordDataset(C_DATA_DIR, C_SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = ds.TFRecordDataset(C_DATA_DIR, C_SCHEMA_DIR, columns_list=["image"], shuffle=False)

    # Serialize and Load dataset requires using vision.Decode instead of vision.Decode().
    if degrees is None:
        c_op = vision.RandomColor()
    else:
        c_op = vision.RandomColor(degrees)

    data1 = data1.map(operations=[vision.Decode()], input_columns=["image"])
    data2 = data2.map(operations=[vision.Decode(), c_op], input_columns=["image"])

    image_random_color_op = []
    image = []

    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        actual = item1["image"]
        expected = item2["image"]
        image.append(actual)
        image_random_color_op.append(expected)

    if run_golden:
        # Compare with expected md5 from images
        filename = "random_color_op_02_result.npz"
        save_and_check_md5(data2, filename, generate_golden=GENERATE_GOLDEN)

    if plot:
        visualize_list(image, image_random_color_op)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_random_color_py_md5():
    """
    Test Python RandomColor with md5 check
    """
    logger.info("Test RandomColor with md5 check")
    original_seed = config_get_set_seed(10)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms = mindspore.dataset.transforms.py_transforms.Compose([F.Decode(),
                                                                     F.RandomColor((2.0, 2.5)),
                                                                     F.ToTensor()])

    data = data.map(operations=transforms, input_columns="image")
    # Compare with expected md5 from images
    filename = "random_color_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_compare_random_color_op(degrees=None, plot=False):
    """
    Compare Random Color op in Python and Cpp
    """

    logger.info("test_random_color_op")

    original_seed = config_get_set_seed(5)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Decode with rgb format set to True
    data1 = ds.TFRecordDataset(C_DATA_DIR, C_SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = ds.TFRecordDataset(C_DATA_DIR, C_SCHEMA_DIR, columns_list=["image"], shuffle=False)

    if degrees is None:
        c_op = vision.RandomColor()
        p_op = F.RandomColor()
    else:
        c_op = vision.RandomColor(degrees)
        p_op = F.RandomColor(degrees)

    transforms_random_color_py = mindspore.dataset.transforms.py_transforms.Compose(
        [lambda img: img.astype(np.uint8), F.ToPIL(),
         p_op, np.array])

    data1 = data1.map(operations=[vision.Decode(), c_op], input_columns=["image"])
    data2 = data2.map(operations=[vision.Decode()], input_columns=["image"])
    data2 = data2.map(operations=transforms_random_color_py, input_columns=["image"])

    image_random_color_op = []
    image = []

    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        actual = item1["image"]
        expected = item2["image"]
        image_random_color_op.append(actual)
        image.append(expected)
        assert actual.shape == expected.shape
        mse = diff_mse(actual, expected)
        logger.info("MSE= {}".format(str(np.mean(mse))))

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

    if plot:
        visualize_list(image, image_random_color_op)


def test_random_color_c_errors():
    """
    Test that Cpp RandomColor errors with bad input
    """
    with pytest.raises(TypeError) as error_info:
        vision.RandomColor((12))
    assert "degrees must be either a tuple or a list." in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        vision.RandomColor(("col", 3))
    assert "Argument degrees[0] with value col is not of type (<class 'int'>, <class 'float'>)." in str(
        error_info.value)

    with pytest.raises(ValueError) as error_info:
        vision.RandomColor((0.9, 0.1))
    assert "degrees should be in (min,max) format. Got (max,min)." in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        vision.RandomColor((0.9,))
    assert "degrees must be a sequence with length 2." in str(error_info.value)

    # RandomColor Cpp Op will fail with one channel input
    mnist_ds = ds.MnistDataset(dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)
    mnist_ds = mnist_ds.map(operations=vision.RandomColor(), input_columns="image")

    with pytest.raises(RuntimeError) as error_info:
        for _ in enumerate(mnist_ds):
            pass
    assert "image shape is not <H,W,C> or channel is not 3" in str(error_info.value)


if __name__ == "__main__":
    test_random_color_py()
    test_random_color_py(plot=True)
    test_random_color_py(degrees=(2.0, 2.5), plot=True)  # Test with degree values that show more obvious transformation
    test_random_color_py_md5()

    test_random_color_c()
    test_random_color_c(plot=True)
    test_random_color_c(degrees=(2.0, 2.5), plot=True,
                        run_golden=False)  # Test with degree values that show more obvious transformation
    test_random_color_c(degrees=(0.1, 0.1), plot=True, run_golden=False)
    test_compare_random_color_op(plot=True)
    test_random_color_c_errors()
