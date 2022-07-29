# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

Testing RandomLighting op in DE
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import visualize_list, diff_mse, save_and_check_md5_pil, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = "../data/dataset/testImageNetData/train/"
MNIST_DATA_DIR = "../data/dataset/testMnistData"

GENERATE_GOLDEN = False


def test_random_lighting_py(alpha=1, plot=False):
    """
    Feature: RandomLighting
    Description: Test RandomLighting Python implementation
    Expectation: Equal results
    """
    logger.info("Test RandomLighting Python implementation")

    # Original Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = mindspore.dataset.transforms.Compose([vision.Decode(True),
                                                                vision.Resize((224, 224)),
                                                                vision.ToTensor()])

    ds_original = data.map(
        operations=transforms_original, input_columns="image")

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        if idx == 0:
            images_original = np.transpose(image, (0, 2, 3, 1))
        else:
            images_original = np.append(
                images_original, np.transpose(image, (0, 2, 3, 1)), axis=0)

    # Random Lighting Adjusted Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    alpha = alpha if alpha is not None else 0.05
    py_op = vision.RandomLighting(alpha)

    transforms_random_lighting = mindspore.dataset.transforms.Compose([vision.Decode(True),
                                                                       vision.Resize((224, 224)),
                                                                       py_op,
                                                                       vision.ToTensor()])
    ds_random_lighting = data.map(
        operations=transforms_random_lighting, input_columns="image")

    ds_random_lighting = ds_random_lighting.batch(512)

    for idx, (image, _) in enumerate(ds_random_lighting.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        if idx == 0:
            images_random_lighting = np.transpose(image, (0, 2, 3, 1))
        else:
            images_random_lighting = np.append(
                images_random_lighting, np.transpose(image, (0, 2, 3, 1)), axis=0)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_lighting[i], images_original[i])

    logger.info("MSE= {}".format(str(np.mean(mse))))

    if plot:
        visualize_list(images_original, images_random_lighting)


def test_random_lighting_py_md5():
    """
    Feature: RandomLighting
    Description: Test RandomLighting Python implementation with md5 comparison
    Expectation: Same MD5
    """
    logger.info("Test RandomLighting Python implementation with md5 comparison")
    original_seed = config_get_set_seed(140)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # define map operations
    transforms = [
        vision.Decode(True),
        vision.Resize((224, 224)),
        vision.RandomLighting(1),
        vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)

    #  Generate dataset
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data = data.map(operations=transform, input_columns=["image"])

    # check results with md5 comparison
    filename = "random_lighting_py_01_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_lighting_c(alpha=1, plot=False):
    """
    Feature: RandomLighting
    Description: Test RandomLighting cpp op
    Expectation: Equal results from Mindspore and benchmark
    """
    logger.info("Test RandomLighting cpp op")
    # Original Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = [vision.Decode(), vision.Resize((224, 224))]

    ds_original = data.map(
        operations=transforms_original, input_columns="image")

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        if idx == 0:
            images_original = image
        else:
            images_original = np.append(images_original, image, axis=0)

    # Random Lighting Adjusted Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    alpha = alpha if alpha is not None else 0.05
    c_op = vision.RandomLighting(alpha)

    transforms_random_lighting = [
        vision.Decode(), vision.Resize((224, 224)), c_op]

    ds_random_lighting = data.map(
        operations=transforms_random_lighting, input_columns="image")

    ds_random_lighting = ds_random_lighting.batch(512)

    for idx, (image, _) in enumerate(ds_random_lighting.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        if idx == 0:
            images_random_lighting = image
        else:
            images_random_lighting = np.append(
                images_random_lighting, image, axis=0)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_lighting[i], images_original[i])

    logger.info("MSE= {}".format(str(np.mean(mse))))

    if plot:
        visualize_list(images_original, images_random_lighting)


def test_random_lighting_c_py(alpha=1, plot=False):
    """
    Feature: RandomLighting
    Description: Test Random Lighting Cpp and Python Op
    Expectation: Equal results from Cpp and Python
    """
    logger.info("Test RandomLighting Cpp and python Op")

    # RandomLighting Images
    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data = data.map(operations=[vision.Decode(), vision.Resize(
        (200, 300))], input_columns=["image"])

    python_op = vision.RandomLighting(alpha)
    c_op = vision.RandomLighting(alpha)

    transforms_op = mindspore.dataset.transforms.Compose([lambda img: vision.ToPIL()(img.astype(np.uint8)),
                                                          python_op,
                                                          np.array])

    ds_random_lighting_py = data.map(
        operations=transforms_op, input_columns="image")

    ds_random_lighting_py = ds_random_lighting_py.batch(512)

    for idx, (image, _) in enumerate(ds_random_lighting_py.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        if idx == 0:
            images_random_lighting_py = image

        else:
            images_random_lighting_py = np.append(
                images_random_lighting_py, image, axis=0)

    data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data = data.map(operations=[vision.Decode(), vision.Resize(
        (200, 300))], input_columns=["image"])

    ds_images_random_lighting_c = data.map(
        operations=c_op, input_columns="image")

    ds_random_lighting_c = ds_images_random_lighting_c.batch(512)

    for idx, (image, _) in enumerate(ds_random_lighting_c.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        if idx == 0:
            images_random_lighting_c = image
        else:
            images_random_lighting_c = np.append(
                images_random_lighting_c, image, axis=0)

    num_samples = images_random_lighting_c.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_lighting_c[i],
                          images_random_lighting_py[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))
    if plot:
        visualize_list(images_random_lighting_c,
                       images_random_lighting_py, visualize_mode=2)


def test_random_lighting_invalid_params():
    """
    Feature: RandomLighting
    Description: Test RandomLighting with invalid input parameters
    Expectation: Throw correct error and message
    """
    logger.info("Test RandomLighting with invalid input parameters.")
    with pytest.raises(ValueError) as error_info:
        data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data = data.map(operations=[vision.Decode(), vision.Resize((224, 224)),
                                    vision.RandomLighting(-2)], input_columns=["image"])
    assert "Input alpha is not within the required interval of [0, 16777216]." in str(
        error_info.value)

    with pytest.raises(TypeError) as error_info:
        data = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data = data.map(operations=[vision.Decode(), vision.Resize((224, 224)),
                                    vision.RandomLighting('1')], input_columns=["image"])
    err_msg = "Argument alpha with value 1 is not of type [<class 'float'>, <class 'int'>], but got <class 'str'>."
    assert err_msg in str(error_info.value)


if __name__ == "__main__":
    test_random_lighting_py()
    test_random_lighting_py_md5()
    test_random_lighting_c()
    test_random_lighting_c_py()
    test_random_lighting_invalid_params()
