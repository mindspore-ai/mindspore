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
Testing RandomGrayscale op in DE
"""
import numpy as np

import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset as ds
from mindspore import log as logger
from util import save_and_check_md5, visualize_list, \
    config_get_set_seed, config_get_set_num_parallel_workers

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_random_grayscale_valid_prob(plot=False):
    """
    Test RandomGrayscale Op: valid input, expect to pass
    """
    logger.info("test_random_grayscale_valid_prob")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms1 = [
        py_vision.Decode(),
        # Note: prob is 1 so the output should always be grayscale images
        py_vision.RandomGrayscale(1),
        py_vision.ToTensor()
    ]
    transform1 = mindspore.dataset.transforms.py_transforms.Compose(transforms1)
    data1 = data1.map(operations=transform1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms2 = [
        py_vision.Decode(),
        py_vision.ToTensor()
    ]
    transform2 = mindspore.dataset.transforms.py_transforms.Compose(transforms2)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    image_gray = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_gray.append(image1)
        image.append(image2)
    if plot:
        visualize_list(image, image_gray)


def test_random_grayscale_input_grayscale_images():
    """
    Test RandomGrayscale Op: valid parameter with grayscale images as input, expect to pass
    """
    logger.info("test_random_grayscale_input_grayscale_images")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms1 = [
        py_vision.Decode(),
        py_vision.Grayscale(1),
        # Note: If the input images is grayscale image with 1 channel.
        py_vision.RandomGrayscale(0.5),
        py_vision.ToTensor()
    ]
    transform1 = mindspore.dataset.transforms.py_transforms.Compose(transforms1)
    data1 = data1.map(operations=transform1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms2 = [
        py_vision.Decode(),
        py_vision.ToTensor()
    ]
    transform2 = mindspore.dataset.transforms.py_transforms.Compose(transforms2)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    image_gray = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_gray.append(image1)
        image.append(image2)

        assert len(image1.shape) == 3
        assert image1.shape[2] == 1
        assert len(image2.shape) == 3
        assert image2.shape[2] == 3

    # Restore config
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_grayscale_md5_valid_input():
    """
    Test RandomGrayscale with md5 comparison: valid parameter, expect to pass
    """
    logger.info("test_random_grayscale_md5_valid_input")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        py_vision.Decode(),
        py_vision.RandomGrayscale(0.8),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    # Check output images with md5 comparison
    filename = "random_grayscale_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_grayscale_md5_no_param():
    """
    Test RandomGrayscale with md5 comparison: no parameter given, expect to pass
    """
    logger.info("test_random_grayscale_md5_no_param")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        py_vision.Decode(),
        py_vision.RandomGrayscale(),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    # Check output images with md5 comparison
    filename = "random_grayscale_02_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_grayscale_invalid_param():
    """
    Test RandomGrayscale: invalid parameter given, expect to raise error
    """
    logger.info("test_random_grayscale_invalid_param")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    try:
        transforms = [
            py_vision.Decode(),
            py_vision.RandomGrayscale(1.5),
            py_vision.ToTensor()
        ]
        transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
        data = data.map(operations=transform, input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(e)


if __name__ == "__main__":
    test_random_grayscale_valid_prob(True)
    test_random_grayscale_input_grayscale_images()
    test_random_grayscale_md5_valid_input()
    test_random_grayscale_md5_no_param()
    test_random_grayscale_invalid_param()
