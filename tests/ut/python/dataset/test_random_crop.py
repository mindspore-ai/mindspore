# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
Testing RandomCrop op in DE
"""
import numpy as np
import pytest
from PIL import Image

import mindspore.dataset.transforms as ops
import mindspore.dataset.vision as vision
import mindspore.dataset.vision.utils as mode
import mindspore.dataset as ds
from mindspore import log as logger
from util import save_and_check_md5, save_and_check_md5_pil, visualize_list, config_get_set_seed, \
    config_get_set_num_parallel_workers, diff_mse

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_random_crop_op_c(plot=False):
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Cpp implementation
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_op_c")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    random_crop_op = vision.RandomCrop([512, 512], [200, 200, 200, 200])
    decode_op = vision.Decode()

    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_crop_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    image_cropped = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = item1["image"]
        image2 = item2["image"]
        image_cropped.append(image1)
        image.append(image2)
    if plot:
        visualize_list(image, image_cropped)


def test_random_crop_op_py(plot=False):
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Python transformations
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_op_py")
    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms1 = [
        vision.Decode(True),
        vision.RandomCrop([512, 512], [200, 200, 200, 200]),
        vision.ToTensor()
    ]
    transform1 = ops.Compose(transforms1)
    data1 = data1.map(operations=transform1, input_columns=["image"])
    # Second dataset
    # Second dataset for comparison
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms2 = [
        vision.Decode(True),
        vision.ToTensor()
    ]
    transform2 = ops.Compose(transforms2)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    crop_images = []
    original_images = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        crop = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        original = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        crop_images.append(crop)
        original_images.append(original)
    if plot:
        visualize_list(original_images, crop_images)


def test_random_crop_01_c():
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Cpp implementation where size is a single integer
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_01_c")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: If size is an int, a square crop of size (size, size) is returned.
    random_crop_op = vision.RandomCrop(512)
    decode_op = vision.Decode()
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_crop_op, input_columns=["image"])

    filename = "random_crop_01_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_crop_01_py():
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Python implementation where size is a single integer
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_01_py")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: If size is an int, a square crop of size (size, size) is returned.
    transforms = [
        vision.Decode(True),
        vision.RandomCrop(512),
        vision.ToTensor()
    ]
    transform = ops.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    filename = "random_crop_01_py_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_crop_02_c():
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Cpp implementation where size is a list/tuple with length 2
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_02_c")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: If size is a sequence of length 2, it should be (height, width).
    random_crop_op = vision.RandomCrop([512, 375])
    decode_op = vision.Decode()
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_crop_op, input_columns=["image"])

    filename = "random_crop_02_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_crop_02_py():
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Python implementation where size is a list/tuple with length 2
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_02_py")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: If size is a sequence of length 2, it should be (height, width).
    transforms = [
        vision.Decode(True),
        vision.RandomCrop([512, 375]),
        vision.ToTensor()
    ]
    transform = ops.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    filename = "random_crop_02_py_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_crop_03_c():
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Cpp implementation where input image size == crop size
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_03_c")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The size of the image is 4032*2268
    random_crop_op = vision.RandomCrop([2268, 4032])
    decode_op = vision.Decode()
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_crop_op, input_columns=["image"])

    filename = "random_crop_03_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_crop_03_py():
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Python implementation where input image size == crop size
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_03_py")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The size of the image is 4032*2268
    transforms = [
        vision.Decode(True),
        vision.RandomCrop([2268, 4032]),
        vision.ToTensor()
    ]
    transform = ops.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    filename = "random_crop_03_py_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_crop_04_c():
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Cpp implementation where input image size < crop size
    Expectation: Error is raised as expected
    """
    logger.info("test_random_crop_04_c")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The size of the image is 4032*2268
    random_crop_op = vision.RandomCrop([2268, 4033])
    decode_op = vision.Decode()
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_crop_op, input_columns=["image"])
    try:
        data.create_dict_iterator(num_epochs=1).__next__()
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "crop size is bigger than the image dimensions" in str(e)


def test_random_crop_04_py():
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Python implementation where input image size < crop size
    Expectation: Error is raised as expected
    """
    logger.info("test_random_crop_04_py")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The size of the image is 4032*2268
    transforms = [
        vision.Decode(True),
        vision.RandomCrop([2268, 4033]),
        vision.ToTensor()
    ]
    transform = ops.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])
    try:
        data.create_dict_iterator(num_epochs=1).__next__()
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Crop size" in str(e)


def test_random_crop_05_c():
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Cpp implementation where input image size < crop size, pad_if_needed is enabled
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_05_c")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The size of the image is 4032*2268
    random_crop_op = vision.RandomCrop([2268, 4033], [200, 200, 200, 200], pad_if_needed=True)
    decode_op = vision.Decode()
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_crop_op, input_columns=["image"])

    filename = "random_crop_05_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_crop_05_py():
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Python implementation input image size < crop size, pad_if_needed is enabled
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_05_py")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The size of the image is 4032*2268
    transforms = [
        vision.Decode(True),
        vision.RandomCrop([2268, 4033], [200, 200, 200, 200], pad_if_needed=True),
        vision.ToTensor()
    ]
    transform = ops.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    filename = "random_crop_05_py_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_crop_06_c():
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Cpp implementation with invalid size
    Expectation: Error is raised as expected
    """
    logger.info("test_random_crop_06_c")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    try:
        # Note: if size is neither an int nor a list of length 2, an exception will raise
        random_crop_op = vision.RandomCrop([512, 512, 375])
        decode_op = vision.Decode()
        data = data.map(operations=decode_op, input_columns=["image"])
        data = data.map(operations=random_crop_op, input_columns=["image"])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Size should be a single integer" in str(e)


def test_random_crop_06_py():
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Python implementation with invalid size
    Expectation: Error is raised as expected
    """
    logger.info("test_random_crop_06_py")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    try:
        # Note: if size is neither an int nor a list of length 2, an exception will raise
        transforms = [
            vision.Decode(True),
            vision.RandomCrop([512, 512, 375]),
            vision.ToTensor()
        ]
        transform = ops.Compose(transforms)
        data = data.map(operations=transform, input_columns=["image"])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Size should be a single integer" in str(e)


def test_random_crop_07_c():
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Cpp implementation with padding_mode is Border.CONSTANT, fill_value is 255
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_07_c")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The padding_mode is default as Border.CONSTANT and set filling color to be white.
    random_crop_op = vision.RandomCrop(512, [200, 200, 200, 200], fill_value=(255, 255, 255))
    decode_op = vision.Decode()
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_crop_op, input_columns=["image"])

    filename = "random_crop_07_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_crop_07_py():
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Python implementation with padding_mode is Border.CONSTANT, fill_value is 255
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_07_py")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The padding_mode is default as Border.CONSTANT and set filling color to be white.
    transforms = [
        vision.Decode(True),
        vision.RandomCrop(512, [200, 200, 200, 200], fill_value=(255, 255, 255)),
        vision.ToTensor()
    ]
    transform = ops.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    filename = "random_crop_07_py_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_crop_08_c():
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Cpp implementation with padding_mode is Border.EDGE
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_08_c")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The padding_mode is Border.EDGE.
    random_crop_op = vision.RandomCrop(512, [200, 200, 200, 200], padding_mode=mode.Border.EDGE)
    decode_op = vision.Decode()
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_crop_op, input_columns=["image"])

    filename = "random_crop_08_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_crop_08_py():
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op in Python implementation with padding_mode is Border.EDGE
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_08_py")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Note: The padding_mode is Border.EDGE.
    transforms = [
        vision.Decode(True),
        vision.RandomCrop(512, [200, 200, 200, 200], padding_mode=mode.Border.EDGE),
        vision.ToTensor()
    ]
    transform = ops.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    filename = "random_crop_08_py_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_crop_09():
    """
    Feature: RandomCrop
    Description: Test RandomCrop with invalid image format
    Expectation: RuntimeError is raised
    """

    logger.info("test_random_crop_09")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.ToTensor(),
        # Note: Input is wrong image format
        vision.RandomCrop(512)
    ]
    transform = ops.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])
    with pytest.raises(RuntimeError) as error_info:
        for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
    error_msg = "Expecting tensor in channel of (1, 3)"
    assert error_msg in str(error_info.value)


def test_random_crop_10():
    """
    Feature: RandomCrop
    Description: Test Py RandomCrop with grayscale/binary image
    Expectation: The dataset is processed as expected
    """
    path = "../data/dataset/apple.jpg"
    image_list = [Image.open(path), Image.open(path).convert('1'), Image.open(path).convert('L')]
    for image in image_list:
        _ = vision.RandomCrop((28))(image)


def test_random_crop_comp(plot=False):
    """
    Feature: RandomCrop op
    Description: Test RandomCrop and compare between Python and Cpp image augmentation
    Expectation: Resulting datasets from both op are the same as expected
    """
    logger.info("Test RandomCrop with c_transform and py_transform comparison")
    cropped_size = 512

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    random_crop_op = vision.RandomCrop(cropped_size)
    decode_op = vision.Decode()
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_crop_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.RandomCrop(cropped_size),
        vision.ToTensor()
    ]
    transform = ops.Compose(transforms)
    data2 = data2.map(operations=transform, input_columns=["image"])

    image_c_cropped = []
    image_py_cropped = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        c_image = item1["image"]
        py_image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_c_cropped.append(c_image)
        image_py_cropped.append(py_image)
    if plot:
        visualize_list(image_c_cropped, image_py_cropped, visualize_mode=2)


def test_random_crop_09_c():
    """
    Feature: RandomCrop op
    Description: Test RandomCrop Op with different fields
    Expectation: The dataset is processed as expected
    """
    logger.info("Test RandomCrop with different fields.")

    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=ops.Duplicate(), input_columns=["image"],
                    output_columns=["image", "image_copy"])
    random_crop_op = vision.RandomCrop([512, 512], [200, 200, 200, 200])
    decode_op = vision.Decode()

    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=decode_op, input_columns=["image_copy"])
    data = data.map(operations=random_crop_op, input_columns=["image", "image_copy"])

    num_iter = 0
    for data1 in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = data1["image"]
        image_copy = data1["image_copy"]
        mse = diff_mse(image, image_copy)
        assert mse == 0
        num_iter += 1


def test_random_crop_high_dimensions():
    """
    Feature: RandomCrop
    Description: Use randomly generated tensors and batched dataset as video inputs
    Expectation: Cropped images should in correct shape
    """

    # use randomly generated tensor for testing
    video_frames = np.random.randint(0, 255, size=(32, 64, 64, 3), dtype=np.uint8)
    random_crop_op = vision.RandomCrop(32)
    video_frames = random_crop_op(video_frames)
    assert video_frames.shape[1] == 32
    assert video_frames.shape[2] == 32

    # use a batch of real image for testing
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    random_crop_op = vision.RandomCrop([32, 32])
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1_batch = data1.batch(batch_size=2)

    for item in data1_batch.create_dict_iterator(num_epochs=1, output_numpy=True):
        original_channel = item["image"].shape[-1]

    data1_batch = data1_batch.map(
        operations=random_crop_op, input_columns=["image"])

    for item in data1_batch.create_dict_iterator(num_epochs=1, output_numpy=True):
        shape = item["image"].shape
        assert shape[-3] == 32
        assert shape[-2] == 32
        assert shape[-1] == original_channel


if __name__ == "__main__":
    test_random_crop_01_c()
    test_random_crop_02_c()
    test_random_crop_03_c()
    test_random_crop_04_c()
    test_random_crop_05_c()
    test_random_crop_06_c()
    test_random_crop_07_c()
    test_random_crop_08_c()
    test_random_crop_01_py()
    test_random_crop_02_py()
    test_random_crop_03_py()
    test_random_crop_04_py()
    test_random_crop_05_py()
    test_random_crop_06_py()
    test_random_crop_07_py()
    test_random_crop_08_py()
    test_random_crop_09()
    test_random_crop_10()
    test_random_crop_op_c(True)
    test_random_crop_op_py(True)
    test_random_crop_comp(True)
    test_random_crop_09_c()
    test_random_crop_high_dimensions()
