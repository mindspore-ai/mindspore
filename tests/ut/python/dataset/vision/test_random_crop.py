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
Testing RandomCrop op in DE
"""
import cv2
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms as ops
import mindspore.dataset.vision.transforms as vision
import mindspore.dataset.vision.utils as mode
from mindspore.dataset.vision import Border
from mindspore import log as logger
from util import save_and_check_md5, save_and_check_md5_pil, visualize_list, config_get_set_seed, \
    config_get_set_num_parallel_workers, diff_mse

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

DATA_DIR_1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_1.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")


def generator_mc(maxid=3, h=500, w=500):
    """ Multi-column generator function as callable input """

    image = np.random.randn(h, w, 3)
    for _ in range(maxid):
        yield image, image, image, image, image


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


def test_random_crop_operation_01():
    """
    Feature: RandomCrop operation
    Description: Testing the normal functionality of the RandomCrop operator
    Expectation: The Output is equal to the expected output
    """
    # When the parameter size is 1, the RandomCrop interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    size = 1
    random_crop_op = vision.RandomCrop(size=size)
    dataset = dataset.map(input_columns=["image"], operations=random_crop_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the parameter size is (400, 600), the RandomCrop interface call succeeds.
    source = generator_mc
    column_names = ["image1", "image2", "image3", "image4", "image5"]
    dataset = ds.GeneratorDataset(source, column_names)
    size = (400, 600)
    padding = (100, 100, 100, 100)
    pad_if_needed = True
    random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed)
    dataset = dataset.map(input_columns=["image1", "image2", "image3", "image4", "image5"], operations=random_crop_op)
    image_data = []
    for data in dataset.create_dict_iterator(output_numpy=True):
        image_data.append(data["image1"])
        assert (data["image1"] == data["image2"]).all()
        assert (data["image1"] == data["image3"]).all()
        assert (data["image1"] == data["image4"]).all()
        assert (data["image1"] == data["image5"]).all()
    assert (image_data[0] != image_data[1]).any()
    assert (image_data[0] != image_data[2]).any()
    assert (image_data[1] != image_data[2]).any()

    # When the parameter size is [500, 520], the RandomCrop interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    size = [500, 520]
    random_crop_op = vision.RandomCrop(size=size)
    dataset = dataset.map(input_columns=["image"], operations=random_crop_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the parameter padding is [1, 1, 1, 1], the RandomCrop interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    size = (500, 520)
    padding = [1, 1, 1, 1]
    fill_value = 0
    random_crop_op = vision.RandomCrop(size=size, padding=padding, fill_value=fill_value)
    dataset = dataset.map(input_columns=["image"], operations=random_crop_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the parameter "pad_if_needed" is set to True, the RandomCrop interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    size = 3000
    padding = 100
    pad_if_needed = True
    fill_value = (255, 255, 255)
    padding_mode = mode.Border.CONSTANT
    random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                      fill_value=fill_value,
                                      padding_mode=padding_mode)
    dataset = dataset.map(input_columns=["image"], operations=random_crop_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the input image format is PNG, the RandomCrop interface call succeeds.
    with Image.open(image_png) as image:
        size = 10000
        padding = 0
        pad_if_needed = True
        fill_value = 0
        padding_mode = Border.CONSTANT
        random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                          fill_value=fill_value, padding_mode=padding_mode)
        _ = random_crop_op(image)


def test_random_crop_operation_02():
    """
    Feature: RandomCrop operation
    Description: Testing the normal functionality of the RandomCrop operator
    Expectation: The Output is equal to the expected output
    """
    # When the input image format is GIF, the RandomCrop interface call succeeds.
    with Image.open(image_gif).convert("RGB") as image:
        size = 1
        padding = 2000
        pad_if_needed = False
        fill_value = 255
        padding_mode = Border.CONSTANT
        random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                          fill_value=fill_value, padding_mode=padding_mode)
        _ = random_crop_op(image)

    # When no padding parameter is specified, the RandomCrop interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    size = 3000
    pad_if_needed = True
    fill_value = (255, 255, 255)
    padding_mode = mode.Border.CONSTANT
    random_crop_op = vision.RandomCrop(size=size, pad_if_needed=pad_if_needed, fill_value=fill_value,
                                      padding_mode=padding_mode)
    dataset = dataset.map(input_columns=["image"], operations=random_crop_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When no padding parameter is specified, the RandomCrop interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    size = 300
    padding = 100
    fill_value = (255, 255, 255)
    padding_mode = mode.Border.CONSTANT
    random_crop_op = vision.RandomCrop(size=size, padding=padding, fill_value=fill_value, padding_mode=padding_mode)
    dataset = dataset.map(input_columns=["image"], operations=random_crop_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When no fill_value parameter is provided, the RandomCrop interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    size = 3000
    padding = (100, 100, 100, 100)
    pad_if_needed = True
    padding_mode = mode.Border.CONSTANT
    random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                      padding_mode=padding_mode)
    dataset = dataset.map(input_columns=["image"], operations=random_crop_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the padding_mode parameter is omitted, the RandomCrop interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    size = 3000
    padding = (100, 100, 100, 100)
    pad_if_needed = True
    fill_value = (255, 255, 255)
    random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed, fill_value=fill_value)
    dataset = dataset.map(input_columns=["image"], operations=random_crop_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the parameter size is 600, the RandomCrop interface call succeeds.
    with Image.open(image_file) as image:
        size = 600
        padding = 100
        pad_if_needed = True
        fill_value = 250
        padding_mode = Border.CONSTANT
        random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                          fill_value=fill_value, padding_mode=padding_mode)
        _ = random_crop_op(image, image)

    # The parameter size is (2500, 2000), and pad_if_needed is True.
    image = np.random.randn(560, 560)
    size = (2500, 2000)
    padding = (300, 400)
    pad_if_needed = True
    fill_value = (100, 50, 128)
    padding_mode = Border.CONSTANT
    random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed, fill_value=fill_value,
                                      padding_mode=padding_mode)
    _ = random_crop_op(image)


def test_random_crop_operation_03():
    """
    Feature: RandomCrop operation
    Description: Testing the normal functionality of the RandomCrop operator
    Expectation: The Output is equal to the expected output
    """
    # The parameter size is (600, 700), which is smaller than the input shape. pad_if_needed is set to False.
    image = np.random.randint(0, 255, (800, 700, 3)).astype(np.uint8)
    image2 = np.random.randint(0, 255, (800, 700)).astype(np.float32)
    size = (600, 700)
    padding = (100, 200, 300, 400)
    pad_if_needed = False
    fill_value = 0
    padding_mode = Border.CONSTANT
    random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed, fill_value=fill_value,
                                      padding_mode=padding_mode)
    out = random_crop_op(image, image, image2)
    assert (out[0] == out[1]).all()
    assert out[2].shape == (600, 700)

    # The parameter size equals the input image dimensions plus padding.
    image = cv2.imread(image_file)
    size = [1884, 1168]
    padding = [50, 200, 400, 800]
    pad_if_needed = False
    fill_value = 0
    padding_mode = Border.EDGE
    random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed, fill_value=fill_value,
                                      padding_mode=padding_mode)
    _ = random_crop_op(image)

    # The parameter size[0] equals image.shape[0] plus padding[0] plus padding[1].
    with Image.open(image_file) as image:
        size = [1184, 1018]
        padding = [100, 200]
        pad_if_needed = True
        fill_value = 0
        padding_mode = Border.CONSTANT
        random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                          fill_value=fill_value, padding_mode=padding_mode)
        _ = random_crop_op(image)

    # The padding parameter has only one value. If the size is smaller than the input image dimensions plus padding.
    image = np.random.randint(0, 255, (658, 714, 1)).astype(np.uint8)
    size = 800
    padding = 300
    pad_if_needed = True
    fill_value = 0
    padding_mode = Border.SYMMETRIC
    random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed, fill_value=fill_value,
                                      padding_mode=padding_mode)
    _ = random_crop_op(image)

    # The parameter size is greater than the input image dimensions plus padding, with pad_if_needed set to true.
    with Image.open(image_file) as image:
        size = 10000
        padding = 0
        pad_if_needed = True
        fill_value = 0
        padding_mode = Border.CONSTANT
        random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                          fill_value=fill_value, padding_mode=padding_mode)
        _ = random_crop_op(image)

    # When the parameter size is 1 and padding is 2000, the RandomCrop interface call succeeds.
    with Image.open(image_file) as image:
        size = 1
        padding = 2000
        pad_if_needed = False
        fill_value = 255
        padding_mode = Border.CONSTANT
        random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                          fill_value=fill_value, padding_mode=padding_mode)
        _ = random_crop_op(image)

    # In eager mode, when the parameter size is 710, the RandomCrop interface call succeeds.
    with Image.open(image_file) as image:
        size = 710
        random_crop_op = vision.RandomCrop(size=size)
        _ = random_crop_op(image)

    # When input data is converted to a PIL image, the RandomCrop interface call succeeds.
    image = np.random.randint(0, 255, (658, 714, 3)).astype(np.uint8)
    image = vision.ToPIL()(image)
    size = (600, 70)
    padding = (100, 200, 300, 400)
    pad_if_needed = False
    fill_value = 0
    padding_mode = Border.CONSTANT
    random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                      fill_value=fill_value, padding_mode=padding_mode)
    _ = random_crop_op(image)


def test_random_crop_operation_04():
    """
    Feature: RandomCrop operation
    Description: Testing the normal functionality of the RandomCrop operator
    Expectation: The Output is equal to the expected output
    """
    # The parameter size[0] equals image.shape[0] plus padding[0] plus padding[1].
    with Image.open(image_bmp) as image:
        size = [333, 128]
        padding = [100, 200]
        pad_if_needed = False
        fill_value = 0
        padding_mode = Border.EDGE
        random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                          fill_value=fill_value, padding_mode=padding_mode)
        _ = random_crop_op(image)

    # When the input image is in PNG format, the RandomCrop interface call succeeds.
    with Image.open(image_png) as image:
        size = [10, 200]
        padding = [100, 200]
        pad_if_needed = False
        fill_value = 0
        padding_mode = Border.REFLECT
        random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                          fill_value=fill_value, padding_mode=padding_mode)
        _ = random_crop_op(image)

    # When the input image format is GIF, the RandomCrop interface call succeeds.
    with Image.open(image_gif) as image:
        size = [500, 400]
        padding = [100, 200]
        pad_if_needed = False
        fill_value = 0
        padding_mode = Border.SYMMETRIC
        random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                          fill_value=fill_value,
                                          padding_mode=padding_mode)
        _ = random_crop_op(image)


def test_random_crop_exception_01():
    """
    Feature: RandomCrop operation
    Description: Testing the RandomCrop Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the size parameter is 0, the RandomCrop interface call fails.
    size = 0
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.RandomCrop(size=size)

    # When the parameter size exceeds 16777216, the RandomCrop interface call fails.
    size = 16777217
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.RandomCrop(size=size)

    # When the parameter size is a 3-tuple, the RandomCrop interface call fails.
    size = (500, 500, 520)
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        vision.RandomCrop(size=size)

    # When the size parameter is empty, the RandomCrop interface call fails.
    size = ""
    with pytest.raises(TypeError, match="Argument size"):
        vision.RandomCrop(size=size)

    # When the padding parameter is negative, the RandomCrop interface call fails.
    size = (500, 520)
    padding = -1
    with pytest.raises(ValueError, match="Input padding is not within the required interval"):
        vision.RandomCrop(size=size, padding=padding)

    # When the padding parameter exceeds the maximum value, the RandomCrop interface call fails.
    size = (500, 520)
    padding = 21474836488
    with pytest.raises(ValueError,
                       match=r"Input padding is not within the required interval of \[0, 2147483647\]."):
        vision.RandomCrop(size=size, padding=padding)

    # When the padding parameter is a 3-tuple, the RandomCrop interface call fails.
    size = (500, 520)
    padding = (1, 1, 1)
    with pytest.raises(ValueError, match="The size of the padding list or tuple should be 2 or 4."):
        vision.RandomCrop(size=size, padding=padding)

    # When the padding parameter is empty, the RandomCrop interface call fails.
    size = (500, 520)
    padding = ""
    with pytest.raises(TypeError, match="Argument padding"):
        vision.RandomCrop(size=size, padding=padding)

    # The parameter size is larger than the image dimensions, and pad_if_needed is set to False.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    size = 3000
    padding = 100
    pad_if_needed = False
    with pytest.raises(RuntimeError,
                       match="RandomCrop: invalid crop size, crop size is bigger than the image dimensions."):
        random_crop_op = vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed)
        dataset = dataset.map(input_columns=["image"], operations=random_crop_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # When the parameter "pad_if_needed" is empty, the RandomCrop interface call fails.
    size = 3000
    padding = 100
    pad_if_needed = ""
    with pytest.raises(TypeError, match="Argument pad_if_needed"):
        vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed)

    # When the fill_value parameter is set to 256, the RandomCrop interface call fails.
    size = 3000
    padding = 100
    pad_if_needed = True
    fill_value = 256
    with pytest.raises(ValueError, match="Input fill_value is not within the required interval"):
        vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed, fill_value=fill_value)

    # When the fill_value parameter is a list, the RandomCrop interface call fails.
    size = 3000
    padding = 100
    pad_if_needed = True
    fill_value = [255, 255, 255]
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed, fill_value=fill_value)

    # When the fill_value parameter is a float, the RandomCrop interface call fails.
    size = 3000
    padding = 100
    pad_if_needed = True
    fill_value = 1.5
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed, fill_value=fill_value)


def test_random_crop_exception_02():
    """
    Feature: RandomCrop operation
    Description: Testing the RandomCrop Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the parameter "padding_mode" is empty, the RandomCrop interface call fails.
    size = 3000
    padding = 100
    pad_if_needed = True
    fill_value = (255, 255, 255)
    padding_mode = ""
    with pytest.raises(TypeError, match="Argument padding_mode"):
        vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed, fill_value=fill_value,
                          padding_mode=padding_mode)

    # When no size parameter is provided, the RandomCrop interface call fails.
    padding = (100, 100, 100, 100)
    pad_if_needed = True
    fill_value = (255, 255, 255)
    padding_mode = mode.Border.CONSTANT
    with pytest.raises(TypeError, match="missing a required argument"):
        vision.RandomCrop(padding=padding, pad_if_needed=pad_if_needed, fill_value=fill_value,
                          padding_mode=padding_mode)

    # When setting redundant parameters, the RandomCrop interface call fails.
    size = 3000
    padding = 100
    pad_if_needed = True
    fill_value = (255, 255, 255)
    padding_mode = mode.Border.CONSTANT
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        vision.RandomCrop(size, padding, pad_if_needed, fill_value, padding_mode, more_para)

    # When input data is 4-dimensional, the RandomCrop interface call fails.
    image = np.random.randint(0, 255, (658, 714, 3, 3)).astype(np.uint8)
    size = (500, 500)
    random_crop_op = vision.RandomCrop(size=size)
    with pytest.raises(RuntimeError,
                       match=r"RandomCrop: invalid crop size, crop size is bigger than the image dimensions, "
                             r"got crop height: 500, crop width: 500"):
        random_crop_op(image)

    # When input data is 1-dimensional, the RandomCrop interface call fails.
    image = np.random.randint(0, 255, (658,)).astype(np.uint8)
    size = (500, 500)
    random_crop_op = vision.RandomCrop(size=size)
    with pytest.raises(RuntimeError,
                       match="RandomCropOp: input tensor should have at least 2 dimensions, but got: 1"):
        random_crop_op(image)

    # When the size parameter is a float, the RandomCrop interface call fails.
    size = 500.5
    with pytest.raises(TypeError, match=("Argument size with value 500.5 is not of type \\[<class "
                                         "'int'>, <class 'list'>, <class 'tuple'>\\].")):
        vision.RandomCrop(size=size)

    # When the size parameter is a NumPy array, the RandomCrop interface call fails.
    size = np.array([500, 500])
    with pytest.raises(TypeError, match=("Argument size with value \\[500 500\\] is not of "
                                         "type \\[<class 'int'>, <class 'list'>, <class 'tuple'>\\].")):
        vision.RandomCrop(size=size)

    # When the parameter size is greater than 2, the RandomCrop interface call fails.
    size = [500, 500, 500]
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple \\(h, w\\) of length 2."):
        vision.RandomCrop(size=size)

    # When the size parameter is not set, the RandomCrop interface call fails.
    with pytest.raises(TypeError, match="missing a required argument: 'size'"):
        vision.RandomCrop()

    # When the parameter size exceeds 16777216, the RandomCrop interface call fails.
    size = (2147483647, 500)
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[1, 16777216\\]."):
        vision.RandomCrop(size=size)

    # When the parameter padding exceeds 2147483647, the RandomCrop interface call fails.
    size = 500
    padding = 2147483648
    with pytest.raises(ValueError, match="Input padding is not within the required interval of \\[0, 2147483647\\]."):
        vision.RandomCrop(size=size, padding=padding)

    # When the parameter "padding" is a list containing only one value, the RandomCrop interface call fails.
    size = 500
    padding = [100]
    with pytest.raises(ValueError, match="The size of the padding list or tuple should be 2 or 4."):
        vision.RandomCrop(size=size, padding=padding)


def test_random_crop_exception_03():
    """
    Feature: RandomCrop operation
    Description: Testing the RandomCrop Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the padding parameter is a list containing three values, the RandomCrop interface call fails.
    size = 500
    padding = [100, 100, 100]
    with pytest.raises(ValueError, match="The size of the padding list or tuple should be 2 or 4."):
        vision.RandomCrop(size=size, padding=padding)

    # When the padding parameter is a float, the RandomCrop interface call fails.
    size = 500
    padding = [100.0, 100]
    with pytest.raises(TypeError,
                       match="Argument padding\\[0\\] with value 100.0 is not of type \\[<class 'int'>\\]."):
        vision.RandomCrop(size=size, padding=padding)

    # When the padding parameter is set to 5 values, the RandomCrop interface call fails.
    size = 500
    padding = [100, 100, 100, 100, 100]
    with pytest.raises(ValueError, match="The size of the padding list or tuple should be 2 or 4."):
        vision.RandomCrop(size=size, padding=padding)

    # When the parameter "padding" is set to "set", the RandomCrop interface call fails.
    size = 500
    padding = {100}
    with pytest.raises(TypeError, match=("Argument padding with value \\{100\\} is not of type \\[<class "
                                         "'tuple'>, <class 'list'>, <class 'numbers.Number'>\\].")):
        vision.RandomCrop(size=size, padding=padding)

    # When the padding parameter is a numpy array, the RandomCrop interface call fails.
    size = 500
    padding = np.array([100, 100])
    with pytest.raises(TypeError, match=("Argument padding with value \\[100 100\\] is not of type \\[<class "
                                         "'tuple'>, <class 'list'>, <class 'numbers.Number'>\\].")):
        vision.RandomCrop(size=size, padding=padding)

    # When the parameter "pad_if_needed" is set to 0, the RandomCrop interface call fails.
    size = 500
    padding = [100, 100]
    pad_if_needed = 0
    with pytest.raises(TypeError, match="Argument pad_if_needed with value 0 is not of type"):
        vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed)

    # When the parameter "pad_if_needed" is a list, the RandomCrop interface call fails.
    size = 500
    padding = [100, 100]
    pad_if_needed = [True]
    with pytest.raises(TypeError, match="Argument pad_if_needed with value \\[True\\] is not of type"):
        vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed)

    # When the parameter "pad_if_needed" is a string, the RandomCrop interface call fails.
    size = 500
    padding = [100, 100]
    pad_if_needed = "True"
    with pytest.raises(TypeError, match="Argument pad_if_needed with value True is not of type"):
        vision.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed)

    # When the parameter "fill_value" is a list, the RandomCrop interface call fails.
    size = 500
    fill_value = [120, 150, 148]
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.RandomCrop(size=size, fill_value=fill_value)

    # When the length of the fill_value parameter exceeds 3, the RandomCrop interface call fails.
    size = 500
    fill_value = (120, 150, 150, 168)
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.RandomCrop(size=size, fill_value=fill_value)

    # When the parameter "fill_value" is a float, the RandomCrop interface call fails.
    size = 500
    fill_value = 100.5
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.RandomCrop(size=size, fill_value=fill_value)

    # When the fill_value parameter is negative, the RandomCrop interface call fails.
    size = 500
    fill_value = -1
    with pytest.raises(ValueError, match="Input fill_value is not within the required interval of \\[0, 255\\]."):
        vision.RandomCrop(size=size, fill_value=fill_value)

    # When the fill_value parameter exceeds 255, the RandomCrop interface call fails.
    size = 500
    fill_value = (256, 88, 132)
    with pytest.raises(ValueError, match=r"Input fill_value\[0\] is not within the required interval of \[0, 255\]."):
        vision.RandomCrop(size=size, fill_value=fill_value)

    # When the fill_value parameter is a 1-tuple, the RandomCrop interface call fails.
    size = 500
    fill_value = (200,)
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.RandomCrop(size=size, fill_value=fill_value)

    # When the parameter "padding_mode" is set to 0, the RandomCrop interface call fails.
    size = 500
    padding_mode = 0
    with pytest.raises(TypeError, match="Argument padding_mode with value 0 is not of type \\[<enum 'Border'>\\]."):
        vision.RandomCrop(size=size, padding_mode=padding_mode)


def test_random_crop_exception_04():
    """
    Feature: RandomCrop operation
    Description: Testing the RandomCrop Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the parameter "padding_mode" is set to "string", the RandomCrop interface call fails.
    size = 500
    padding_mode = "Border.CONSTANT"
    with pytest.raises(TypeError,
                       match="Argument padding_mode with value Border.CONSTANT is not of type \\[<enum 'Border'>\\]"):
        vision.RandomCrop(size=size, padding_mode=padding_mode)

    # When the parameter "padding_mode" is a list, the RandomCrop interface call fails.
    size = 500
    padding_mode = [Border.CONSTANT]
    with pytest.raises(TypeError, match=("Argument padding_mode with value \\[<Border.CONSTANT: 'constant'>\\] "
                                         "is not of type \\[<enum 'Border'>\\].")):
        vision.RandomCrop(size=size, padding_mode=padding_mode)

    # When the input data shape differs, the RandomCrop interface call fails.
    image = np.random.randint(0, 255, (658, 714, 3)).astype(np.uint8)
    image2 = np.random.randint(0, 255, (800,)).astype(np.uint8)
    size = (600, 700)
    random_crop_op = vision.RandomCrop(size=size)
    with pytest.raises(RuntimeError, match="RandomCropOp: input tensor should have at least 2 dimensions, but got: 1"):
        random_crop_op(image, image2)

    # The input image's shape is not <H, W> or <H, W, C>. The RandomCrop interface call failed.
    image = np.random.randint(0, 255, (658, 714, 3)).astype(np.uint8)
    size = (600, 700)
    random_crop_op = vision.RandomCrop(size=size)
    with pytest.raises(RuntimeError, match="RandomCropOp: input tensor should have at least 2 dimensions, but got: 0"):
        random_crop_op(np.array(10), image)

    # When input data is a list, the RandomCrop interface call fails.
    image = np.random.randint(0, 255, (658, 714, 3)).astype(np.uint8).tolist()
    size = 200
    random_crop_op = vision.RandomCrop(size=size)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        random_crop_op(image)

    # When the input channel is not 3, the RandomCrop interface call fails.
    image = np.random.randint(0, 255, (658, 714, 4)).astype(np.uint8)
    size = (600, 700)
    random_crop_op = vision.RandomCrop(size=size)
    with pytest.raises(RuntimeError, match=r"Pad: the channel of image tensor does not match "
                                           r"the requirement of operator. Expecting tensor in channel of \(1, 3\). "
                                           r"But got channel 4."):
        random_crop_op(image)

    # When the padding parameter is set to 1.5, the RandomCrop interface call fails.
    size = (500, 520)
    padding = 1.5
    with pytest.raises(TypeError, match="Argument padding with value 1.5 is not of type "
                                        "\\[<class 'int'>\\], but got <class 'float'>."):
        vision.RandomCrop(size=size, padding=padding)


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
    test_random_crop_operation_01()
    test_random_crop_operation_02()
    test_random_crop_operation_03()
    test_random_crop_operation_04()
    test_random_crop_exception_01()
    test_random_crop_exception_02()
    test_random_crop_exception_03()
    test_random_crop_exception_04()
