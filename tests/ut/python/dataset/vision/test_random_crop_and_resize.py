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
Testing RandomResizedCrop op in DE
"""
import cv2
import numpy as np
import os
import pytest
from PIL import Image

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms as ops
import mindspore.dataset.vision.transforms as vision
import mindspore.dataset.vision.utils as mode
from mindspore.dataset.vision.utils import Inter
from mindspore import log as logger
from util import diff_mse, save_and_check_md5, save_and_check_md5_pil, visualize_list, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

GENERATE_GOLDEN = False


def generator_mc(maxid=3):
    """ input five image """
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    size = (300, 300)
    resize_op = vision.Resize(size, Inter.LINEAR)
    with Image.open(image_jpg) as image:
        image = np.array(resize_op(image))
    with Image.open(image_bmp) as image1:
        image1 = np.array(resize_op(image1))
    with Image.open(image_gif) as image2:
        image2 = np.array(resize_op(image2))
    for _ in range(maxid):
        yield image, image, image1, image1, image2


def test_random_crop_and_resize_callable_numpy():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize is callable with NumPy input
    Expectation: Passes the shape equality test
    """
    logger.info("test_random_crop_and_resize_callable_numpy")
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    decode_op = vision.Decode()
    img = decode_op(img)
    assert img.shape == (2268, 4032, 3)

    # test one tensor with interpolation=Inter.AREA
    random_crop_and_resize_op1 = vision.RandomResizedCrop(size=(256, 512), scale=(2, 2), ratio=(1, 3),
                                                          interpolation=Inter.AREA)
    img1 = random_crop_and_resize_op1(img)
    assert img1.shape == (256, 512, 3)

    # test one tensor with interpolation=Inter.PILCUBIC
    random_crop_and_resize_op1 = vision.RandomResizedCrop(size=(128, 512), scale=(3, 3), ratio=(1, 4),
                                                          interpolation=Inter.PILCUBIC)
    img1 = random_crop_and_resize_op1(img)
    assert img1.shape == (128, 512, 3)


def test_random_crop_and_resize_callable_pil():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize is callable with PIL input
    Expectation: Passes the shape equality test
    """
    logger.info("test_random_crop_and_resize_callable_pil")

    img = Image.open("../data/dataset/apple.jpg").convert("RGB")

    assert img.size == (4032, 2268)

    # test one tensor
    random_crop_and_resize_op1 = vision.RandomResizedCrop(size=(256, 512), scale=(2, 2), ratio=(1, 3),
                                                          interpolation=Inter.ANTIALIAS)
    img1 = random_crop_and_resize_op1(img)
    assert img1.size == (512, 256)


def test_random_crop_and_resize_op_c(plot=False):
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize with Cpp implementation
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_and_resize_op_c")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    # With these inputs we expect the code to crop the whole image
    random_crop_and_resize_op = vision.RandomResizedCrop((256, 512), (2, 2), (1, 3))
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_crop_and_resize_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])
    num_iter = 0
    crop_and_resize_images = []
    original_images = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        crop_and_resize = item1["image"]
        original = item2["image"]
        # Note: resize the original image with the same size as the one applied RandomResizedCrop()
        original = cv2.resize(original, (512, 256))
        mse = diff_mse(crop_and_resize, original)
        assert mse == 0
        logger.info("random_crop_and_resize_op_{}, mse: {}".format(num_iter + 1, mse))
        num_iter += 1
        crop_and_resize_images.append(crop_and_resize)
        original_images.append(original)
    if plot:
        visualize_list(original_images, crop_and_resize_images)


def test_random_crop_and_resize_op_py(plot=False):
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize with Python transformations
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_and_resize_op_py")
    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # With these inputs we expect the code to crop the whole image
    transforms1 = [
        vision.Decode(True),
        vision.RandomResizedCrop((256, 512), (2, 2), (1, 3)),
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
    num_iter = 0
    crop_and_resize_images = []
    original_images = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        crop_and_resize = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        original = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        original = cv2.resize(original, (512, 256))
        mse = diff_mse(crop_and_resize, original)
        # Due to rounding error the mse for Python is not exactly 0
        assert mse <= 0.05
        logger.info("random_crop_and_resize_op_{}, mse: {}".format(num_iter + 1, mse))
        num_iter += 1
        crop_and_resize_images.append(crop_and_resize)
        original_images.append(original)
    if plot:
        visualize_list(original_images, crop_and_resize_images)


def test_random_crop_and_resize_op_py_antialias():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize with Python transformations where image interpolation mode is Inter.ANTIALIAS
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_and_resize_op_py_antialias")
    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # With these inputs we expect the code to crop the whole image
    transforms1 = [
        vision.Decode(True),
        vision.RandomResizedCrop((256, 512), (2, 2), (1, 3), Inter.ANTIALIAS),
        vision.ToTensor()
    ]
    transform1 = ops.Compose(transforms1)
    data1 = data1.map(operations=transform1, input_columns=["image"])
    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    logger.info("use RandomResizedCrop by Inter.ANTIALIAS process {} images.".format(num_iter))
    assert num_iter == 3


def test_random_crop_and_resize_01():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize with md5 check
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_and_resize_01")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    random_crop_and_resize_op = vision.RandomResizedCrop((256, 512), (0.5, 0.5), (1, 1))
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_crop_and_resize_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.RandomResizedCrop((256, 512), (0.5, 0.5), (1, 1)),
        vision.ToTensor()
    ]
    transform = ops.Compose(transforms)
    data2 = data2.map(operations=transform, input_columns=["image"])

    filename1 = "random_crop_and_resize_01_c_result.npz"
    filename2 = "random_crop_and_resize_01_py_result.npz"
    save_and_check_md5(data1, filename1, generate_golden=GENERATE_GOLDEN)
    save_and_check_md5_pil(data2, filename2, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_crop_and_resize_02():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize with md5 check where image interpolation mode is Inter.NEAREST
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_and_resize_02")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    random_crop_and_resize_op = vision.RandomResizedCrop((256, 512), interpolation=mode.Inter.NEAREST)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_crop_and_resize_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.RandomResizedCrop((256, 512), interpolation=mode.Inter.NEAREST),
        vision.ToTensor()
    ]
    transform = ops.Compose(transforms)
    data2 = data2.map(operations=transform, input_columns=["image"])

    filename1 = "random_crop_and_resize_02_c_result.npz"
    filename2 = "random_crop_and_resize_02_py_result.npz"
    save_and_check_md5(data1, filename1, generate_golden=GENERATE_GOLDEN)
    save_and_check_md5_pil(data2, filename2, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_crop_and_resize_03():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize with md5 check where max_attempts is 1
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_and_resize_03")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    random_crop_and_resize_op = vision.RandomResizedCrop((256, 512), max_attempts=1)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_crop_and_resize_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.RandomResizedCrop((256, 512), max_attempts=1),
        vision.ToTensor()
    ]
    transform = ops.Compose(transforms)
    data2 = data2.map(operations=transform, input_columns=["image"])

    filename1 = "random_crop_and_resize_03_c_result.npz"
    filename2 = "random_crop_and_resize_03_py_result.npz"
    save_and_check_md5(data1, filename1, generate_golden=GENERATE_GOLDEN)
    save_and_check_md5_pil(data2, filename2, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_crop_and_resize_04_c():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize with pp with invalid range of scales (max<min)
    Expectation: Error is raised as expected
    """
    logger.info("test_random_crop_and_resize_04_c")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    try:
        # If input range of scale is not in the order of (min, max), ValueError will be raised.
        random_crop_and_resize_op = vision.RandomResizedCrop((256, 512), (1, 0.5), (0.5, 0.5))
        data = data.map(operations=decode_op, input_columns=["image"])
        data = data.map(operations=random_crop_and_resize_op, input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "scale should be in (min,max) format. Got (max,min)." in str(e)


def test_random_crop_and_resize_04_py():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize with Python transformations with invalid range of scales (max<min)
    Expectation: Error is raised as expected
    """
    logger.info("test_random_crop_and_resize_04_py")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    try:
        transforms = [
            vision.Decode(True),
            # If input range of scale is not in the order of (min, max), ValueError will be raised.
            vision.RandomResizedCrop((256, 512), (1, 0.5), (0.5, 0.5)),
            vision.ToTensor()
        ]
        transform = ops.Compose(transforms)
        data = data.map(operations=transform, input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "scale should be in (min,max) format. Got (max,min)." in str(e)


def test_random_crop_and_resize_05_c():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize with pp with invalid range of ratio (max<min)
    Expectation: Error is raised as expected
    """
    logger.info("test_random_crop_and_resize_05_c")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    try:
        random_crop_and_resize_op = vision.RandomResizedCrop((256, 512), (1, 1), (1, 0.5))
        # If input range of ratio is not in the order of (min, max), ValueError will be raised.
        data = data.map(operations=decode_op, input_columns=["image"])
        data = data.map(operations=random_crop_and_resize_op, input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "ratio should be in (min,max) format. Got (max,min)." in str(e)


def test_random_crop_and_resize_05_py():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize with Python transformations with invalid range of ratio (max<min)
    Expectation: Error is raised as expected
    """
    logger.info("test_random_crop_and_resize_05_py")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    try:
        transforms = [
            vision.Decode(True),
            # If input range of ratio is not in the order of (min, max), ValueError will be raised.
            vision.RandomResizedCrop((256, 512), (1, 1), (1, 0.5)),
            vision.ToTensor()
        ]
        transform = ops.Compose(transforms)
        data = data.map(operations=transform, input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "ratio should be in (min,max) format. Got (max,min)." in str(e)


def test_random_crop_and_resize_comp(plot=False):
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize and compare between Python and Cpp image augmentation
    Expectation: Resulting datasets from both operations are expected to be the same
    """
    logger.info("test_random_crop_and_resize_comp")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    random_crop_and_resize_op = vision.RandomResizedCrop(512, (1, 1), (0.5, 0.5))
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_crop_and_resize_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.RandomResizedCrop(512, (1, 1), (0.5, 0.5)),
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
        mse = diff_mse(c_image, py_image)
        assert mse < 0.02  # rounding error
    if plot:
        visualize_list(image_c_cropped, image_py_cropped, visualize_mode=2)


def test_random_crop_and_resize_06():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize with pp with invalid values for scale
    Expectation: Error is raised as expected
    """
    logger.info("test_random_crop_and_resize_05_c")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    try:
        random_crop_and_resize_op = vision.RandomResizedCrop((256, 512), scale="", ratio=(1, 0.5))
        data = data.map(operations=decode_op, input_columns=["image"])
        data.map(operations=random_crop_and_resize_op, input_columns=["image"])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Argument scale with value \"\" is not of type [<class 'tuple'>, <class 'list'>]" in str(e)

    try:
        random_crop_and_resize_op = vision.RandomResizedCrop((256, 512), scale=(1, "2"), ratio=(1, 0.5))
        data = data.map(operations=decode_op, input_columns=["image"])
        data.map(operations=random_crop_and_resize_op, input_columns=["image"])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Argument scale[1] with value 2 is not of type [<class 'float'>, <class 'int'>]" in str(e)


def test_random_crop_and_resize_07():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize with different fields
    Expectation: The dataset is processed as expected
    """
    logger.info("Test RandomCropAndResize with different fields.")

    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=ops.Duplicate(), input_columns=["image"],
                    output_columns=["image", "image_copy"])
    random_crop_and_resize_op = vision.RandomResizedCrop((256, 512), (2, 2), (1, 3))
    decode_op = vision.Decode()

    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=decode_op, input_columns=["image_copy"])
    data = data.map(operations=random_crop_and_resize_op, input_columns=["image", "image_copy"])

    num_iter = 0
    for data1 in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = data1["image"]
        image_copy = data1["image_copy"]
        mse = diff_mse(image, image_copy)
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1


def test_random_crop_and_resize_08():
    """
    Feature: RandomCropAndResize
    Description: Test RandomCropAndResize with 4 dim image
    Expectation: The data is processed successfully
    """
    logger.info("test_random_crop_and_resize_08")

    original_seed = config_get_set_seed(5)
    original_worker = config_get_set_num_parallel_workers(1)

    data = np.random.randint(0, 255, (3, 3, 4, 3), np.uint8)
    res1 = [[[83, 24, 209], [114, 181, 190]], [[200, 201, 36], [154, 13, 117]]]
    res2 = [[[158, 140, 182], [104, 154, 109]], [[230, 79, 193], [87, 170, 223]]]
    res3 = [[[179, 202, 143], [150, 178, 67]], [[20, 94, 159], [253, 151, 82]]]
    expected_result = np.array([res1, res2, res3], dtype=np.uint8)

    random_crop_and_resize_op = vision.RandomResizedCrop((2, 2))
    output = random_crop_and_resize_op(data)

    mse = diff_mse(output, expected_result)
    assert mse < 0.0001
    assert output.shape[-2] == 2
    assert output.shape[-3] == 2

    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_worker)


def test_random_crop_and_resize_pipeline():
    """
    Feature: RandomCropAndResize
    Description: Test RandomCropAndResize with 4 dim image
    Expectation: The data is processed successfully
    """
    logger.info("Test RandomCropAndResize pipeline with 4 dimension input")

    original_seed = config_get_set_seed(5)
    original_worker = config_get_set_num_parallel_workers(1)

    data = np.random.randint(0, 255, (1, 3, 3, 4, 3), np.uint8)
    res1 = [[[83, 24, 209], [114, 181, 190]], [[200, 201, 36], [154, 13, 117]]]
    res2 = [[[158, 140, 182], [104, 154, 109]], [[230, 79, 193], [87, 170, 223]]]
    res3 = [[[179, 202, 143], [150, 178, 67]], [[20, 94, 159], [253, 151, 82]]]
    expected_result = np.array([[res1, res2, res3]], dtype=np.uint8)

    random_crop_and_resize = vision.RandomResizedCrop((2, 2))
    dataset = ds.NumpySlicesDataset(data, column_names=["image"], shuffle=False)
    dataset = dataset.map(input_columns=["image"], operations=random_crop_and_resize)

    for i, item in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        mse = diff_mse(item["image"], expected_result[i])
        assert mse < 0.0001

    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_worker)


def test_random_crop_and_resize_eager_error_01():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize in eager mode with PIL input and C++ only interpolation AREA and PILCUBIC
    Expectation: Correct error is thrown as expected
    """
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    with pytest.raises(TypeError) as error_info:
        random_crop_and_resize_op = vision.RandomResizedCrop(size=(100, 200), scale=[1.0, 2.0],
                                                             interpolation=Inter.AREA)
        _ = random_crop_and_resize_op(img)
    assert "Current Interpolation is not supported with PIL input." in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        random_crop_and_resize_op = vision.RandomResizedCrop(size=(100, 200), scale=[1.0, 2.0],
                                                             interpolation=Inter.PILCUBIC)
        _ = random_crop_and_resize_op(img)
    assert "Current Interpolation is not supported with PIL input." in str(error_info.value)


def test_random_crop_and_resize_eager_error_02():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize in eager mode with NumPy input and Python only interpolation ANTIALIAS
    Expectation: Correct error is thrown as expected
    """
    img = np.random.randint(0, 1, (100, 100, 3)).astype(np.uint8)
    with pytest.raises(TypeError) as error_info:
        random_crop_and_resize_op = vision.RandomResizedCrop(size=(100, 200), scale=[1.0, 2.0],
                                                             interpolation=Inter.ANTIALIAS)
        _ = random_crop_and_resize_op(img)
    assert "img should be PIL image. Got <class 'numpy.ndarray'>." in str(error_info.value)


def test_random_resized_crop_operation_01():
    """
    Feature: RandomResizedCrop operation
    Description: Testing the normal functionality of the RandomResizedCrop operator
    Expectation: The Output is equal to the expected output
    """
    # Test RandomResizedCrop function parameter size is 1
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = 1
    random_resize_crop_op = vision.RandomResizedCrop(size=size)
    dataset = dataset.map(input_columns=["image"],
                            operations=random_resize_crop_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCrop function parameter size is [500, 520]
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = [500, 520]
    random_resize_crop_op = vision.RandomResizedCrop(size=size)
    dataset = dataset.map(input_columns=["image"],
                            operations=random_resize_crop_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCrop function parameter max_attempts is 16777216
    source = generator_mc
    column_names = ["image1", "image2", "image3", "image4", "image5"]
    dataset = ds.GeneratorDataset(source, column_names)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.BILINEAR
    max_attempts = 16777216
    random_resize_crop_op = vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=interpolation,
                                                    max_attempts=max_attempts)
    dataset = dataset.map(
        input_columns=["image1", "image2", "image3", "image4", "image5"],
        operations=random_resize_crop_op)
    image_data = []
    for data in dataset.create_dict_iterator(output_numpy=True):
        image_data.append(data["image1"])
        assert (data["image1"] == data["image2"]).all()
        assert (data["image3"] == data["image4"]).all()
    assert ((image_data[0] != image_data[1]).any() or (image_data[0] != image_data[2]).any())

    # Test RandomResizedCrop function all parameter
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (500, 520)
    scale = [0.5, 1.0]
    ratio = [0.5, 1.0]
    interpolation = Inter.BILINEAR
    max_attempts = 1
    random_resize_crop_op = vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=interpolation,
                                                    max_attempts=max_attempts)
    dataset = dataset.map(input_columns=["image"],
                            operations=random_resize_crop_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCrop function no scale parameter
    source = generator_mc
    column_names = ["image1", "image2", "image3", "image4", "image5"]
    dataset = ds.GeneratorDataset(source, column_names)
    size = (500, 520)
    ratio = (0.5, 1.0)
    interpolation = Inter.BILINEAR
    max_attempts = 10
    random_resize_crop_op = vision.RandomResizedCrop(size=size, ratio=ratio, interpolation=interpolation,
                                                    max_attempts=max_attempts)
    dataset = dataset.map(input_columns=["image1", "image2", "image5"], operations=random_resize_crop_op)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["image1"] == data["image2"]).all()


def test_random_resized_crop_operation_02():
    """
    Feature: RandomResizedCrop operation
    Description: Testing the normal functionality of the RandomResizedCrop operator
    Expectation: The Output is equal to the expected output
    """
    # Test RandomResizedCrop function no ratio parameter
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (500, 520)
    scale = (0.5, 1.0)
    interpolation = Inter.BILINEAR
    max_attempts = 10
    random_resize_crop_op = vision.RandomResizedCrop(size=size, scale=scale, interpolation=interpolation,
                                                    max_attempts=max_attempts)
    dataset = dataset.map(input_columns=["image"], operations=random_resize_crop_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCrop function no interpolation parameter
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = [0.5, 1.0]
    max_attempts = 10
    random_resize_crop_op = vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio, max_attempts=max_attempts)
    dataset = dataset.map(input_columns=["image"], operations=random_resize_crop_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCrop function no max_attempts parameter
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.BILINEAR
    random_resize_crop_op = vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=random_resize_crop_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCrop function image is jpg
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image = Image.open(image_jpg)
    size = 600
    scale = (0.5, 1.5)
    ratio = (0.5, 2)
    interpolation = Inter.BILINEAR
    max_attempts = 1
    random_resize_crop_op = vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=interpolation,
                                                    max_attempts=max_attempts)
    _ = random_resize_crop_op(image)
    image.close()

    # Test RandomResizedCrop function image is bmp
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    image = Image.open(image_bmp)
    size = [400, 500]
    scale = [0.1, 0.1]
    ratio = (0.2, 1)
    interpolation = Inter.NEAREST
    max_attempts = 3
    random_resize_crop_op = vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=interpolation,
                                                    max_attempts=max_attempts)
    _ = random_resize_crop_op(image)
    image.close()

    # Test RandomResizedCrop function image is png
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    image = Image.open(image_png)
    size = (40, 50)
    scale = [1.0, 1.0]
    ratio = [0.2, 1]
    interpolation = Inter.BICUBIC
    max_attempts = 8
    random_resize_crop_op = vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=interpolation,
                                                    max_attempts=max_attempts)
    _ = random_resize_crop_op(image)
    image.close()


def test_random_resized_crop_operation_03():
    """
    Feature: RandomResizedCrop operation
    Description: Testing the normal functionality of the RandomResizedCrop operator
    Expectation: The Output is equal to the expected output
    """
    # Test RandomResizedCrop function image is gif
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    image = Image.open(image_gif)
    size = (1400, 2500)
    scale = (0.01, 0.86)
    ratio = (1.0, 3.2)
    interpolation = Inter.BILINEAR
    max_attempts = 100
    random_resize_crop_op = vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=interpolation,
                                                    max_attempts=max_attempts)
    out = random_resize_crop_op(image, image)
    assert (out[0] == out[1]).all()
    image.close()

    # Test RandomResizedCrop function input multiple shapes
    image1 = np.random.randn(250, 500, 3)
    image2 = np.random.randn(250, 500, 3)
    image3 = np.random.randn(250, 500)
    image4 = np.random.randn(250, 500, 4)
    size = (400, 300)
    scale = (0.01, 0.02)
    ratio = (0.9, 3.2)
    interpolation = Inter.BILINEAR
    max_attempts = 1
    random_resize_crop_op = vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=interpolation,
                                                    max_attempts=max_attempts)
    _ = random_resize_crop_op(image1, image2, image3, image4)

    # Test RandomResizedCrop function input data shape is (250, 500, 1)
    image = np.random.randn(250, 500, 1)

    random_resize_crop_op = vision.RandomResizedCrop(size=100, scale=(0.6, 0.6), ratio=(0.5, 0.5),
                                                    interpolation=Inter.BILINEAR, max_attempts=2)
    random_resize_crop_op(image)

    # Test RandomResizedCrop function input data shape is (1000, 1500)
    image = np.random.randn(1000, 1500)

    random_resize_crop_op = vision.RandomResizedCrop(size=2000, scale=(0.8, 1.0), ratio=(0.7, 2.5),
                                                    interpolation=Inter.BILINEAR, max_attempts=2)
    _ = random_resize_crop_op(image)

    # Test RandomResizedCrop function input data shape is (250, 500, 4)
    image = np.random.randn(250, 500, 4)
    random_resize_crop_op = vision.RandomResizedCrop(size=[400, 300])
    _ = random_resize_crop_op(image)

    # Test RandomResizedCrop function input data type is list
    image = np.random.randint(0, 1, (2, 3, 4), dtype=np.uint8)
    size = [400, 300]
    random_resize_crop_op = vision.RandomResizedCrop(size=size)
    random_resize_crop_op(image)

    # Test RandomResizedCrop function input is list
    image = np.random.randint(0, 255, (52, 52, 3)).astype(np.uint8).tolist()
    random_resize_crop_op = vision.RandomResizedCrop(30)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        random_resize_crop_op(image)

    # Test RandomResizedCrop function size is [500, 50]
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = [200, 50]
    scale = (0.2, 2.0)
    ratio = [0.5, 1.0]
    interpolation = Inter.AREA
    random_resize_op = vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=random_resize_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCrop function pepiline interpolation_c is Inter.PILCUBIC
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (250, 300)
    ratio = [0.5, 2.0]
    interpolation = Inter.PILCUBIC
    random_resize_op = vision.RandomResizedCrop(size=size, ratio=ratio, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=random_resize_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCrop function eager interpolation_c is Inter.PILCUBIC
    image = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
    size = (100, 100)
    interpolation = Inter.PILCUBIC
    scale = [1.0, 1.0]
    random_resize_op = vision.RandomResizedCrop(size, scale=scale, interpolation=interpolation)
    _ = random_resize_op(image)


def test_random_resized_crop_operation_04():
    """
    Feature: RandomResizedCrop operation
    Description: Testing the normal functionality of the RandomResizedCrop operator
    Expectation: The Output is equal to the expected output
    """
    # Test RandomResizedCrop function eager interpolation_c is Inter.NEAREST
    image = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
    size = (100, 100)
    interpolation = Inter.NEAREST
    scale = [1.0, 1.0]
    random_resize_op = vision.RandomResizedCrop(size, scale=scale, interpolation=interpolation)
    _ = random_resize_op(image)

    # Test RandomResizedCrop function eager interpolation_c is Inter.BICUBIC
    image = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
    random_resize_op = vision.RandomResizedCrop(size=(100, 100), scale=[1.0, 1.0], interpolation=Inter.BICUBIC)
    _ = random_resize_op(image)

    # Test RandomResizedCrop function eager interpolation_c is Inter.AREA
    image = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
    size = (100, 100)
    interpolation = Inter.AREA
    scale = [1.0, 1.0]
    random_resize_op = vision.RandomResizedCrop(size, scale=scale, interpolation=interpolation)
    _ = random_resize_op(image)

    # Test RandomResizedCrop function eager interpolation_c is Inter.ANTIALIAS
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        random_resize_op = vision.RandomResizedCrop(size=(100, 100), scale=[1.0, 1.0], interpolation=Inter.ANTIALIAS)
        _ = random_resize_op(image)

    # Test RandomResizedCrop function pipeline input datatype is PIL
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False)
    size = 1
    random_resize_crop_op = [vision.Decode(to_pil=True), vision.RandomResizedCrop(size=size)]
    dataset = dataset.map(input_columns=["image"],
                            operations=random_resize_crop_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCrop function pipeline input datatype is Numpy
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False)
    size = 1
    random_resize_crop_op = [vision.Decode(to_pil=False), vision.RandomResizedCrop(size=size)]
    dataset = dataset.map(input_columns=["image"],
                            operations=random_resize_crop_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_random_resized_crop_exception_01():
    """
    Feature: RandomResizedCrop operation
    Description: Testing the RandomResizedCrop Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test RandomResizedCrop function parameter size is 0
    size = 0
    with pytest.raises(ValueError,
                       match="Input is not within the required interval"):
        vision.RandomResizedCrop(size=size)

    # Test RandomResizedCrop function parameter size is 16777217
    size = 16777217
    with pytest.raises(ValueError,
                       match="Input is not within the required interval"):
        vision.RandomResizedCrop(size=size)

    # Test RandomResizedCrop function parameter size is 500.5
    size = 500.5
    with pytest.raises(TypeError, match="Argument size with value 500.5 is not of type \\[<class 'int'>, "
                                        "<class 'list'>, <class 'tuple'>\\], but got <class 'float'>"):
        vision.RandomResizedCrop(size=size)

    # Test RandomResizedCrop function parameter size is (500, 500, 520)
    size = (500, 500, 520)
    with pytest.raises(TypeError,
                       match="Size should be a single integer or a list/tuple"):
        vision.RandomResizedCrop(size=size)

    # Test RandomResizedCrop function parameter size is ""
    size = ""
    with pytest.raises(TypeError, match='''Argument size with value "" is not of type \\[<class 'int'>, '''
                                        '''<class 'list'>, <class 'tuple'>\\], but got <class 'str'>.'''):
        vision.RandomResizedCrop(size=size)

    # Test RandomResizedCrop function parameter scale is (-0.5, 1.5)
    size = (500, 520)
    scale = (-0.5, 1.5)
    with pytest.raises(ValueError,
                       match="Input is not within the required interval"):
        vision.RandomResizedCrop(size=size, scale=scale)

    # Test RandomResizedCrop function parameter scale is (1.5, 0.5)
    size = (500, 520)
    scale = (1.5, 0.5)
    with pytest.raises(ValueError,
                       match="scale should be in \\(min,max\\) format. Got \\(max,min\\)."):
        vision.RandomResizedCrop(size=size, scale=scale)

    # Test RandomResizedCrop function parameter scale is ("", "")
    size = (500, 520)
    scale = ("", "")
    with pytest.raises(TypeError, match='''Argument scale\\[0\\] with value "" is not of type \\[<class \'float\'>, '''
                                        '''<class \'int\'>\\], but got <class \'str\'>.'''):
        vision.RandomResizedCrop(size=size, scale=scale)

    # Test RandomResizedCrop function parameter scale is ms.Tensor([0.5, 1.0])
    size = (500, 520)
    scale = ms.Tensor([0.5, 1.0])
    with pytest.raises(TypeError) as e:
        vision.RandomResizedCrop(size=size, scale=scale)
    assert "Argument scale with value {} is not of type [<class 'tuple'>, <class 'list'>]".format(scale) in str(e)

    # Test RandomResizedCrop function parameter scale is ""
    size = (500, 520)
    scale = ""
    with pytest.raises(TypeError, match='''Argument scale with value "" is not of type \\[<class \'tuple\'>, '''
                                        '''<class \'list\'>\\], but got <class \'str\'>.'''):
        vision.RandomResizedCrop(size=size, scale=scale)

    # Test RandomResizedCrop function parameter ratio is (-0.5, 1.0)
    size = (500, 520)
    scale = (0.5, 1)
    ratio = (-0.5, 1.0)
    with pytest.raises(ValueError,
                       match="Input ratio\\[0\\] is not within the required interval of \\(0, 16777216\\]."):
        vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio)

    # Test RandomResizedCrop function parameter ratio is (1.5, 0.5)
    size = (500, 520)
    scale = (0.5, 1)
    ratio = (1.5, 0.5)
    with pytest.raises(ValueError,
                       match="ratio should be in \\(min,max\\) format. Got \\(max,min\\)."):
        vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio)

    # Test RandomResizedCrop function parameter ratio is ("", "")
    size = (500, 520)
    scale = (0.5, 1)
    ratio = ("", "")
    with pytest.raises(TypeError, match='''Argument ratio\\[0\\] with value "" is not of type \\[<class 'float'>, '''
                                        '''<class 'int'>\\], but got <class 'str'>.'''):
        vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio)

    # Test RandomResizedCrop function parameter ratio is ms.Tensor([0.5, 1.0])
    size = (500, 520)
    scale = (0.5, 1)
    ratio = ms.Tensor([0.5, 1.0])
    with pytest.raises(TypeError) as e:
        vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio)
    assert "Argument ratio with value {} is not of type [<class 'tuple'>, <class 'list'>]".format(ratio) in str(e)


def test_random_resized_crop_exception_02():
    """
    Feature: RandomResizedCrop operation
    Description: Testing the RandomResizedCrop Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test RandomResizedCrop function parameter ratio is ""
    size = (500, 520)
    scale = (0.5, 1)
    ratio = ""
    msg = r'''Argument ratio with value "" is not of type \[<class 'tuple'>, <class 'list'>\], but got <class 'str'>.'''
    with pytest.raises(TypeError, match=msg):
        vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio)

    # Test RandomResizedCrop function parameter interpolation is ""
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = ""
    msg = r'''Argument interpolation with value "" is not of type \[<enum 'Inter'>\], but got <class 'str'>.'''
    with pytest.raises(TypeError, match=msg):
        vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=interpolation)

    # Test RandomResizedCrop function parameter max_attempts is 0
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.BILINEAR
    max_attempts = 0
    with pytest.raises(ValueError,
                       match="Input max_attempts is not within the required interval of \\[1, 2147483647\\]"):
        vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=interpolation,
                                  max_attempts=max_attempts)

    # Test RandomResizedCrop function parameter max_attempts is 1.5
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.BILINEAR
    max_attempts = 1.5
    with pytest.raises(TypeError, match="Argument max_attempts with value 1.5 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'float'>."):
        random_resize_crop_op = vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio,
                                                        interpolation=interpolation, max_attempts=max_attempts)
        dataset = dataset.map(input_columns=["image"],
                                operations=random_resize_crop_op)
        for _ in zip(dataset1.create_dict_iterator(output_numpy=True),
                     dataset.create_dict_iterator(output_numpy=True)):
            pass

    # Test RandomResizedCrop function parameter max_attempts is ""
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.BILINEAR
    max_attempts = ""
    with pytest.raises(TypeError, match="""Argument max_attempts with value "" is not of type \\[<class 'int'>\\], """
                                        "but got <class 'str'>."):
        vision.RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=interpolation,
                                  max_attempts=max_attempts)

    # Test RandomResizedCrop function no required argument
    with pytest.raises(TypeError, match="missing a required argument"):
        vision.RandomResizedCrop()

    # Test RandomResizedCrop function more para
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.BILINEAR
    max_attempts = 10
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        vision.RandomResizedCrop(size, scale, ratio, interpolation, max_attempts, more_para)

    # Test RandomResizedCrop function input data shape is 1d
    image = np.random.randn(250)
    size = [400, 300]
    random_resize_crop_op = vision.RandomResizedCrop(size=size)
    with pytest.raises(RuntimeError,
                       match="RandomResizedCrop: input tensor should have at least 2 dimensions"):
        random_resize_crop_op(image)

    # Test RandomResizedCrop function input data type is Tensor
    image = ms.Tensor(np.random.randn(25, 50, 3))
    size = [40, 30]
    random_resize_crop_op = vision.RandomResizedCrop(size=size)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image."):
        random_resize_crop_op(image)

    # Test RandomResizedCrop function size is {400, 300}
    size = {400, 300}
    with pytest.raises(TypeError, match=("Argument size with value {400, 300} is not of type \\[<class "
                                         "'int'>, <class 'list'>, <class 'tuple'>\\].")):
        vision.RandomResizedCrop(size=size)


def test_random_resized_crop_exception_03():
    """
    Feature: RandomResizedCrop operation
    Description: Testing the RandomResizedCrop Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test RandomResizedCrop function size is [400.0, 300]
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image = Image.open(image_jpg)
    size = [400.0, 300]
    with pytest.raises(TypeError, match="Argument size\\[0\\] with value 400.0 is not of type \\[<class 'int'>\\]."):
        random_resize_crop_op = vision.RandomResizedCrop(size=size)
        random_resize_crop_op(image)
        image.close()

    # Test RandomResizedCrop function size is np.array([400, 300])
    size = np.array([400, 300])
    with pytest.raises(TypeError, match=("Argument size with value \\[400 300\\] is not of type \\[<class "
                                         "'int'>, <class 'list'>, <class 'tuple'>\\].")):
        vision.RandomResizedCrop(size=size)

    # Test RandomResizedCrop function size is -1
    size = -1
    with pytest.raises(ValueError,
                       match="Input is not within the required interval of \\[1, 16777216\\]."):
        vision.RandomResizedCrop(size=size)

    # Test RandomResizedCrop function size is [250001, 500000]
    image = np.random.randn(250, 500, 3)
    size = [250001, 500000]
    random_resize_crop_op = vision.RandomResizedCrop(size=size)
    with pytest.raises(RuntimeError,
                       match="1\\) is too big, it's up to 1000 times the original image; 2\\) can not be 0."):
        random_resize_crop_op(image)

    # Test RandomResizedCrop function scale is numpy.ndarray
    size = [300, 300]
    scale = np.array([0.3, 0.8])
    with pytest.raises(TypeError, match="Argument scale with value \\[0.3 0.8\\] is not of "
                                        "type \\[<class 'tuple'>, <class 'list'>\\]."):
        vision.RandomResizedCrop(size=size, scale=scale)

    # Test RandomResizedCrop function scale is 0.5
    size = [300, 300]
    scale = 0.5
    with pytest.raises(TypeError,
                       match="Argument scale with value 0.5 is not of type \\[<class 'tuple'>, <class 'list'>\\]."):
        vision.RandomResizedCrop(size=size, scale=scale)

    # Test RandomResizedCrop function scale is (0.5, 0.7, 0.8),
    image = np.random.randn(250, 500, 3)
    size = [300, 300]
    scale = (0.5, 0.7, 0.8)
    with pytest.raises(TypeError, match="scale should be a list/tuple of length 2."):
        random_resize_crop_op = vision.RandomResizedCrop(size=size, scale=scale)
        random_resize_crop_op(image)

    # Test RandomResizedCrop function scale is (0.5,)
    size = [300, 300]
    scale = (0.5,)
    with pytest.raises(TypeError, match="scale should be a list/tuple of length 2."):
        vision.RandomResizedCrop(size=size, scale=scale)

    # Test RandomResizedCrop function scale is (0.5, 16777216.01)
    size = [300, 300]
    scale = (0.5, 16777216.01)
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]."):
        vision.RandomResizedCrop(size=size, scale=scale)

    # Test RandomResizedCrop function scale is (0.0, 0.0)
    size = [300, 300]
    scale = (0.0, 0.0)
    with pytest.raises(ValueError, match="Input scale\\[1\\] must be greater than 0."):
        vision.RandomResizedCrop(size=size, scale=scale)

    # Test RandomResizedCrop function ratio is numpy.ndarray
    size = [300, 300]
    ratio = np.array([0.5, 2])
    with pytest.raises(TypeError, match="Argument ratio with value \\[0.5 2. \\] is not "
                                        "of type \\[<class 'tuple'>, <class 'list'>\\]."):
        vision.RandomResizedCrop(size=size, ratio=ratio)

    # Test RandomResizedCrop function ratio is 0.8
    size = [300, 300]
    ratio = 0.8
    with pytest.raises(TypeError,
                       match="Argument ratio with value 0.8 is not of type \\[<class 'tuple'>, <class 'list'>\\]."):
        vision.RandomResizedCrop(size=size, ratio=ratio)

    # Test RandomResizedCrop function ratio is (0.3, 0.5, 1.2)
    image = np.random.randn(250, 500, 3)
    size = [300, 300]
    ratio = (0.3, 0.5, 1.2)
    with pytest.raises(TypeError, match="ratio should be a list/tuple of length 2."):
        random_resize_crop_op = vision.RandomResizedCrop(size=size, ratio=ratio)
        random_resize_crop_op(image)

    # Test RandomResizedCrop function ratio is (0.3,)
    size = [300, 300]
    ratio = (0.3,)
    with pytest.raises(TypeError, match="ratio should be a list/tuple of length 2."):
        vision.RandomResizedCrop(size=size, ratio=ratio)


def test_random_resized_crop_exception_04():
    """
    Feature: RandomResizedCrop operation
    Description: Testing the RandomResizedCrop Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test RandomResizedCrop function ratio is (0.3, 16777216.01)
    size = [300, 300]
    ratio = (0.3, 16777216.01)
    with pytest.raises(ValueError,
                       match="Input ratio\\[1\\] is not within the required interval of \\(0, 16777216\\]."):
        vision.RandomResizedCrop(size=size, ratio=ratio)

    # Test RandomResizedCrop function ratio is (0.0, 0.5)
    size = [300, 300]
    ratio = (0.0, 0.5)
    with pytest.raises(ValueError, match="Input ratio\\[0\\] is not within the required interval of \\(0, 16777216\\]"):
        vision.RandomResizedCrop(size=size, ratio=ratio)

    # Test RandomResizedCrop function interpolation is 1
    size = [300, 300]
    interpolation = 1
    with pytest.raises(TypeError, match="Argument interpolation with value 1 is not of type \\[<enum 'Inter'>\\]."):
        vision.RandomResizedCrop(size=size, interpolation=interpolation)

    # Test RandomResizedCrop function interpolation is str
    size = [300, 300]
    interpolation = "Inter.BILINEAR"
    with pytest.raises(TypeError,
                       match="Argument interpolation with value Inter.BILINEAR is not of type \\[<enum 'Inter'>\\]."):
        vision.RandomResizedCrop(size=size, interpolation=interpolation)

    # Test RandomResizedCrop function interpolation is Inter
    size = [300, 300]
    interpolation = Inter
    with pytest.raises(TypeError,
                       match="Argument interpolation with value <enum 'Inter'> is not of type \\[<enum 'Inter'>\\]."):
        vision.RandomResizedCrop(size=size, interpolation=interpolation)

    # Test RandomResizedCrop function interpolation is list
    size = [300, 300]
    interpolation = [Inter.BILINEAR]
    with pytest.raises(TypeError, match=("Argument interpolation with value \\[<Inter.BILINEAR: 2>\\] "
                                         "is not of type \\[<enum 'Inter'>\\].")):
        vision.RandomResizedCrop(size=size, interpolation=interpolation)

    # Test RandomResizedCrop function max_attempts is None
    image = np.random.randn(250, 500, 3)
    size = [300, 300]
    max_attempts = None
    with pytest.raises(TypeError, match="incompatible constructor arguments."):
        random_resize_crop_op = vision.RandomResizedCrop(size=size, max_attempts=max_attempts)
        random_resize_crop_op(image)

    # Test RandomResizedCrop function max_attempts is "1"
    size = [300, 300]
    max_attempts = "1"
    with pytest.raises(TypeError, match="Argument max_attempts with value 1 is not of type \\[<class 'int'>\\]"):
        vision.RandomResizedCrop(size=size, max_attempts=max_attempts)

    # Test RandomResizedCrop function max_attempts is list
    size = [300, 300]
    max_attempts = [10]
    with pytest.raises(TypeError, match="Argument max_attempts with value \\[10\\] is not of type \\[<class 'int'>\\]"):
        vision.RandomResizedCrop(size=size, max_attempts=max_attempts)

    # Test RandomResizedCrop function max_attempts is 16777217
    size = [300, 300]
    max_attempts = 2147483648
    with pytest.raises(ValueError, match="max_attempts is not within the required interval of \\[1, 2147483647\\]."):
        vision.RandomResizedCrop(size=size, max_attempts=max_attempts)

    # Test RandomResizedCrop function max_attempts is 5.2
    image = np.random.randn(250, 500, 3)
    size = [300, 300]
    max_attempts = 5.2
    with pytest.raises(TypeError, match="Argument max_attempts with value 5.2 is not of type \\[<class 'int'>\\]"):
        random_resize_crop_op = vision.RandomResizedCrop(size=size, max_attempts=max_attempts)
        random_resize_crop_op(image)

    # Test RandomResizedCrop function input is 1d
    random_resize_op = vision.RandomResizedCrop(30)
    with pytest.raises(RuntimeError,
                       match="RandomResizedCrop: input tensor should have at least 2 dimensions"):
        random_resize_op(np.array(10))

    # Test RandomResizedCrop function input the same dataset
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    image2 = np.random.randint(0, 255, (51, 50, 3)).astype(np.uint8)
    random_resize_op = vision.RandomResizedCrop(30)
    with pytest.raises(RuntimeError,
                       match="Input tensor in different columns of each row must have the same size."):
        random_resize_op(image, image2)

    # Test RandomResizedCrop function eager interpolation_c is Inter.PILCUBIC
    image = np.random.randn(2, 3, 4)
    random_resize_op = vision.RandomResizedCrop(size=(100, 100), interpolation=Inter.PILCUBIC)
    with pytest.raises(RuntimeError, match="CropAndResize: Interpolation mode PILCUBIC only "
                                           "supports image with 3 channels, but got:.*"):
        random_resize_op(image)

    # Test RandomResizedCrop function eager interpolation_c is Inter.PILCUBIC
    image = np.random.randint(0, 255, (100, 100)).astype(np.uint8)
    random_resize_op = vision.RandomResizedCrop(size=(100, 100), interpolation=Inter.PILCUBIC)
    with pytest.raises(RuntimeError, match="CropAndResize: Interpolation mode PILCUBIC only "
                                           "supports image with 3 channels, but got:.*"):
        random_resize_op(image)


if __name__ == "__main__":
    test_random_crop_and_resize_callable_numpy()
    test_random_crop_and_resize_callable_pil()
    test_random_crop_and_resize_op_c(True)
    test_random_crop_and_resize_op_py(True)
    test_random_crop_and_resize_op_py_antialias()
    test_random_crop_and_resize_01()
    test_random_crop_and_resize_02()
    test_random_crop_and_resize_03()
    test_random_crop_and_resize_04_c()
    test_random_crop_and_resize_04_py()
    test_random_crop_and_resize_05_c()
    test_random_crop_and_resize_05_py()
    test_random_crop_and_resize_06()
    test_random_crop_and_resize_comp(True)
    test_random_crop_and_resize_07()
    test_random_crop_and_resize_08()
    test_random_crop_and_resize_pipeline()
    test_random_crop_and_resize_eager_error_01()
    test_random_crop_and_resize_eager_error_02()
    test_random_resized_crop_operation_01()
    test_random_resized_crop_operation_02()
    test_random_resized_crop_operation_03()
    test_random_resized_crop_operation_04()
    test_random_resized_crop_exception_01()
    test_random_resized_crop_exception_02()
    test_random_resized_crop_exception_03()
    test_random_resized_crop_exception_04()
