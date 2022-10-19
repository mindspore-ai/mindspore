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
Testing RandomCropAndResize op in DE
"""
import numpy as np
import cv2
import pytest
from PIL import Image

import mindspore.dataset.transforms as ops
import mindspore.dataset.vision as vision
import mindspore.dataset.vision.utils as mode
import mindspore.dataset as ds
from mindspore.dataset.vision.utils import Inter
from mindspore import log as logger
from util import diff_mse, save_and_check_md5, save_and_check_md5_pil, visualize_list, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False


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
