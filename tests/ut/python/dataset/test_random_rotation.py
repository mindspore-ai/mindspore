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
Testing RandomRotation in DE
"""
import cv2
import numpy as np
from PIL import Image
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.vision as vision
from mindspore.dataset.vision.utils import Inter
from mindspore import log as logger
from util import visualize_image, visualize_list, diff_mse, save_and_check_md5, save_and_check_md5_pil, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False


def test_random_rotation_op_c(plot=False):
    """
    Feature: RandomRotation
    Description: Test RandomRotation in Cpp transformations op
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_rotation_op_c")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = vision.Decode()
    # use [90, 90] to force rotate 90 degrees, expand is set to be True to match output size
    random_rotation_op = vision.RandomRotation((90, 90), expand=True)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_rotation_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        rotation_de = item1["image"]
        original = item2["image"]
        logger.info("shape before rotate: {}".format(original.shape))
        rotation_cv = cv2.rotate(original, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mse = diff_mse(rotation_de, rotation_cv)
        logger.info("random_rotation_op_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1
        if plot:
            visualize_image(original, rotation_de, mse, rotation_cv)


def test_random_rotation_op_c_area():
    """
    Feature: RandomRotation
    Description: Test RandomRotation in Cpp transformations op with Interpolation AREA
    Expectation: Number of returned data rows is correct
    """
    logger.info("test_random_rotation_op_c_area")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = vision.Decode()
    # Use [180, 180] to force rotate 180 degrees, expand is set to be True to match output size
    # Use resample with Interpolation AREA
    random_rotation_op = vision.RandomRotation((180, 180), expand=True, resample=Inter.AREA)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_rotation_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        rotation_de = item1["image"]
        original = item2["image"]
        logger.info("shape before rotate: {}".format(original.shape))
        rotation_cv = cv2.rotate(original, cv2.ROTATE_180)
        mse = diff_mse(rotation_de, rotation_cv)
        logger.info("random_rotation_op_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1
    assert num_iter == 3


def test_random_rotation_op_py(plot=False):
    """
    Feature: RandomRotation
    Description: Test RandomRotation in Python transformations op
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_rotation_op_py")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    # use [90, 90] to force rotate 90 degrees, expand is set to be True to match output size
    transform1 = mindspore.dataset.transforms.Compose([vision.Decode(True),
                                                       vision.RandomRotation((90, 90), expand=True),
                                                       vision.ToTensor()])
    data1 = data1.map(operations=transform1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transform2 = mindspore.dataset.transforms.Compose([vision.Decode(True),
                                                       vision.ToTensor()])
    data2 = data2.map(operations=transform2, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        rotation_de = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        original = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        logger.info("shape before rotate: {}".format(original.shape))
        rotation_cv = cv2.rotate(original, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mse = diff_mse(rotation_de, rotation_cv)
        logger.info("random_rotation_op_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1
        if plot:
            visualize_image(original, rotation_de, mse, rotation_cv)


def test_random_rotation_op_py_antialias():
    """
    Feature: RandomRotation
    Description: Test RandomRotation in Python transformations op with resample=Inter.ANTIALIAS
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_rotation_op_py_antialias")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    # use [90, 90] to force rotate 90 degrees, expand is set to be True to match output size
    transform1 = mindspore.dataset.transforms.Compose([vision.Decode(True),
                                                       vision.RandomRotation((90, 90),
                                                                             expand=True,
                                                                             resample=Inter.ANTIALIAS),
                                                       vision.ToTensor()])
    data1 = data1.map(operations=transform1, input_columns=["image"])

    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    logger.info("use RandomRotation by Inter.ANTIALIAS process {} images.".format(num_iter))
    assert num_iter == 3


def test_random_rotation_expand():
    """
    Feature: RandomRotation
    Description: Test RandomRotation with expand
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_rotation_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    # expand is set to be True to match output size
    random_rotation_op = vision.RandomRotation((0, 90), expand=True)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_rotation_op, input_columns=["image"])

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1):
        rotation = item["image"]
        logger.info("shape after rotate: {}".format(rotation.shape))
        num_iter += 1


def test_random_rotation_md5():
    """
    Feature: RandomRotation
    Description: Test RandomRotation with md5 check
    Expectation: The dataset is processed as expected
    """
    logger.info("Test RandomRotation with md5 check")
    original_seed = config_get_set_seed(5)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    resize_op = vision.RandomRotation((0, 90),
                                      expand=True,
                                      resample=Inter.BILINEAR,
                                      center=(50, 50),
                                      fill_value=150)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=resize_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    transform2 = mindspore.dataset.transforms.Compose([vision.Decode(True),
                                                       vision.RandomRotation((0, 90),
                                                                             expand=True,
                                                                             resample=Inter.BILINEAR,
                                                                             center=(50, 50),
                                                                             fill_value=150),
                                                       vision.ToTensor()])
    data2 = data2.map(operations=transform2, input_columns=["image"])

    # Compare with expected md5 from images
    filename1 = "random_rotation_01_c_result.npz"
    save_and_check_md5(data1, filename1, generate_golden=GENERATE_GOLDEN)
    filename2 = "random_rotation_01_py_result.npz"
    save_and_check_md5_pil(data2, filename2, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_rotation_diff(plot=False):
    """
    Feature: RandomRotation
    Description: Test RandomRotation difference between Python and Cpp transformations op
    Expectation: Both datasets are processed the same as expected
    """
    logger.info("test_random_rotation_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()

    rotation_op = vision.RandomRotation((45, 45))
    ctrans = [decode_op,
              rotation_op
              ]

    data1 = data1.map(operations=ctrans, input_columns=["image"])

    # Second dataset
    transforms = [
        vision.Decode(True),
        vision.RandomRotation((45, 45)),
        vision.ToTensor(),
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform, input_columns=["image"])

    num_iter = 0
    image_list_c, image_list_py = [], []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        c_image = item1["image"]
        py_image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_list_c.append(c_image)
        image_list_py.append(py_image)

        logger.info("shape of c_image: {}".format(c_image.shape))
        logger.info("shape of py_image: {}".format(py_image.shape))

        logger.info("dtype of c_image: {}".format(c_image.dtype))
        logger.info("dtype of py_image: {}".format(py_image.dtype))

        mse = diff_mse(c_image, py_image)
        assert mse < 0.001  # Rounding error
    if plot:
        visualize_list(image_list_c, image_list_py, visualize_mode=2)


def test_random_rotation_op_exception():
    """
    Feature: RandomRotation
    Description: Test RandomRotation in Python transformations op with resample=Inter.ANTIALIAS, but center is not None
    Expectation: ValueError
    """
    logger.info("test_random_rotation_op_exception")

    image = Image.open("../data/dataset/testImageNetData2/train/class1/1_1.jpg")

    with pytest.raises(ValueError) as error_info:
        random_rotation_op = vision.RandomRotation((90, 90), expand=True, resample=Inter.ANTIALIAS, center=(50, 50))
        _ = random_rotation_op(image)
    assert "When using Inter.ANTIALIAS, center needs to be None and angle needs to be an integer multiple of 90." \
           in str(error_info.value)


def test_random_rotation_op_exception_c_pilcubic():
    """
    Feature: RandomRotation
    Description: Test RandomRotation with resample=Inter.PILCUBIC for NumPy input in eager mode
    Expectation: Exception is raised as expected
    """
    logger.info("test_random_rotation_op_exception_c_pilcubic")

    image = cv2.imread("../data/dataset/apple.jpg")

    with pytest.raises(RuntimeError) as error_info:
        random_rotation_op = vision.RandomRotation((90, 90), expand=True, resample=Inter.PILCUBIC)
        _ = random_rotation_op(image)
    assert "RandomRotation: Invalid InterpolationMode" in str(error_info.value)


def test_random_rotation_op_exception_py_pilcubic():
    """
    Feature: RandomRotation
    Description: Test RandomRotation with resample=Inter.PILCUBIC for PIL input in eager mode
    Expectation: Exception is raised as expected
    """
    logger.info("test_random_rotation_op_exception_py_pilcubic")

    image = Image.open("../data/dataset/apple.jpg").convert("RGB")

    with pytest.raises(TypeError) as error_info:
        random_rotation_op = vision.RandomRotation((90, 90), expand=True, resample=Inter.PILCUBIC)
        _ = random_rotation_op(image)
    assert "Current Interpolation is not supported with PIL input." in str(error_info.value)


def test_random_rotation_with_channel_5():
    """
    Feature: RandomRotation
    Description: Test RandomRotation with 5 channel image
    Expectation: The image is processed as expected
    """
    logger.info("test_random_rotation_invalid_dim")

    image = np.random.random((128, 64, 5)).astype(np.float32)
    random_rotation = vision.RandomRotation((90, 90), resample=Inter.NEAREST, expand=True)
    out = random_rotation(image)
    assert out.shape == (64, 128, 5)


def test_random_rotation_with_channel_5_and_invalid_resample():
    """
    Feature: RandomRotation
    Description: Test RandomRotation with 5 channel image and Inter.BICUBIC
    Expectation: RuntimeError is raised
    """
    logger.info("test_random_rotation_with_channel_5_and_invalid_resample")

    image = np.random.random((128, 64, 5)).astype(np.float32)
    with pytest.raises(RuntimeError) as error_info:
        random_rotation = vision.RandomRotation((90, 90), resample=Inter.BICUBIC)
        _ = random_rotation(image)
    assert "interpolation can not be CUBIC when image channel is greater than 4" in str(error_info.value)


if __name__ == "__main__":
    test_random_rotation_op_c(plot=True)
    test_random_rotation_op_c_area()
    test_random_rotation_op_py(plot=True)
    test_random_rotation_op_py_antialias()
    test_random_rotation_expand()
    test_random_rotation_md5()
    test_rotation_diff(plot=True)
    test_random_rotation_op_exception()
    test_random_rotation_op_exception_c_pilcubic()
    test_random_rotation_op_exception_py_pilcubic()
    test_random_rotation_with_channel_5()
    test_random_rotation_with_channel_5_and_invalid_resample()
