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
Testing RandomRotation in DE
"""
import cv2
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.vision.transforms as vision
import mindspore.dataset.vision.utils as mode
from mindspore.dataset.vision.utils import Inter
from mindspore import log as logger
from util import visualize_image, visualize_list, diff_mse, save_and_check_md5, save_and_check_md5_pil, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

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


def test_random_rotation_operation_01():
    """
    Feature: RandomRotation operation
    Description: Testing the normal functionality of the RandomRotation operator
    Expectation: The Output is equal to the expected output
    """
    # Test RandomRotation func degrees is 0
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    degrees = 0
    random_rotation_op = vision.RandomRotation(degrees=degrees)
    dataset = dataset.map(input_columns=["image"], operations=random_rotation_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomRotation func degrees is 1000.5
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    degrees = 1000.5
    random_rotation_op = vision.RandomRotation(degrees=degrees)
    dataset = dataset.map(input_columns=["image"], operations=random_rotation_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomRotation func degrees is [0,100]
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    degrees = [0, 100]
    random_rotation_op = vision.RandomRotation(degrees=degrees)
    dataset = dataset.map(input_columns=["image"], operations=random_rotation_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomRotation func  center is (-100, -200)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    degrees = 100
    resample = mode.Inter.BILINEAR
    expand = True
    center = (-100, -200)
    random_rotation_op = vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center)
    dataset = dataset.map(input_columns=["image"], operations=random_rotation_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomRotation func  fill_value is 0
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    degrees = 100
    resample = mode.Inter.BILINEAR
    expand = True
    center = (100, 200)
    fill_value = 0
    random_rotation_op = vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center,
                                               fill_value=fill_value)
    dataset = dataset.map(input_columns=["image"], operations=random_rotation_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomRotation func all para
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    degrees = 100
    resample = mode.Inter.BILINEAR
    expand = True
    center = (100, 200)
    fill_value = (100, 100, 100)
    random_rotation_op = vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center,
                                               fill_value=fill_value)
    dataset = dataset.map(input_columns=["image"], operations=random_rotation_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_random_rotation_operation_02():
    """
    Feature: RandomRotation operation
    Description: Testing the normal functionality of the RandomRotation operator
    Expectation: The Output is equal to the expected output
    """
    # Test RandomRotation func no 2nd para
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    degrees = 100
    expand = True
    center = (100, 200)
    fill_value = (100, 100, 100)
    random_rotation_op = vision.RandomRotation(degrees=degrees, expand=expand, center=center,
                                               fill_value=fill_value)
    dataset = dataset.map(input_columns=["image"], operations=random_rotation_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomRotation func no 3rd para
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    degrees = 100
    resample = mode.Inter.BILINEAR
    center = (100, 200)
    fill_value = (100, 100, 100)
    random_rotation_op = vision.RandomRotation(degrees=degrees, resample=resample, center=center,
                                               fill_value=fill_value)
    dataset = dataset.map(input_columns=["image"], operations=random_rotation_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomRotation func no 4th para
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    degrees = 100
    resample = mode.Inter.BILINEAR
    expand = True
    fill_value = (100, 100, 100)
    random_rotation_op = vision.RandomRotation(degrees=degrees, resample=resample, expand=expand,
                                               fill_value=fill_value)
    dataset = dataset.map(input_columns=["image"], operations=random_rotation_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomRotation func no  5th para
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    degrees = 100
    resample = mode.Inter.BILINEAR
    expand = True
    center = (100, 200)
    random_rotation_op = vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center)
    dataset = dataset.map(input_columns=["image"], operations=random_rotation_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomRotation func resample is mode.Inter.NEAREST
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    image = Image.open(image_bmp)
    degrees = 36.5
    resample = mode.Inter.NEAREST
    expand = False
    center = (-0.3, -15)
    fill_value = 50
    random_rotation_op = vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center,
                                               fill_value=fill_value)
    _ = random_rotation_op(image)
    image.close()

    # Test RandomRotation func resample is mode.Inter.BICUBIC
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    image = Image.open(image_png)
    degrees = (0, 360.5)
    resample = mode.Inter.BICUBIC
    expand = False
    fill_value = (0, 120, 255)
    random_rotation_op = vision.RandomRotation(degrees=degrees, resample=resample, expand=expand,
                                               fill_value=fill_value)
    _ = random_rotation_op(image)
    image.close()


def test_random_rotation_operation_03():
    """
    Feature: RandomRotation operation
    Description: Testing the normal functionality of the RandomRotation operator
    Expectation: The Output is equal to the expected output
    """
    # Test RandomRotation func resample is mode.Inter.AREA
    image = np.random.randn(212, 31, 3)
    degrees = (120, 120)
    resample = mode.Inter.AREA
    expand = True
    center = (500, 500)
    fill_value = (0, 120, 255)
    random_rotation_op = vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center,
                                               fill_value=fill_value)
    _ = random_rotation_op(image)

    # Test RandomRotation func resample is mode.Inter.BILINEAR
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image = Image.open(image_jpg)
    degrees = 361
    resample = mode.Inter.BILINEAR
    expand = True
    center = (0, 0)
    fill_value = (20, 10, 100)
    random_rotation_op = vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center,
                                               fill_value=fill_value)
    _ = random_rotation_op(image)
    image.close()

    # Test RandomRotation func image.shape is (256, 256, 3)
    image = np.random.randn(256, 256, 3)
    degrees = [-180, 180]
    resample = mode.Inter.BILINEAR
    expand = False
    center = (10, 10)
    fill_value = 128
    random_rotation_op = vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center,
                                               fill_value=fill_value)
    _ = random_rotation_op(image)

    # Test RandomRotation func image.shape is (300, 150, 1)
    image = np.random.randint(0, 255, (300, 150, 1)).astype(np.uint8)
    degrees = [0, 0]
    resample = mode.Inter.BILINEAR
    expand = False
    center = (1000, 1000)
    fill_value = 128
    random_rotation_op = vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center,
                                               fill_value=fill_value)
    random_rotation_op(image)

    # Test RandomRotation func image is 2d
    image = np.random.randint(-255, 255, (20, 18)).astype(np.int32)
    degrees = [10.5, 250]
    random_rotation_op = vision.RandomRotation(degrees=degrees)
    _ = random_rotation_op(image)

    # Test RandomRotation func center is (-10, 10)
    image = np.random.randn(14, 50)
    degrees = [10.5, 250]
    center = (-10, 10)
    random_rotation_op = vision.RandomRotation(degrees=degrees, center=center)
    _ = random_rotation_op(image)

    # Test RandomRotation func image.shape is (1024, 560, 4)
    image = np.random.randn(1024, 560, 4)
    degrees = [-450, -250]
    expand = True
    random_rotation_op = vision.RandomRotation(degrees=degrees, expand=expand)
    _ = random_rotation_op(image)


def test_random_rotation_exception_01():
    """
    Feature: RandomRotation operation
    Description: Testing the RandomRotation Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test RandomRotation func image is 1d
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image = np.fromfile(image_jpg, dtype=np.int32)
    degrees = 250
    fill_value = 12
    random_rotation_op = vision.RandomRotation(degrees=degrees, fill_value=fill_value)
    with pytest.raises(RuntimeError, match="Rotate: input tensor is not in shape of <H,W> or <H,W,C>, but got rank: 1. "
                                           "You may need to perform Decode first."):
        random_rotation_op(image)

    # Test RandomRotation func degrees is (100,0)
    degrees = (100, 0)
    with pytest.raises(ValueError, match="degrees should be in \\(min,max\\) format. Got \\(max,min\\)."):
        vision.RandomRotation(degrees=degrees)

    # Test RandomRotation func degrees is -10
    degrees = -10
    with pytest.raises(ValueError, match="Input degrees is not within the required interval"):
        vision.RandomRotation(degrees=degrees)

    # Test RandomRotation func degrees is ""
    degrees = ""
    with pytest.raises(TypeError, match="""Argument degrees with value "" is not of type \\[<class 'int'>, """
                                        """<class 'float'>, <class 'list'>, <class 'tuple'>]"""):
        vision.RandomRotation(degrees=degrees)

    # Test RandomRotation func resample is ""
    degrees = 100
    resample = ""
    with pytest.raises(TypeError, match="""Argument resample with value "" is not of type \\[<enum 'Inter'>\\], """
                                        """but got <class 'str'>."""):
        vision.RandomRotation(degrees=degrees, resample=resample)

    # Test RandomRotation func expand is ""
    degrees = 100
    resample = mode.Inter.BILINEAR
    expand = ""
    with pytest.raises(TypeError, match="""Argument expand with value "" is not of type \\[<class 'bool'>\\], """
                                        """but got <class 'str'>."""):
        vision.RandomRotation(degrees=degrees, resample=resample, expand=expand)

    # Test RandomRotation func  center is ""
    degrees = 100
    resample = mode.Inter.BILINEAR
    expand = True
    center = ""
    with pytest.raises(ValueError, match="Value center needs to be a 2-tuple"):
        vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center)

    # Test RandomRotation func  fill_value is -1
    degrees = 100
    resample = mode.Inter.BILINEAR
    expand = True
    center = (100, 200)
    fill_value = -1
    with pytest.raises(ValueError, match=r"Input fill_value is not within the required interval of \[0, 255\]."):
        vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center, fill_value=fill_value)

    # Test RandomRotation func  fill_value is (100,100)
    degrees = 100
    resample = mode.Inter.BILINEAR
    expand = True
    center = (100, 200)
    fill_value = (100, 100)
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center,
                               fill_value=fill_value)

    # Test RandomRotation func  fill_value is ""
    degrees = 100
    resample = mode.Inter.BILINEAR
    expand = True
    center = (100, 200)
    fill_value = ""
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center,
                               fill_value=fill_value)

    # Test RandomRotation func no para
    with pytest.raises(TypeError, match="missing a required argument"):
        vision.RandomRotation()

    # Test RandomRotation func no 1st para
    resample = mode.Inter.BILINEAR
    expand = True
    center = (100, 200)
    fill_value = (100, 100, 100)
    with pytest.raises(TypeError, match="missing a required argument"):
        vision.RandomRotation(resample=resample, expand=expand, center=center, fill_value=fill_value)

    # Test RandomRotation func more para
    degrees = 100
    resample = mode.Inter.BILINEAR
    expand = True
    center = (100, 200)
    fill_value = (100, 100, 100)
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        vision.RandomRotation(degrees, resample, expand, center, fill_value, more_para)


def test_random_rotation_exception_02():
    """
    Feature: RandomRotation operation
    Description: Testing the RandomRotation Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # resample is mode.Inter.ANTIALIAS
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image = Image.open(image_jpg)
    degrees = 180
    resample = mode.Inter.ANTIALIAS
    expand = True
    center = (50.5, 61.2)
    fill_value = (100, 100, 100)
    random_rotation_op = vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center,
                                               fill_value=fill_value)
    with pytest.raises(ValueError, match="When using Inter.ANTIALIAS, center needs to be None and angle needs to be "
                                         "an integer multiple of 90."):
        random_rotation_op(image)
        image.close()

    # Test RandomRotation func image.shape is (200, 200, 3, 3)
    image = np.random.randn(200, 200, 3, 3)
    degrees = 250
    random_rotation_op = vision.RandomRotation(degrees=degrees)
    with pytest.raises(RuntimeError):
        random_rotation_op(image)

    # Test RandomRotation func image is list
    image = np.random.randn(200, 200, 3).tolist()
    degrees = 50
    random_rotation_op = vision.RandomRotation(degrees=degrees)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        random_rotation_op(image)

    # Test RandomRotation func image is 10
    image = 10
    degrees = 50
    random_rotation_op = vision.RandomRotation(degrees=degrees)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'int'>."):
        random_rotation_op(image)

    # Test RandomRotation func image is tuple
    image = tuple(np.random.randn(200, 200, 3).tolist())
    degrees = 50
    random_rotation_op = vision.RandomRotation(degrees=degrees)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>."):
        random_rotation_op(image)

    # Test RandomRotation func degrees is (10, 16777216.1)
    degrees = (10, 16777216.1)
    resample = mode.Inter.BILINEAR
    expand = True
    center = (1, 1)
    fill_value = 128
    with pytest.raises(ValueError,
                       match="Input degrees is not within the required interval of \\[-16777216, 16777216\\]."):
        vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center,
                               fill_value=fill_value)

    # Test RandomRotation func degrees is 1-tuple
    degrees = (50,)
    with pytest.raises(TypeError, match="If degrees is a sequence, the length must be 2."):
        vision.RandomRotation(degrees=degrees)

    # Test RandomRotation func degrees is 3-list
    degrees = [50, 100, 150]
    with pytest.raises(TypeError, match="If degrees is a sequence, the length must be 2."):
        vision.RandomRotation(degrees=degrees)

    # Test RandomRotation func degrees is np
    degrees = np.array([50, 100])
    with pytest.raises(TypeError, match="Argument degrees with value \\[ 50 100\\] is not of "
                                        "type \\[<class 'int'>, <class 'float'>, <class 'list'>, "
                                        "<class 'tuple'>\\], but got <class 'numpy.ndarray'>."):
        vision.RandomRotation(degrees=degrees)

    # Test RandomRotation func degrees is "100"
    degrees = "100"
    with pytest.raises(TypeError, match="Argument degrees with value 100 is not of type \\[<class 'int'>, <class "
                                        "'float'>, <class 'list'>, <class 'tuple'>\\], but got <class 'str'>."):
        vision.RandomRotation(degrees=degrees)

    # Test RandomRotation func resample is mode.Inter
    degrees = 100
    resample = mode.Inter
    with pytest.raises(TypeError,
                       match="Argument resample with value <enum 'Inter'> is not of type \\[<enum 'Inter'>\\]."):
        vision.RandomRotation(degrees=degrees, resample=resample)

    # Test RandomRotation func resample is 10
    degrees = 100
    resample = 10
    with pytest.raises(TypeError, match="Argument resample with value 10 is not of type \\[<enum 'Inter'>\\]."):
        vision.RandomRotation(degrees=degrees, resample=resample)

    # Test RandomRotation func resample is list
    degrees = 100
    resample = [mode.Inter.BILINEAR]
    with pytest.raises(TypeError, match="Argument resample with value \\[<Inter.BILINEAR: 2>\\] "
                                        "is not of type \\[<enum 'Inter'>\\]."):
        vision.RandomRotation(degrees=degrees, resample=resample)


def test_random_rotation_exception_03():
    """
    Feature: RandomRotation operation
    Description: Testing the RandomRotation Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test RandomRotation func expand is 1
    degrees = 100
    resample = mode.Inter.BILINEAR
    expand = 1
    with pytest.raises(TypeError, match="Argument expand with value 1 is not of type \\[<class 'bool'>\\]."):
        vision.RandomRotation(degrees=degrees, resample=resample, expand=expand)

    # Test RandomRotation func expand is list
    degrees = 50
    resample = mode.Inter.BILINEAR
    expand = [True]
    with pytest.raises(TypeError, match="Argument expand with value \\[True\\] is not of type \\[<class 'bool'>\\]."):
        vision.RandomRotation(degrees=degrees, resample=resample, expand=expand)

    # Test RandomRotation func expand is bool
    degrees = 50
    resample = mode.Inter.BILINEAR
    expand = bool
    with pytest.raises(TypeError,
                       match="Argument expand with value <class 'bool'> is not of type \\[<class 'bool'>\\]."):
        vision.RandomRotation(degrees=degrees, resample=resample, expand=expand)

    # Test RandomRotation func center is list
    degrees = 50
    center = [10, 10]
    with pytest.raises(ValueError, match="Value center needs to be a 2-tuple."):
        vision.RandomRotation(degrees=degrees, center=center)

    # Test RandomRotation func center is 3-tuple
    degrees = 50
    center = (10, 10, 15)
    with pytest.raises(ValueError, match="Value center needs to be a 2-tuple."):
        vision.RandomRotation(degrees=degrees, center=center)

    # Test RandomRotation func center is str
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image = Image.open(image_jpg)
    degrees = 50
    center = (10, "10")
    with pytest.raises(TypeError, match="Argument center with value 10 is not of type \\[<class "
                                        "'int'>, <class 'float'>\\], but got <class 'str'>."):
        random_rotation_op = vision.RandomRotation(degrees=degrees, center=center)
        random_rotation_op(image)
        image.close()

    # Test RandomRotation func fill_value is 256
    degrees = 50
    center = (10, 10)
    fill_value = (10, 100, 256)
    with pytest.raises(ValueError, match=r"Input fill_value\[2\] is not within the required interval of \[0, 255\]."):
        vision.RandomRotation(degrees=degrees, center=center, fill_value=fill_value)

    # Test RandomRotation func fill_value is 4-tuple
    degrees = 50
    fill_value = (10, 100, 255, 45)
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.RandomRotation(degrees=degrees, fill_value=fill_value)

    # Test RandomRotation func fill_value is list
    degrees = 50
    fill_value = [10, 100, 255]
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.RandomRotation(degrees=degrees, fill_value=fill_value)

    # Test RandomRotation func fill_value is float
    degrees = 150
    fill_value = (10, 100, 25.5)
    with pytest.raises(TypeError, match=r"Argument fill_value\[2\] with value 25.5 is not of type \[<class 'int'>\], "
                                        r"but got <class 'float'>."):
        vision.RandomRotation(degrees=degrees, fill_value=fill_value)

    # Test RandomRotation func degrees is (-16777216.1, 200)
    degrees = (-16777216.1, 200)
    resample = mode.Inter.BILINEAR
    expand = True
    center = (1, 1)
    fill_value = 128
    with pytest.raises(ValueError,
                       match="Input degrees is not within the required interval of \\[-16777216, 16777216\\]."):
        vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center,
                               fill_value=fill_value)

    # resample is mode.Inter.BICUBIC ,input Numpy data
    image = np.random.randn(2, 3, 6)
    degrees = 180
    resample = mode.Inter.BICUBIC
    expand = True
    center = (50.5, 61.2)
    fill_value = (100, 100, 100)
    random_rotation_op = vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center,
                                               fill_value=fill_value)
    with pytest.raises(RuntimeError, match="interpolation can not be CUBIC when image channel is greater than 4."):
        random_rotation_op(image)


def test_random_rotation_exception_04():
    """
    Feature: RandomRotation operation
    Description: Testing the RandomRotation Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # resample is mode.Inter.BICUBIC ,input PIL data
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    image = Image.open(image_png)
    degrees = 180
    resample = mode.Inter.AREA
    expand = True
    center = (50.5, 61.2)
    fill_value = (100, 100, 100)
    random_rotation_op = vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center,
                                               fill_value=fill_value)
    with pytest.raises(TypeError, match="Current Interpolation is not supported with PIL input."):
        random_rotation_op(image)
        image.close()

    # resample is mode.Inter.ANTIALIAS ,input Numpy
    image = np.random.randn(2, 3, 6)
    degrees = 180
    resample = mode.Inter.ANTIALIAS
    expand = True
    center = (50.5, 61.2)
    fill_value = (100, 100, 100)
    random_rotation_op = vision.RandomRotation(degrees=degrees, resample=resample, expand=expand, center=center,
                                               fill_value=fill_value)
    with pytest.raises(TypeError, match=r"img should be PIL image. Got <class 'numpy.ndarray'>. Use Decode\(\) for "
                                        r"encoded data or ToPIL\(\) for decoded data."):
        random_rotation_op(image)

    # Test RandomRotation func image is list,use list()
    image = list(np.random.randn(200, 200, 3))
    degrees = 50
    random_rotation_op = vision.RandomRotation(degrees=degrees)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        random_rotation_op(image)


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
    test_random_rotation_operation_01()
    test_random_rotation_operation_02()
    test_random_rotation_operation_03()
    test_random_rotation_exception_01()
    test_random_rotation_exception_02()
    test_random_rotation_exception_03()
    test_random_rotation_exception_04()
