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
Testing CutOut op in DE
"""
import cv2
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.transforms.transforms as t_trans
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import visualize_image, visualize_list, diff_mse, save_and_check_md5, save_and_check_md5_pil, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False
TEST_DATA_DATASET_FUNC ="../data/dataset/"


def dir_data():
    """Obtain the dataset"""
    data_list = []
    data_dir1 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train")
    data_dir2 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train", "class1", "1_1.jpg")
    data_dir3 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    data_dir4 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    data_dir5 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    data_dir6 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    data_list.append(data_dir1)
    data_list.append(data_dir2)
    data_list.append(data_dir3)
    data_list.append(data_dir4)
    data_list.append(data_dir5)
    data_list.append(data_dir6)
    return data_list


def test_cut_out_op(plot=False):
    """
    Feature: CutOut op
    Description: Test CutOut op by comparing between Python and Cpp implementation
    Expectation: Both outputs are equal to each other
    """
    logger.info("test_cut_out")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    transforms_1 = [
        vision.Decode(True),
        vision.ToTensor(),
        vision.RandomErasing(value='random')
    ]
    transform_1 = mindspore.dataset.transforms.Compose(transforms_1)
    data1 = data1.map(operations=transform_1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    cut_out_op = vision.CutOut(80, is_hwc=True)

    transforms_2 = [
        decode_op,
        cut_out_op
    ]

    data2 = data2.map(operations=transforms_2, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        image_1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        # C image doesn't require transpose
        image_2 = item2["image"]

        logger.info("shape of image_1: {}".format(image_1.shape))
        logger.info("shape of image_2: {}".format(image_2.shape))

        logger.info("dtype of image_1: {}".format(image_1.dtype))
        logger.info("dtype of image_2: {}".format(image_2.dtype))

        mse = diff_mse(image_1, image_2)
        if plot:
            visualize_image(image_1, image_2, mse)


def test_cut_out_op_multicut(plot=False):
    """
    Feature: CutOut op
    Description: Test CutOut where Python is implemented without RandomErasing and Cpp is implemented with num_patches
    Expectation: Both outputs are equal to each other
    """
    logger.info("test_cut_out")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    transforms_1 = [
        vision.Decode(True),
        vision.ToTensor(),
    ]
    transform_1 = mindspore.dataset.transforms.Compose(transforms_1)
    data1 = data1.map(operations=transform_1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    cut_out_op = vision.CutOut(80, num_patches=10, is_hwc=True)

    transforms_2 = [
        decode_op,
        cut_out_op
    ]

    data2 = data2.map(operations=transforms_2, input_columns=["image"])

    num_iter = 0
    image_list_1, image_list_2 = [], []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        image_1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        # C image doesn't require transpose
        image_2 = item2["image"]
        image_list_1.append(image_1)
        image_list_2.append(image_2)

        logger.info("shape of image_1: {}".format(image_1.shape))
        logger.info("shape of image_2: {}".format(image_2.shape))

        logger.info("dtype of image_1: {}".format(image_1.dtype))
        logger.info("dtype of image_2: {}".format(image_2.dtype))
    if plot:
        visualize_list(image_list_1, image_list_2)


def test_cut_out_md5():
    """
    Feature: CutOut op
    Description: Test CutOut with md5 comparison check
    Expectation: Passes the md5 check test
    """
    logger.info("test_cut_out_md5")
    original_seed = config_get_set_seed(2)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    cut_out_op = vision.CutOut(100, is_hwc=True)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=cut_out_op, input_columns=["image"])

    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.ToTensor(),
        vision.CutOut(100, is_hwc=False)
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    data2 = data2.map(operations=transform, input_columns=["image"])

    # Compare with expected md5 from images
    filename1 = "cut_out_01_c_result.npz"
    save_and_check_md5(data1, filename1, generate_golden=GENERATE_GOLDEN)
    filename2 = "cut_out_02_c_result.npz"
    save_and_check_md5_pil(data2, filename2, generate_golden=GENERATE_GOLDEN)

    # Restore config
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_cut_out_comp_hwc(plot=False):
    """
    Feature: CutOut op
    Description: Test CutOut with HWC input, Decode(to_pil=True) & ToTensor versus Decode(to_pil=False) comparison
    Expectation: Test succeeds. Manual confirmation of logged info. Manual visualization confirmation
    """
    logger.info("test_cut_out_comp")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    transforms_1 = [
        vision.Decode(True),
        vision.ToTensor(),
        vision.CutOut(250, is_hwc=False)
    ]
    transform_1 = mindspore.dataset.transforms.Compose(transforms_1)
    data1 = data1.map(operations=transform_1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    transforms_2 = [
        vision.Decode(),
        vision.CutOut(250, is_hwc=True)
    ]

    data2 = data2.map(operations=transforms_2, input_columns=["image"])

    num_iter = 0
    image_list_1, image_list_2 = [], []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        image_1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        # C image doesn't require transpose
        image_2 = item2["image"]
        image_list_1.append(image_1)
        image_list_2.append(image_2)

        logger.info("shape of image_1: {}".format(image_1.shape))
        logger.info("shape of image_2: {}".format(image_2.shape))

        logger.info("dtype of image_1: {}".format(image_1.dtype))
        logger.info("dtype of image_2: {}".format(image_2.dtype))
    if plot:
        visualize_list(image_list_2, image_list_1, visualize_mode=2)


def test_cut_out_comp_chw(plot=False):
    """
    Feature: CutOut op
    Description: Test CutOut with CHW input, Decode(to_pil=True) & ToTensor versus Decode(to_pil=False) & HWC2CHW
                 comparison
    Expectation: Test succeeds. Manual confirmation of logged info
    """
    logger.info("test_cut_out_comp_chw")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    transforms_1 = [
        vision.Decode(),
        vision.HWC2CHW(),
        vision.CutOut(500, num_patches=3, is_hwc=False)
    ]
    transform_1 = mindspore.dataset.transforms.Compose(transforms_1)
    data1 = data1.map(operations=transform_1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    transforms_2 = [
        vision.Decode(True),
        vision.ToTensor(),
        vision.CutOut(500, num_patches=5, is_hwc=False)
    ]

    data2 = data2.map(operations=transforms_2, input_columns=["image"])

    num_iter = 0
    image_list_1, image_list_2 = [], []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        image_1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        if image_1.shape != image_2.shape:
            raise RuntimeError("image_1.shape != image_2.shape: " + str(image_1.shape) + " " + str(image_2.shape))
        image_list_1.append(image_1)
        image_list_2.append(image_2)

        logger.info("shape of image_1: {}".format(image_1.shape))
        logger.info("shape of image_2: {}".format(image_2.shape))

        logger.info("dtype of image_1: {}".format(image_1.dtype))
        logger.info("dtype of image_2: {}".format(image_2.dtype))

    if plot:
        visualize_list(image_list_1, image_list_2, visualize_mode=1)


def test_cutout_4channel_chw():
    """
    Feature: CutOut op
    Description: Test CutOut Op with multi-channel CHW input
    Expectation: Test succeeds.
    """
    image = np.random.randn(4, 1024, 856).astype(np.uint8)
    op = vision.CutOut(length=500, num_patches=3, is_hwc=False)
    op(image)


def test_cutout_4channel_hwc():
    """
    Feature: CutOut op
    Description: Test CutOut Op with multi-channel HWC input
    Expectation: Test succeeds.
    """
    image = np.random.randn(1024, 856, 4).astype(np.uint8)
    op = vision.CutOut(length=500, num_patches=3, is_hwc=True)
    op(image)


def test_cut_out_validation():
    """
    Feature: CutOut op
    Description: Test CutOut Op with patch length greater than image dimensions
    Expectation: Raises an exception
    """
    image = np.random.randn(3, 1024, 856).astype(np.uint8)
    op = vision.CutOut(length=1500, num_patches=3, is_hwc=False)
    with pytest.raises(RuntimeError) as errinfo:
        op(image)
    assert 'box size is too large for image erase' in str(errinfo.value)


def test_cutout_operation_01():
    """
    Feature: CutOut operation
    Description: Testing the normal functionality of the CutOut operator
    Expectation: The Output is equal to the expected output
    """
    # CutOut Normal Scenario: Test length is 100
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    length = 100
    cutout_op = vision.CutOut(length=length)
    dataset2 = dataset2.map(input_columns=["image"], operations=cutout_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # CutOut Normal Scenario: Test length is 1
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    length = 1
    cutout_op = vision.CutOut(length=length)
    dataset2 = dataset2.map(input_columns=["image"], operations=cutout_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # CutOut Normal Scenario: Test num_patches is 2
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    length = 100
    num_patches = 2
    cutout_op = vision.CutOut(length=length, num_patches=num_patches)
    dataset2 = dataset2.map(input_columns=["image"], operations=cutout_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # CutOut Normal Scenario: Test all parameters
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    length = 100
    num_patches = 1
    cutout_op = vision.CutOut(length=length, num_patches=num_patches)
    dataset2 = dataset2.map(input_columns=["image"], operations=cutout_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # CutOut Normal Scenario: Test length is 50
    with Image.open(dir_data()[1]) as image:
        cutout_op = vision.CutOut(50, 2)
        _ = cutout_op(image)

    # CutOut Normal Scenario: Test length is 2.
    with Image.open(dir_data()[1]) as image:
        cutout_op = vision.CutOut(2, 200)
        _ = cutout_op(image)

    # CutOut Normal Scenario: Test length is 718.
    image = cv2.imread(dir_data()[1])
    cutout_op = vision.CutOut(718, 1)
    _ = cutout_op(image)

    # CutOut Normal Scenario: num_patches is 1024
    image = cv2.imread(dir_data()[1])
    cutout_op = vision.CutOut(118, 1024)
    _ = cutout_op(image)

    # CutOut Normal Scenario: Test length is 200, num_patches is 20
    image = np.random.randint(0, 255, (1500, 1500, 3)).astype(np.uint8)
    cutout_op = vision.CutOut(200, 20)
    _ = cutout_op(image)

    # CutOut Normal Scenario: Test length is 1
    image = np.random.randint(0, 255, (32, 32, 3)).astype(np.uint8)
    cutout_op = vision.CutOut(1)
    _ = cutout_op(image)

    # CutOut Normal Scenario: Test is_hwc is True
    image = np.random.randint(0, 255, (1500, 1500, 3)).astype(np.uint8)
    cutout_op = vision.CutOut(length=200, num_patches=20, is_hwc=True)
    _ = cutout_op(image)

    # CutOut Normal Scenario: input.shape is (3, 1500, 1500), Test is_hwc is False
    image = np.random.randint(0, 255, (3, 1500, 1500)).astype(np.uint8)
    cutout_op = vision.CutOut(length=200, num_patches=20, is_hwc=False)
    _ = cutout_op(image)


def test_cutout_operation_02():
    """
    Feature: CutOut operation
    Description: Testing the normal functionality of the CutOut operator
    Expectation: The Output is equal to the expected output
    """
    # CutOut Normal Scenario: Test normal
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    length = 100
    num_patches = 2
    transforms1 = [
        vision.Decode(to_pil=True),
        vision.ToTensor(),
        vision.CutOut(length=length, num_patches=num_patches, is_hwc=False)
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for _ in ds2.create_dict_iterator(output_numpy=True):
        pass

    # CutOut Normal Scenario: Test length is 1
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    length = 1
    transforms1 = [
        vision.Decode(to_pil=True),
        vision.ToTensor(),
        vision.CutOut(length=length, is_hwc=False)
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for _ in ds2.create_dict_iterator(output_numpy=True):
        pass

    # CutOut Normal Scenario: Test num_patches is 2
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    length = 100
    num_patches = 2
    transforms1 = [
        vision.Decode(to_pil=True),
        vision.ToTensor(),
        vision.CutOut(length=length, num_patches=num_patches, is_hwc=False)
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for _ in ds2.create_dict_iterator(output_numpy=True):
        pass

    # CutOut Normal Scenario: Test all parameters
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    length = 100
    num_patches = 2
    transforms1 = [
        vision.Decode(to_pil=True),
        vision.ToTensor(),
        vision.CutOut(length=length, num_patches=num_patches, is_hwc=False)
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for _ in ds2.create_dict_iterator(output_numpy=True):
        pass


def test_cutout_operation_03():
    """
    Feature: CutOut operation
    Description: Testing the normal functionality of the CutOut operator
    Expectation: The Output is equal to the expected output
    """
    # CutOut Normal Scenario: Test no 2 parameters
    image = np.transpose(np.array(Image.open(dir_data()[3])), (2, 0, 1))
    length = 50
    num_patches = 2
    cutout_op = vision.CutOut(length=length, num_patches=num_patches, is_hwc=False)
    _ = cutout_op(image)

    # CutOut Normal Scenario: Test no num_patches parameters
    image = np.transpose(np.array(Image.open(dir_data()[2])), (2, 0, 1))
    length = 128
    cutout_op = vision.CutOut(length=length, is_hwc=False)
    _ = cutout_op(image)

    # CutOut Normal Scenario: Test length is 1, num_patches is 1000
    image = np.random.randn(3, 28, 38)
    length = 1
    num_patches = 1000
    cutout_op = vision.CutOut(length=length, num_patches=num_patches, is_hwc=False)
    _ = cutout_op(image)

    # CutOut Normal Scenario: Test length is 56, num_patches is 12
    image = np.random.randn(1, 128, 128)
    length = 56
    num_patches = 12
    cutout_op = vision.CutOut(length=length, num_patches=num_patches, is_hwc=False)
    _ = cutout_op(image)

    # CutOut Normal Scenario: input.shape is (4, 1024, 856)
    image = np.random.randn(4, 1024, 856)
    length = 500
    num_patches = 3
    cutout_op = vision.CutOut(length=length, num_patches=num_patches, is_hwc=False)
    _ = cutout_op(image)


def test_cutout_exception_01():
    """
    Feature: CutOut operation
    Description: Testing the CutOut Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # CutOut Exception Scenario: Test length is 0
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    length = 0
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        cutout_op = vision.CutOut(length=length)
        dataset2 = dataset2.map(input_columns=["image"], operations=cutout_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # CutOut Exception Scenario: Test length is 1000
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    length = 1000
    with pytest.raises(RuntimeError, match="box size is too large for image erase"):
        cutout_op = vision.CutOut(length=length)
        dataset2 = dataset2.map(input_columns=["image"], operations=cutout_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # CutOut Exception Scenario: Test length is 100
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    length = 100.
    with pytest.raises(TypeError, match="Argument length with value 100.0 is not of type \\[<class 'int'>\\]"):
        cutout_op = vision.CutOut(length=length)
        dataset2 = dataset2.map(input_columns=["image"], operations=cutout_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # CutOut Exception Scenario: Test length is (100, 200)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    length = (100, 200)
    with pytest.raises(TypeError,
                       match="Argument length with value \\(100, 200\\) is not of type \\[<class 'int'>\\]."):
        cutout_op = vision.CutOut(length=length)
        dataset2 = dataset2.map(input_columns=["image"], operations=cutout_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # CutOut Exception Scenario: Test length is [100, 200]
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    length = [100, 200]
    with pytest.raises(TypeError,
                       match="Argument length with value \\[100, 200\\] is not of type \\[<class 'int'>\\]."):
        cutout_op = vision.CutOut(length=length)
        dataset2 = dataset2.map(input_columns=["image"], operations=cutout_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # CutOut Exception Scenario: Test length is ''
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    length = ""
    with pytest.raises(TypeError, match="Argument length with value"):
        cutout_op = vision.CutOut(length=length)
        dataset2 = dataset2.map(input_columns=["image"], operations=cutout_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # CutOut Exception Scenario: Test num_patches is 0
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    length = 100
    num_patches = 0
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        cutout_op = vision.CutOut(length=length, num_patches=num_patches)
        dataset2 = dataset2.map(input_columns=["image"], operations=cutout_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

def test_cutout_exception_02():
    """
    Feature: CutOut operation
    Description: Testing the CutOut Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # CutOut Exception Scenario: Test num_patches is 16777217
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    length = 100
    num_patches = 16777217
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        cutout_op = vision.CutOut(length=length, num_patches=num_patches)
        dataset2 = dataset2.map(input_columns=["image"], operations=cutout_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # CutOut Exception Scenario: Test num_patches is ''
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    length = 100
    num_patches = ""
    with pytest.raises(TypeError, match="Argument num_patches with value"):
        cutout_op = vision.CutOut(length=length, num_patches=num_patches)
        dataset2 = dataset2.map(input_columns=["image"], operations=cutout_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # CutOut Exception Scenario: Test no parameters
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    with pytest.raises(TypeError, match="missing a required argument"):
        cutout_op = vision.CutOut()
        dataset2 = dataset2.map(input_columns=["image"], operations=cutout_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # CutOut Exception Scenario: Test no 1 parameters
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    num_patches = 1
    with pytest.raises(TypeError, match="missing a required argument"):
        cutout_op = vision.CutOut(num_patches=num_patches)
        dataset2 = dataset2.map(input_columns=["image"], operations=cutout_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # CutOut Exception Scenario: Test length is ''
    with pytest.raises(TypeError, match="missing a required argument: 'length'"):
        vision.CutOut()

    # CutOut Exception Scenario: Test length is 0
    with pytest.raises(ValueError, match="Input is not within the required interval of"):
        vision.CutOut(0, 2)

    # CutOut Exception Scenario: Test length is 16777217
    with pytest.raises(ValueError, match="Input is not within the required interval of"):
        vision.CutOut(16777217, 1)

    # CutOut Exception Scenario: Test length is -10
    with pytest.raises(ValueError, match="Input is not within the required interval of"):
        vision.CutOut(-10, 1)

    # CutOut Exception Scenario: Test length is 50.5
    with pytest.raises(TypeError, match="Argument length with value 50.5 is not of type \\[<class 'int'>\\]."):
        vision.CutOut(length=50.5, num_patches=1)

    # CutOut Exception Scenario: Test length is [100]
    with pytest.raises(TypeError, match="Argument length with value \\[100\\] is not of type \\[<class 'int'>\\]."):
        vision.CutOut([100], 1)

    # CutOut Normal Scenario: Test length is tuple
    with pytest.raises(TypeError, match="Argument length with value \\(50,\\) is not of type \\[<class 'int'>\\]."):
        vision.CutOut((50,), 1)

    # CutOut Exception Scenario: Test length is None
    with pytest.raises(TypeError, match="Argument length with value None is not of type \\[<class 'int'>\\]."):
        vision.CutOut(length=None, num_patches=2)

    # CutOut Exception Scenario: Test num_patches is 0
    with pytest.raises(ValueError, match="Input is not within the required interval of"):
        vision.CutOut(length=100, num_patches=0)

    # CutOut Exception Scenario: Test num_patches is 16777217
    with pytest.raises(ValueError, match="Input is not within the required interval of"):
        vision.CutOut(length=100, num_patches=16777217)

    # CutOut Exception Scenario: Test num_patches is 1.5
    with pytest.raises(TypeError, match="Argument num_patches with value 1.5 is not of type \\[<class 'int'>\\]."):
        vision.CutOut(length=100, num_patches=1.5)

    # CutOut Exception Scenario: Test num_patches is tuple  (2,)
    with pytest.raises(TypeError, match="Argument num_patches with value \\(2,\\) is not of type \\[<class 'int'>\\]"):
        vision.CutOut(length=100, num_patches=(2,))


def test_cutout_exception_03():
    """
    Feature: CutOut operation
    Description: Testing the CutOut Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # CutOut Exception Scenario: Test num_patches is list
    with pytest.raises(TypeError, match="Argument num_patches with value \\[3\\] is not of type \\[<class 'int'>\\]"):
        vision.CutOut(length=100, num_patches=[3])

    # CutOut Exception Scenario: Test length 大于 image.shape
    image = np.random.randn(128, 64, 3)
    cutout_op = vision.CutOut(65)
    with pytest.raises(RuntimeError, match="CutOut: box size is too large for image erase"):
        cutout_op(image)

    # CutOut Exception Scenario: Test input is 2d
    image = np.random.randn(128, 128)
    cutout_op = vision.CutOut(65)
    with pytest.raises(RuntimeError, match="CutOut: shape is invalid."):
        cutout_op(image)

    # CutOut Exception Scenario: Test input is 4d
    image = np.random.randn(128, 128, 3, 3)
    cutout_op = vision.CutOut(65)
    with pytest.raises(RuntimeError, match="CutOut: image shape is not <H,W,C>."):
        cutout_op(image)

    # CutOut Exception Scenario: Test input is list
    image = [10, 10, 10]
    cutout_op = vision.CutOut(65)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        cutout_op(image)

    # CutOut Exception Scenario: Test input is int
    image = 20
    cutout_op = vision.CutOut(65)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'int'>."):
        cutout_op(image)

    # CutOut Exception Scenario: Test input is tuple
    image = (20, 30)
    cutout_op = vision.CutOut(65)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>."):
        cutout_op(image)

    # CutOut Exception Scenario: Test input channel is 4
    image = np.random.randn(128, 128, 4)
    cutout_op = vision.CutOut(65)
    _ = cutout_op(image)

    # CutOut Exception Scenario: input.shape is (5, 1500, 1500), Test is_hwc is False
    image = np.random.randint(0, 255, (5, 1500, 1500)).astype(np.uint8)
    cutout_op = vision.CutOut(length=200, num_patches=20, is_hwc=False)
    _ = cutout_op(image)

    # CutOut Exception Scenario: input.shape is (3, 1500, 1500), Test is_hwc is 1.
    image = np.random.randint(0, 255, (1500, 1500, 3)).astype(np.uint8)
    with pytest.raises(TypeError,
                       match="Argument is_hwc with value 1 is not of type \\[<class 'bool'>\\],"
                             " but got <class 'int'>."):
        cutout_op = vision.CutOut(length=200, num_patches=20, is_hwc=1)
        cutout_op(image)

    # CutOut Exception Scenario: input.shape is (3, 1500, 1500), Test is_hwc is 'a'.
    image = np.random.randint(0, 255, (1500, 1500, 3)).astype(np.uint8)
    with pytest.raises(TypeError,
                       match="Argument is_hwc with value a is not of type \\[<class 'bool'>\\],"
                             " but got <class 'str'>."):
        cutout_op = vision.CutOut(length=200, num_patches=20, is_hwc='a')
        cutout_op(image)

    # CutOut Exception Scenario: input.shape is (3, 1500, 1500), Test is_hwc is None.
    image = np.random.randint(0, 255, (1500, 1500, 3)).astype(np.uint8)
    with pytest.raises(TypeError,
                       match="Argument is_hwc with value None is not of type \\[<class 'bool'>\\],"
                             " but got <class 'NoneType'>."):
        cutout_op = vision.CutOut(length=200, num_patches=20, is_hwc=None)
        cutout_op(image)

    # CutOut Exception Scenario: Test length is 0
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        length = 0
        transforms1 = [
            vision.Decode(to_pil=True),
            vision.ToTensor(),
            vision.CutOut(length=length, is_hwc=False)
        ]
        transform1 = t_trans.Compose(transforms1)
        ds2 = ds2.map(input_columns=["image"], operations=transform1)
        for _ in ds2.create_dict_iterator():
            pass


def test_cutout_exception_04():
    """
    Feature: CutOut operation
    Description: Testing the CutOut Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # CutOut Exception Scenario: Test length is 16777217
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        length = 16777217
        transforms1 = [
            vision.Decode(to_pil=True),
            vision.ToTensor(),
            vision.CutOut(length=length, is_hwc=False)
        ]
        transform1 = t_trans.Compose(transforms1)
        ds2 = ds2.map(input_columns=["image"], operations=transform1)
        for _ in ds2.create_dict_iterator():
            pass

    # CutOut Exception Scenario: Test length is more than image size
    image = np.random.randn(3, 1024, 856).astype(np.uint8)
    op = vision.CutOut(length=1500, num_patches=3, is_hwc=False)
    with pytest.raises(RuntimeError, match=r"box size is too large for image erase"):
        op(image)

    # CutOut Exception Scenario: Test length is (100, 200)
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    with pytest.raises(TypeError,
                       match="Argument length with value \\(100, 200\\) is not of type \\[<class 'int'>\\]."):
        length = (100, 200)
        transforms1 = [
            vision.Decode(to_pil=True),
            vision.ToTensor(),
            vision.CutOut(length=length, is_hwc=False)
        ]
        transform1 = t_trans.Compose(transforms1)
        ds2 = ds2.map(input_columns=["image"], operations=transform1)
        for _ in ds2.create_dict_iterator():
            pass

    # CutOut Exception Scenario: Test length is [100, 200]
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    with pytest.raises(TypeError,
                       match="Argument length with value \\[100, 200\\] is not of type \\[<class 'int'>\\]."):
        length = [100, 200]
        transforms1 = [
            vision.Decode(to_pil=True),
            vision.ToTensor(),
            vision.CutOut(length=length, is_hwc=False)
        ]
        transform1 = t_trans.Compose(transforms1)
        ds2 = ds2.map(input_columns=["image"], operations=transform1)
        for _ in ds2.create_dict_iterator():
            pass

    # CutOut Exception Scenario: Test length is ''
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    with pytest.raises(TypeError, match="Argument length with value"):
        length = ""
        transforms1 = [
            vision.Decode(to_pil=True),
            vision.ToTensor(),
            vision.CutOut(length=length, is_hwc=False)
        ]
        transform1 = t_trans.Compose(transforms1)
        ds2 = ds2.map(input_columns=["image"], operations=transform1)
        for _ in ds2.create_dict_iterator():
            pass

    # CutOut Exception Scenario: Test num_patches is 0
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        length = 100
        num_patches = 0
        transforms1 = [
            vision.Decode(to_pil=True),
            vision.ToTensor(),
            vision.CutOut(length=length, num_patches=num_patches)
        ]
        transform1 = t_trans.Compose(transforms1)
        ds2 = ds2.map(input_columns=["image"], operations=transform1)
        for _ in ds2.create_dict_iterator():
            pass

    # CutOut Exception Scenario: Test num_patches is 16777217
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        length = 100
        num_patches = 16777217
        transforms1 = [
            vision.Decode(to_pil=True),
            vision.ToTensor(),
            vision.CutOut(length=length, num_patches=num_patches, is_hwc=False)
        ]
        transform1 = t_trans.Compose(transforms1)
        ds2 = ds2.map(input_columns=["image"], operations=transform1)
        for _ in ds2.create_dict_iterator():
            pass


def test_cutout_exception_05():
    """
    Feature: CutOut operation
    Description: Testing the CutOut Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # CutOut Exception Scenario: Test num_patches is ''
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    with pytest.raises(TypeError, match="is not of type \\[<class 'int'>\\]"):
        length = 100
        num_patches = ""
        transforms1 = [
            vision.Decode(to_pil=True),
            vision.ToTensor(),
            vision.CutOut(length=length, num_patches=num_patches, is_hwc=False)
        ]
        transform1 = t_trans.Compose(transforms1)
        ds2 = ds2.map(input_columns=["image"], operations=transform1)
        for _ in ds2.create_dict_iterator():
            pass

    # CutOut Exception Scenario: Test no parameters
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    with pytest.raises(TypeError, match="missing a required argument"):
        transforms1 = [
            vision.Decode(to_pil=True),
            vision.ToTensor(),
            vision.CutOut(is_hwc=False)
        ]
        transform1 = t_trans.Compose(transforms1)
        ds2 = ds2.map(input_columns=["image"], operations=transform1)
        for _ in ds2.create_dict_iterator():
            pass

    # CutOut Exception Scenario: Test no 1 parameters
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    with pytest.raises(TypeError, match="missing a required argument"):
        num_patches = 1
        transforms1 = [
            vision.Decode(to_pil=True),
            vision.ToTensor(),
            vision.CutOut(num_patches=num_patches)
        ]
        transform1 = t_trans.Compose(transforms1)
        ds2 = ds2.map(input_columns=["image"], operations=transform1)
        for _ in ds2.create_dict_iterator():
            pass

    # CutOut Exception Scenario: input is 2d
    image = np.random.randn(60, 50)
    to_pil = vision.ToPIL()
    image = to_pil(image)
    length = 200
    num_patches = 1
    cutout_op = vision.CutOut(length=length, num_patches=num_patches)
    with pytest.raises(RuntimeError, match="CutOut: shape is invalid."):
        cutout_op(image)

    # CutOut Exception Scenario: input is 4d
    image = np.random.randn(65, 50, 3, 3)
    length = 20
    cutout_op = vision.CutOut(length=length)
    with pytest.raises(RuntimeError, match=r"image shape is not <H,W,C>, but got rank: 4"):
        cutout_op(image)

    # CutOut Exception Scenario: input is bmp obj
    with Image.open(dir_data()[3]) as image:
        length = 50
        num_patches = 2
        cutout_op = vision.CutOut(length=length, num_patches=num_patches)
        _ = cutout_op(image)

    # CutOut Exception Scenario: input is list
    image = np.random.randn(65, 50, 3).tolist()
    length = 20
    cutout_op = vision.CutOut(length=length)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        cutout_op(image)

    # CutOut Exception Scenario: length is 20.5
    length = 20.5
    with pytest.raises(TypeError, match="Argument length with value 20.5 is not of type \\[<class 'int'>\\]."):
        vision.CutOut(length=length)

    # CutOut Exception Scenario: num_patches is 2.6
    length = 20
    num_patches = 2.6
    with pytest.raises(TypeError, match="Argument num_patches with value 2.6 is not of type \\[<class 'int'>\\]."):
        vision.CutOut(length=length, num_patches=num_patches)


if __name__ == "__main__":
    test_cut_out_op(plot=True)
    test_cut_out_op_multicut(plot=True)
    test_cut_out_md5()
    test_cut_out_comp_hwc(plot=True)
    test_cut_out_comp_chw(plot=True)
    test_cutout_4channel_chw()
    test_cutout_4channel_hwc()
    test_cut_out_validation()
    test_cutout_operation_01()
    test_cutout_operation_02()
    test_cutout_operation_03()
    test_cutout_exception_01()
    test_cutout_exception_02()
    test_cutout_exception_03()
    test_cutout_exception_04()
    test_cutout_exception_05()
