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
import mindspore.dataset.vision.transforms as vision

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
