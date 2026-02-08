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
Testing CenterCrop op in DE
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
from mindspore import Tensor
from util import diff_mse, visualize_list, save_and_check_md5

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


def dir_data():
    """Obtain the dataset"""
    data_list = []
    data_dir1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    data_dir2 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_1.jpg")
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


def test_center_crop_op(height=375, width=375, plot=False):
    """
    Feature: CenterCrop op
    Description: Test CenterCrop op basic usage
    Expectation: Runs successfully
    """
    logger.info("Test CenterCrop")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"])
    decode_op = vision.Decode()
    # 3 images [375, 500] [600, 500] [512, 512]
    center_crop_op = vision.CenterCrop([height, width])
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=center_crop_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"])
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    image_cropped = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_cropped.append(item1["image"].copy())
        image.append(item2["image"].copy())
    if plot:
        visualize_list(image, image_cropped)


def test_center_crop_md5(height=375, width=375):
    """
    Feature: CenterCrop op
    Description: Test CenterCrop using md5 check test
    Expectation: Passes the md5 check test
    """
    logger.info("Test CenterCrop")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    decode_op = vision.Decode()
    # 3 images [375, 500] [600, 500] [512, 512]
    center_crop_op = vision.CenterCrop([height, width])
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=center_crop_op, input_columns=["image"])
    # Compare with expected md5 from images
    filename = "center_crop_01_result.npz"
    save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_center_crop_comp(height=375, width=375, plot=False):
    """
    Feature: CenterCrop op
    Description: Test CenterCrop between Python and Cpp image augmentation
    Expectation: Resulting outputs from both operations are expected to be equal
    """
    logger.info("Test CenterCrop")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    decode_op = vision.Decode()
    center_crop_op = vision.CenterCrop([height, width])
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=center_crop_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.CenterCrop([height, width]),
        vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    data2 = data2.map(operations=transform, input_columns=["image"])

    image_c_cropped = []
    image_py_cropped = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        c_image = item1["image"]
        py_image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        # Note: The images aren't exactly the same due to rounding error
        assert diff_mse(py_image, c_image) < 0.001
        image_c_cropped.append(c_image.copy())
        image_py_cropped.append(py_image.copy())
    if plot:
        visualize_list(image_c_cropped, image_py_cropped, visualize_mode=2)


def test_crop_grayscale(height=375, width=375):
    """
    Feature: CenterCrop op
    Description: Test CenterCrop works with pad and grayscale images
    Expectation: Runs successfully
    """

    # Note: image.transpose performs channel swap to allow py transforms to
    # work with c transforms
    transforms = [
        vision.Decode(True),
        vision.Grayscale(1),
        vision.ToTensor(),
        (lambda image: (image.transpose(1, 2, 0) * 255).astype(np.uint8))
    ]

    transform = mindspore.dataset.transforms.Compose(transforms)
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    data1 = data1.map(operations=transform, input_columns=["image"])

    # If input is grayscale, the output dimensions should be single channel
    crop_gray = vision.CenterCrop([height, width])
    data1 = data1.map(operations=crop_gray, input_columns=["image"])

    for item1 in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        c_image = item1["image"]

        # Check that the image is grayscale
        assert (c_image.ndim == 3 and c_image.shape[2] == 1)


def test_center_crop_errors():
    """
    Feature: CenterCrop op
    Description: Test CenterCrop with bad inputs
    Expectation: Error is raised as expected
    """
    try:
        test_center_crop_op(16777216, 16777216)
    except RuntimeError as e:
        assert "Padding size cannot be more than 3 times of the original image size" in \
               str(e)


def test_center_crop_high_dimensions():
    """
    Feature: CenterCrop
    Description: Use randomly generated tensors and batched dataset as video inputs
    Expectation: Cropped images should in correct shape
    """
    logger.info("Test CenterCrop using video inputs.")
    # use randomly generated tensor for testing
    video_frames = np.random.randint(
        0, 255, size=(32, 64, 64, 3), dtype=np.uint8)
    center_crop_op = vision.CenterCrop([32, 32])
    video_frames = center_crop_op(video_frames)
    assert video_frames.shape[1] == 32
    assert video_frames.shape[2] == 32

    # use a batch of real image for testing
    # First dataset
    height = 200
    width = 200
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    center_crop_op = vision.CenterCrop([height, width])
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1_batch = data1.batch(batch_size=2)

    for item in data1_batch.create_dict_iterator(num_epochs=1, output_numpy=True):
        original_channel = item["image"].shape[-1]

    data1_batch = data1_batch.map(
        operations=center_crop_op, input_columns=["image"])

    for item in data1_batch.create_dict_iterator(num_epochs=1, output_numpy=True):
        shape = item["image"].shape
        assert shape[-3] == height
        assert shape[-2] == width
        assert shape[-1] == original_channel


def test_center_crop_operation_01():
    """
    Feature: CenterCrop operation
    Description: Testing the normal functionality of the CenterCrop operator
    Expectation: The Output is equal to the expected output
    """
    # CenterCrop normal function: size set to 1
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    size = 1
    center_crop_op = vision.CenterCrop(size=size)
    dataset2 = dataset2.map(input_columns=["image"], operations=center_crop_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # CenterCrop normal function: size set to (100, 200)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    size = (100, 200)
    center_crop_op = vision.CenterCrop(size=size)
    dataset2 = dataset2.map(input_columns=["image"], operations=center_crop_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # CenterCrop normal function: size set to 100
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    size = 100
    center_crop_op = vision.CenterCrop(size=size)
    dataset2 = dataset2.map(input_columns=["image"], operations=center_crop_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # CenterCrop normal function: size set to (50, 50)
    with Image.open(dir_data()[1]) as image:
        center_crop_op = vision.CenterCrop(size=(50, 50))
        _ = center_crop_op(image)

    # CenterCrop normal function: size set to 150
    image = cv2.imread(dir_data()[1])
    center_crop_op = vision.CenterCrop(size=150)
    _ = center_crop_op(image)

    # CenterCrop normal function: size set to [800, 1500]
    image = cv2.imread(dir_data()[1])
    center_crop_op = vision.CenterCrop(size=[800, 1500])
    _ = center_crop_op(image)

    # CenterCrop normal function: size set to 1
    image = cv2.imread(dir_data()[1])
    center_crop_op = vision.CenterCrop(size=1)
    _ = center_crop_op(image)

    # CenterCrop normal function: size set to (2700, 2100)
    image = cv2.imread(dir_data()[1])
    center_crop_op = vision.CenterCrop(size=(2700, 2100))
    _ = center_crop_op(image)

    # CenterCrop normal function: size set to (60, 60)
    image = np.random.randn(128, 128, 3)
    center_crop_op = vision.CenterCrop(size=(60, 60))
    _ = center_crop_op(image)

    # CenterCrop normal functionality: Input format is (H, W, C)
    image = np.random.randint(0, 255, (32, 32, 1)).astype(np.uint8)
    center_crop_op = vision.CenterCrop(size=(96, 96))
    out = center_crop_op(image)
    assert out.shape == (96, 96, 1)

    # CenterCrop normal functionality: Input format is (H, W)
    image = np.random.randint(-255, 255, (1024, 1024)).astype(np.uint8)
    center_crop_op = vision.CenterCrop(size=(2048, 102))
    out = center_crop_op(image)
    assert out.shape == (2048, 102)


def test_center_crop_operation_02():
    """
    Feature: CenterCrop operation
    Description: Testing the normal functionality of the CenterCrop operator
    Expectation: The Output is equal to the expected output
    """
    # CenterCrop Normal Functionality: Testing Pipeline Mode
    dataset = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    size = 100
    transforms1 = [
        vision.Decode(to_pil=True),
        vision.CenterCrop(size=size),
        vision.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # CenterCrop normal function: size set to 1
    dataset = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    size = 1
    transforms1 = [
        vision.Decode(to_pil=True),
        vision.CenterCrop(size=size),
        vision.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # CenterCrop normal function: size set to (100, 200)
    dataset = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    size = (100, 200)
    transforms1 = [
        vision.Decode(to_pil=True),
        vision.CenterCrop(size=size),
        vision.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # CenterCrop normal function: size set to [100, 200]
    dataset = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    size = [100, 200]
    transforms = [
        vision.Decode(to_pil=True),
        vision.CenterCrop(size=size),
        vision.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    dataset = dataset.map(input_columns=["image"], operations=transform)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # CenterCrop normal functionality: Input is a JPG image
    with Image.open(dir_data()[2]) as image:
        size = 100
        center_crop_op = vision.CenterCrop(size=size)
        _ = center_crop_op(image)

    # CenterCrop normal functionality: Input is a BMP image
    with Image.open(dir_data()[3]) as image:
        size = [96, 120]
        center_crop_op = vision.CenterCrop(size=size)
        out = center_crop_op(image)
        assert (np.array(image) == out).all()

    # CenterCrop normal functionality: Input is a PNG image
    with Image.open(dir_data()[4]) as image:
        size = (384, 384)
        center_crop_op = vision.CenterCrop(size=size)
        _ = center_crop_op(image)

    # CenterCrop normal functionality: Input is a GIF image
    with Image.open(dir_data()[5]) as image:
        size = 1
        center_crop_op = vision.CenterCrop(size=size)
        _ = center_crop_op(image)

    # CenterCrop normal functionality: Input is a GIF image, size set to 1000
    with Image.open(dir_data()[2]) as image:
        size = 1000
        center_crop_op = vision.CenterCrop(size=size)
        _ = center_crop_op(image)


def test_center_crop_exception_01():
    """
    Feature: CenterCrop operation
    Description: Testing the CenterCrop Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # CenterCrop Exception Scenario: size is 0
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    size = 0
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        center_crop_op = vision.CenterCrop(size=size)
        dataset2 = dataset2.map(input_columns=["image"], operations=center_crop_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # CenterCrop Exception Scenario: size is 16777216
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    size = 16777216
    with pytest.raises(RuntimeError,
                       match="Exception thrown from dataset pipeline. Refer to 'Dataset Pipeline Error Message'."):
        center_crop_op = vision.CenterCrop(size=size)
        dataset2 = dataset2.map(input_columns=["image"], operations=center_crop_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # center_crop算子：Test  size 为 100.0
    # CenterCrop Exception Scenario: size is 100.0
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    size = 100.0
    with pytest.raises(TypeError, match="Argument size"):
        center_crop_op = vision.CenterCrop(size=size)
        dataset2 = dataset2.map(input_columns=["image"], operations=center_crop_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # CenterCrop Exception Scenario: size is (100.0, 200.0)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    size = (100.0, 200.0)
    with pytest.raises(TypeError, match="Argument size\\[0\\] with value 100.0 is not of type \\[<class 'int'>\\]."):
        center_crop_op = vision.CenterCrop(size=size)
        dataset2 = dataset2.map(input_columns=["image"], operations=center_crop_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # CenterCrop Exception Scenario: size is (100, 200, 300)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    size = (100, 200, 300)
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        center_crop_op = vision.CenterCrop(size=size)
        dataset2 = dataset2.map(input_columns=["image"], operations=center_crop_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # CenterCrop Exception Scenario: size is ""
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    size = ""
    with pytest.raises(TypeError, match="Argument size"):
        center_crop_op = vision.CenterCrop(size=size)
        dataset2 = dataset2.map(input_columns=["image"], operations=center_crop_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # CenterCrop Exception Scenario: No Parameters Passed
    dataset2 = ds.ImageFolderDataset(dir_data()[0], 1, shuffle=False, decode=True)
    with pytest.raises(TypeError, match="missing a required argument"):
        center_crop_op = vision.CenterCrop()
        dataset2 = dataset2.map(input_columns=["image"], operations=center_crop_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # CenterCrop Exception Scenario: Input shape is (100, 100, 4, 3)
    image = np.random.randint(-255, 255, (100, 100, 4, 3)).astype(np.uint8)
    center_crop_op = vision.CenterCrop(size=(200, 200))
    with pytest.raises(RuntimeError):
        center_crop_op(image)


def test_center_crop_exception_02():
    """
    Feature: CenterCrop operation
    Description: Testing the CenterCrop Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # CenterCrop Exception Scenario: Input shape is (100,)
    image = np.random.randint(-255, 255, (100,)).astype(np.uint8)
    center_crop_op = vision.CenterCrop(size=200)
    with pytest.raises(RuntimeError):
        center_crop_op(image)

    # CenterCrop Exception Scenario: size is (0, 50)
    image = cv2.imread(dir_data()[1])
    with pytest.raises(ValueError, match="Input is not within the required interval of"):
        center_crop_op = vision.CenterCrop(size=(0, 50))
        center_crop_op(image)

    # CenterCrop Exception Scenario: size is 16777216
    image = cv2.imread(dir_data()[1])
    with pytest.raises(RuntimeError,
                       match="Exception thrown from dataset pipeline. Refer to 'Dataset Pipeline Error Message'."):
        center_crop_op = vision.CenterCrop(size=16777216)
        center_crop_op(image)

    # CenterCrop Exception Scenario: size is 50.5
    with pytest.raises(TypeError, match="Argument size with value 50.5 is not of type"):
        vision.CenterCrop(size=50.5)

    # CenterCrop Exception Scenario: size is [50]
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        vision.CenterCrop(size=[50])

    # CenterCrop Exception Scenario: size is (50, 50, 50)
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        vision.CenterCrop(size=(50, 50, 50))

    # CenterCrop Exception Scenario: size is np.array([50, 50]
    with pytest.raises(TypeError, match="is not of type"):
        vision.CenterCrop(size=np.array([50, 50]))

    # CenterCrop Exception Scenario: size is Tensor
    with pytest.raises(TypeError, match="is not of type"):
        vision.CenterCrop(size=Tensor([50, 50]))

    # CenterCrop Exception Scenario: size is 0
    size = 0
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.CenterCrop(size=size)

    # CenterCrop Exception Scenario: size is 16777216
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    with pytest.raises(RuntimeError,
                       match="DecompressionBombError: Image size.* could be decompression bomb DOS attack."):
        size = 16777216
        transforms1 = [
            vision.Decode(to_pil=True),
            vision.CenterCrop(size=size),
            vision.ToTensor()
        ]
        transform1 = t_trans.Compose(transforms1)
        ds2 = ds2.map(input_columns=["image"], operations=transform1)
        for _ in ds2:
            pass

    # CenterCrop Exception Scenario: size is 100.0
    size = 100.0
    with pytest.raises(TypeError, match="Argument size with value 100.0 is not of type "
                                        "\\[<class 'int'>, <class 'list'>, <class 'tuple'>\\]."):
        vision.CenterCrop(size=size)

    # CenterCrop Exception Scenario: size is (100, 200, 300, 400)
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        size = (100, 200, 300, 400)
        transforms1 = [
            vision.Decode(to_pil=True),
            vision.CenterCrop(size=size),
            vision.ToTensor()
        ]
        transform1 = t_trans.Compose(transforms1)
        ds2 = ds2.map(input_columns=["image"], operations=transform1)
        for _ in ds2:
            pass

    # CenterCrop Exception Scenario: size is (100, 200, 300)
    size = (100, 200, 300)
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple \\(h, w\\) of length 2."):
        vision.CenterCrop(size=size)

    # CenterCrop Exception Scenario: size is ""
    size = ""
    with pytest.raises(TypeError, match="Argument size with value"):
        vision.CenterCrop(size=size)

    # CenterCrop Exception Scenario: No Parameters Passed
    with pytest.raises(TypeError, match="missing a required argument"):
        vision.CenterCrop()


def test_center_crop_exception_03():
    """
    Feature: CenterCrop operation
    Description: Testing the CenterCrop Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # CenterCrop Exception Scenario: Input data shape is 0
    image = 10
    size = 100
    center_crop_op = vision.CenterCrop(size=size)
    with pytest.raises(TypeError, match=r"Input should be NumPy or PIL image, got <class 'int'>."):
        center_crop_op(image)

    # CenterCrop Exception Scenario: Input data is a list
    image = np.array(Image.open(dir_data()[2])).tolist()
    size = 100
    center_crop_op = vision.CenterCrop(size=size)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        center_crop_op(image)

    # CenterCrop Exception Scenario: Input data is a tuple
    image = tuple(np.array(Image.open(dir_data()[2])))
    size = 100
    center_crop_op = vision.CenterCrop(size=size)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>."):
        center_crop_op(image)

    # center_crop算子：size为[100]。
    # CenterCrop Exception Scenario: size is [100]
    size = [100]
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple \\(h, w\\) of length 2."):
        vision.CenterCrop(size=size)

    # CenterCrop Exception Scenario: size is numpy
    size = np.array([10, 20])
    with pytest.raises(TypeError, match="Argument size with value \\[10 20\\] is not of "
                                        "type \\[<class 'int'>, <class 'list'>, <class 'tuple'>\\]."):
        vision.CenterCrop(size=size)

    # CenterCrop Exception Scenario: size is True
    size = True
    with pytest.raises(TypeError, match="Argument size with value True is not of type"
                                        " \\(<class 'int'>, <class 'list'>, <class 'tuple'>\\)."):
        vision.CenterCrop(size=size)

    # CenterCrop Exception Scenario: size is {10, 20}
    size = {10, 20}
    with pytest.raises(TypeError, match="Argument size with value {10, 20} is not of "
                                        "type \\[<class 'int'>, <class 'list'>, <class 'tuple'>\\]."):
        vision.CenterCrop(size=size)

    # CenterCrop Exception Scenario: size is [[10, 20]]
    size = [[10, 20]]
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple \\(h, w\\) of length 2."):
        vision.CenterCrop(size=size)

    # CenterCrop Exception Scenario: size is 16777217
    size = 16777217
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[1, 16777216\\]."):
        vision.CenterCrop(size=size)


if __name__ == "__main__":
    test_center_crop_op(600, 600, plot=True)
    test_center_crop_op(300, 600)
    test_center_crop_op(600, 300)
    test_center_crop_md5()
    test_center_crop_comp(plot=True)
    test_crop_grayscale()
    test_center_crop_high_dimensions()
    test_center_crop_operation_01()
    test_center_crop_operation_02()
    test_center_crop_exception_01()
    test_center_crop_exception_02()
    test_center_crop_exception_03()
