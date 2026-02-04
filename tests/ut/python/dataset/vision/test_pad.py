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
Testing Pad op in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms as trans
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from mindspore.dataset.vision import Border as v_Border
from util import diff_mse, save_and_check_md5, save_and_check_md5_pil

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

GENERATE_GOLDEN = False


def test_pad_op():
    """
    Feature: Pad op
    Description: Test Pad op between Python and Cpp implementation
    Expectation: Both outputs are the same as expected
    """
    logger.info("test_random_color_jitter_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()

    pad_op = vision.Pad((100, 100, 100, 100))
    ctrans = [decode_op,
              pad_op,
              ]

    data1 = data1.map(operations=ctrans, input_columns=["image"])

    # Second dataset
    transforms = [
        vision.Decode(True),
        vision.Pad(100),
        vision.ToTensor(),
    ]
    transform = trans.Compose(transforms)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform, input_columns=["image"])

    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        c_image = item1["image"]
        py_image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)

        logger.info("shape of c_image: {}".format(c_image.shape))
        logger.info("shape of py_image: {}".format(py_image.shape))

        logger.info("dtype of c_image: {}".format(c_image.dtype))
        logger.info("dtype of py_image: {}".format(py_image.dtype))

        mse = diff_mse(c_image, py_image)
        logger.info("mse is {}".format(mse))
        assert mse < 0.01


def test_pad_op2():
    """
    Feature: Pad op
    Description: Test Pad op parameter with size 2
    Expectation: Output's shape is the same as expected output's shape
    """
    logger.info("test padding parameter with size 2")

    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    resize_op = vision.Resize([90, 90])
    pad_op = vision.Pad((100, 9,))
    ctrans = [decode_op, resize_op, pad_op]

    data1 = data1.map(operations=ctrans, input_columns=["image"])
    for data in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(data["image"].shape)
        # It pads left, right with 100 and top, bottom with 9,
        # so the final size of image row is 90 + 9 + 9 = 108
        # so the final size of image col is 90 + 100 + 100 = 290
        assert data["image"].shape[0] == 108
        assert data["image"].shape[1] == 290


def test_pad_grayscale():
    """
    Feature: Pad op
    Description: Test Pad op for grayscale images
    Expectation: Output's shape is the same as expected output
    """

    # Note: image.transpose performs channel swap to allow py transforms to
    # work with c transforms
    transforms = [
        vision.Decode(True),
        vision.Grayscale(1),
        vision.ToTensor(),
        (lambda image: (image.transpose(1, 2, 0) * 255).astype(np.uint8))
    ]

    transform = trans.Compose(transforms)
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform, input_columns=["image"])

    # if input is grayscale, the output dimensions should be single channel
    pad_gray = vision.Pad(100, fill_value=(20, 20, 20))
    data1 = data1.map(operations=pad_gray, input_columns=["image"])
    dataset_shape_1 = []
    for item1 in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        c_image = item1["image"]
        dataset_shape_1.append(c_image.shape)

    # Dataset for comparison
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()

    # we use the same padding logic
    ctrans = [decode_op, pad_gray]
    dataset_shape_2 = []

    data2 = data2.map(operations=ctrans, input_columns=["image"])

    for item2 in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
        c_image = item2["image"]
        dataset_shape_2.append(c_image.shape)

    for shape1, shape2 in zip(dataset_shape_1, dataset_shape_2):
        # validate that the first two dimensions are the same
        # we have a little inconsistency here because the third dimension is 1 after vision.Grayscale
        assert shape1[0:1] == shape2[0:1]


def test_pad_md5():
    """
    Feature: Pad op
    Description: Test Pad op with md5 check
    Expectation: Passes the md5 check test
    """
    logger.info("test_pad_md5")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    pad_op = vision.Pad(150)
    ctrans = [decode_op,
              pad_op,
              ]

    data1 = data1.map(operations=ctrans, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    pytrans = [
        vision.Decode(True),
        vision.Pad(150),
        vision.ToTensor(),
    ]
    transform = trans.Compose(pytrans)
    data2 = data2.map(operations=transform, input_columns=["image"])
    # Compare with expected md5 from images
    filename1 = "pad_01_c_result.npz"
    save_and_check_md5(data1, filename1, generate_golden=GENERATE_GOLDEN)
    filename2 = "pad_01_py_result.npz"
    save_and_check_md5_pil(data2, filename2, generate_golden=GENERATE_GOLDEN)


def test_pad_operation_01():
    """
    Feature: Pad operation
    Description: Testing the normal functionality of the Pad operator
    Expectation: The Output is equal to the expected output
    """
    # Pad operator: Normal testing, padding = 100, fill_value = 2, padding_mode = v_Border.CONSTANT
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = 100
    fill_value = 2
    padding_mode = v_Border.CONSTANT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(to_pil=True),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(to_pil=True),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)
    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 200 == py_image.shape[0]
        assert c_image.shape[1] + 200 == py_image.shape[1]
        break

    # Pad operator: Normal testing, padding = 1000, fill_value = 2, padding_mode = v_Border.SYMMETRIC
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = 1000
    fill_value = 2
    padding_mode = v_Border.SYMMETRIC
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 2000 == py_image.shape[0]
        assert c_image.shape[1] + 2000 == py_image.shape[1]
        break

    # Pad operator: Normal testing, padding = 0, fill_value = 2, padding_mode = v_Border.REFLECT
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = 0
    fill_value = 2
    padding_mode = v_Border.REFLECT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] == py_image.shape[0]
        assert c_image.shape[1] == py_image.shape[1]
        break


def test_pad_operation_02():
    """
    Feature: Pad operation
    Description: Testing the normal functionality of the Pad operator
    Expectation: The Output is equal to the expected output
    """
    # Pad operator: Normal testing, padding = (50, 0), fill_value = 2, padding_mode = v_Border.REFLECT
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = (50, 0)
    fill_value = 2
    padding_mode = v_Border.REFLECT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] == py_image.shape[0]
        assert c_image.shape[1] + 50 * 2 == py_image.shape[1]
        break

    # Pad operator: Normal testing, padding = (0, 10), fill_value = 2, padding_mode = v_Border.CONSTANT
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = (0, 10)
    fill_value = 2
    padding_mode = v_Border.CONSTANT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 10 * 2 == py_image.shape[0]
        assert c_image.shape[1] == py_image.shape[1]
        break

    # Pad operator: Normal testing, padding = [0, 10], fill_value = 2, padding_mode = v_Border.EDGE
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = [0, 10]
    fill_value = 2
    padding_mode = v_Border.EDGE
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 10 * 2 == py_image.shape[0]
        assert c_image.shape[1] == py_image.shape[1]
        break


def test_pad_operation_03():
    """
    Feature: Pad operation
    Description: Testing the normal functionality of the Pad operator
    Expectation: The Output is equal to the expected output
    """
    # Pad operator: Normal testing, padding = (0, 10, 20, 30), fill_value = 2, padding_mode = v_Border.CONSTANT
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = (0, 10, 20, 30)
    fill_value = 2
    padding_mode = v_Border.CONSTANT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 40 == py_image.shape[0]
        assert c_image.shape[1] + 20 == py_image.shape[1]
        break

    # Pad operator: Normal testing, padding = [50, 10, 20, 30], fill_value = 2, padding_mode = v_Border.EDGE
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = [50, 10, 20, 30]
    fill_value = 2
    padding_mode = v_Border.EDGE
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 40 == py_image.shape[0]
        assert c_image.shape[1] + 70 == py_image.shape[1]
        break

    # Pad operator: Normal testing, padding = 10, fill_value = 100, padding_mode = v_Border.CONSTANT
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = 10
    fill_value = 100
    padding_mode = v_Border.CONSTANT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 20 == py_image.shape[0]
        assert c_image.shape[1] + 20 == py_image.shape[1]
        assert (py_image[-1][-1][:] == (100, 100, 100)).all()
        break


def test_pad_operation_04():
    """
    Feature: Pad operation
    Description: Testing the normal functionality of the Pad operator
    Expectation: The Output is equal to the expected output
    """
    # Pad operator: Normal testing, fill_value=0
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = 10
    fill_value = 0
    padding_mode = v_Border.CONSTANT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 20 == py_image.shape[0]
        assert c_image.shape[1] + 20 == py_image.shape[1]
        assert (py_image[-1][-1][:] == (0, 0, 0)).all()
        break

    # Pad operator: Normal testing, fill_value=(5,10,20)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = 10
    fill_value = (5, 10, 20)
    padding_mode = v_Border.CONSTANT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 20 == py_image.shape[0]
        assert c_image.shape[1] + 20 == py_image.shape[1]
        assert (py_image[-1][-1][:] == (5, 10, 20)).all()
        break

    # Pad operator: Normal testing, fill_value=(0,100,0)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = 10
    fill_value = (0, 100, 0)
    padding_mode = v_Border.CONSTANT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 20 == py_image.shape[0]
        assert c_image.shape[1] + 20 == py_image.shape[1]
        assert (py_image[-1][-1][:] == (0, 100, 0)).all()
        break


def test_pad_operation_05():
    """
    Feature: Pad operation
    Description: Testing the normal functionality of the Pad operator
    Expectation: The Output is equal to the expected output
    """
    # Pad operator: Normal testing, fill_value=(0,0,200)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = 10
    fill_value = (0, 0, 200)
    padding_mode = v_Border.CONSTANT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 20 == py_image.shape[0]
        assert c_image.shape[1] + 20 == py_image.shape[1]
        assert (py_image[-1][-1][:] == (0, 0, 200)).all()
        break

    # Pad operator: Normal testing, eager mode, input is a JPG image.
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        padding = 100
        fill_value = 100
        padding_mode = v_Border.CONSTANT
        pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)
        _ = pad_op(image)

    # Pad operator: Normal testing, eager mode, input is a GIF image.
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_gif) as image:
        padding = [100, 200]
        fill_value = 0
        padding_mode = v_Border.EDGE
        pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)
        _ = pad_op(image)

    # Pad operator: Normal testing, eager mode, input is a BMP image.
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    with Image.open(image_bmp) as image:
        padding = (100, 200, 300, 400)
        fill_value = (120, 255, 120)
        padding_mode = v_Border.SYMMETRIC
        pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)
        _ = pad_op(image)

    # Pad operator: Normal testing, eager mode, input is a PNG image.
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with Image.open(image_png) as image:
        padding = [185, 100, 250, 440]
        fill_value = 250
        padding_mode = v_Border.REFLECT
        pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)
        out = pad_op(image)
        assert np.array(out).shape[0] == np.array(image).shape[0] + padding[1] + padding[3]
        assert np.array(out).shape[1] == np.array(image).shape[1] + padding[0] + padding[2]

    # Pad operator: Normal testing, eager mode, padding = (1000, 2000)
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        padding = (1000, 2000)
        fill_value = 250
        padding_mode = v_Border.EDGE
        pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)
        out = pad_op(image)
        assert np.array(out).shape[0] == np.array(image).shape[0] + 4000
        assert np.array(out).shape[1] == np.array(image).shape[1] + 2000

    # Pad operator: Normal testing, eager mode, padding = (0, 1)
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    with Image.open(image_bmp) as image:
        padding = (0, 1)
        fill_value = 250
        padding_mode = v_Border.REFLECT
        pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)
        out = pad_op(image)
        assert np.array(out).shape[0] == np.array(image).shape[0] + 2
        assert np.array(out).shape[1] == np.array(image).shape[1]


def test_pad_operation_06():
    """
    Feature: Pad operation
    Description: Testing the normal functionality of the Pad operator
    Expectation: The Output is equal to the expected output
    """
    # Pad operator: Normal testing, eager mode, padding = [100, 200]
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with Image.open(image_png) as image:
        padding = [100, 200]
        fill_value = 250
        padding_mode = v_Border.SYMMETRIC
        pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)
        out = pad_op(image)
        assert np.array(out).shape[0] == np.array(image).shape[0] + 400
        assert np.array(out).shape[1] == np.array(image).shape[1] + 200

    # Pad operator: Normal testing, eager mode, padding = 863
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with Image.open(image_png) as image:
        padding = 863
        fill_value = 250
        padding_mode = v_Border.SYMMETRIC
        pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)
        out = pad_op(image)
        assert np.array(out).shape[0] == np.array(image).shape[0] + 863 * 2
        assert np.array(out).shape[1] == np.array(image).shape[1] + 863 * 2

    # Pad operator: Normal testing, eager mode, padding = (100, 128, 150, 100)
    image = np.random.randint(0, 255, (128, 256, 3)).astype(np.uint8)
    topil_op = vision.ToPIL()
    image2 = topil_op(image)
    padding = (100, 128, 150, 100)
    fill_value = 250
    padding_mode = v_Border.EDGE
    pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)
    out = pad_op(image2)
    assert np.array(out).shape[0] == np.array(image).shape[0] + padding[1] + padding[3]
    assert np.array(out).shape[1] == np.array(image).shape[1] + padding[0] + padding[2]


def test_pad_exception_01():
    """
    Feature: Pad operation
    Description: Testing the Pad Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Pad operator: Exception testing, fill_value=1000
    with pytest.raises(ValueError, match=r"Input fill_value is not within the required interval of \[0, 255\]."):
        vision.Pad(padding=10, fill_value=1000)

    # Pad operator: Exception testing, eager mode, 4-Channel Image
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with Image.open(image_png) as image:
        image = np.array(image)
        padding = 100
        fill_value = 25
        pad_op = vision.Pad(padding=padding, fill_value=fill_value)
        with pytest.raises(RuntimeError, match=r"Pad: the channel of image tensor does not match "
                                               r"the requirement of operator. Expecting tensor in channel of \(1, 3\)"
                                               r". But got channel 4."):
            pad_op(image)

    # Pad operator: Exception testing, eager mode, input data is list
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with Image.open(image_png) as image:
        image = np.array(image).tolist()
        pad_op = vision.Pad(padding=10)
        with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
            pad_op(image)

    # Pad operator: Exception testing, length of padding is 1
    padding = [100]
    with pytest.raises(ValueError, match="The size of the padding list or tuple should be 2 or 4."):
        vision.Pad(padding=padding)

    # Pad operator: Exception testing, padding is of numpy type
    padding = np.array([10, 20])
    with pytest.raises(TypeError, match="Argument padding with value \\[10 20\\] is not of "
                                        "type \\[<class 'tuple'>, <class 'list'>, <class 'numbers.Number'>\\]."):
        vision.Pad(padding=padding)

    # Pad operator: Exception testing, length of padding is 5
    padding = (10, 20, 30, 40, 50)
    with pytest.raises(ValueError, match="The size of the padding list or tuple should be 2 or 4."):
        vision.Pad(padding=padding)

    # Pad operator: Exception testing, The element value in the padding is -1.
    padding = (-1, 20)
    with pytest.raises(ValueError,
                       match="Input pad_value is not within the required interval of \\[0, 2147483647\\]."):
        vision.Pad(padding=padding)

    # Pad operator: Exception testing, The element value in the padding is 2147483648.
    padding = [10, 10, 20, 2147483648]
    with pytest.raises(ValueError,
                       match="Input pad_value is not within the required interval of \\[0, 2147483647\\]."):
        vision.Pad(padding=padding)

    # Pad operator: Exception testing, padding is of float type
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        padding = 100.0
        with pytest.raises(TypeError, match="Argument padding with value 100.0 is not of type "
                                            "\\[<class 'int'>\\], but got <class 'float'>."):
            vision.Pad(padding=padding)(image)

    # Pad operator: Exception testing, Do not pass the padding parameter
    with pytest.raises(TypeError, match="missing a required argument: 'padding'"):
        vision.Pad()

    # Pad operator: Exception testing, The length of fill_value is 1
    padding = 100
    fill_value = (250,)
    padding_mode = v_Border.SYMMETRIC
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)

    # Pad operator: Exception testing, The length of fill_value is 4
    padding = 100
    fill_value = (250, 120, 130, 2)
    padding_mode = v_Border.CONSTANT
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)

    # Pad operator: Exception testing, The element value in the fill_value is 256
    padding = 100
    fill_value = (256, 120, 130)
    with pytest.raises(ValueError, match=r"Input fill_value\[0\] is not within the required interval of \[0, 255\]."):
        vision.Pad(padding=padding, fill_value=fill_value)

    # Pad operator: Exception testing, The element value in the fill_value is -1
    padding = 100
    fill_value = (25, -1, 130)
    padding_mode = v_Border.REFLECT
    with pytest.raises(ValueError, match=r"Input fill_value\[1\] is not within the required interval of \[0, 255\]."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)

    # Pad operator: Exception testing, fill_value is of float type
    padding = 100
    fill_value = (25, 10.0, 130)
    padding_mode = v_Border.EDGE
    with pytest.raises(TypeError, match=r"Argument fill_value\[1\] with value 10.0 is not of type \[<class 'int'>\],"
                                        r" but got <class 'float'>."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)


def test_pad_exception_02():
    """
    Feature: Pad operation
    Description: Testing the Pad Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Pad operator: Exception testing, fill_value is of list type
    padding = 100
    fill_value = [25, 10, 130]
    padding_mode = v_Border.SYMMETRIC
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)

    # Pad operator: Exception testing, fill_value is of numpy type
    padding = 100
    fill_value = np.array([25, 10, 130])
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.Pad(padding=padding, fill_value=fill_value)

    # Pad operator: Exception testing, padding_mode is of str type
    padding = 100
    fill_value = (25, 10, 130)
    padding_mode = "v_Border.EDGE"
    with pytest.raises(TypeError,
                       match="Argument padding_mode with value v_Border.EDGE is not of type \\[<enum 'Border'>\\]."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)

    # Pad operator: Exception testing, padding_mode is of Border type
    padding = 100
    fill_value = (25, 10, 130)
    padding_mode = v_Border
    with pytest.raises(TypeError, match="Argument padding_mode with value <enum 'Border'> is "
                                        "not of type \\[<enum 'Border'>\\]."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)

    # Pad operator: Exception testing, padding_mode is of int type
    padding = 100
    fill_value = (25, 10, 130)
    padding_mode = 1
    with pytest.raises(TypeError, match="Argument padding_mode with value 1 is not of type \\[<enum 'Border'>\\]."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)

    # Pad operator: Exception testing, padding_mode is of list type
    padding = 100
    fill_value = (25, 10, 130)
    padding_mode = [v_Border.EDGE]
    with pytest.raises(TypeError, match="Argument padding_mode with value \\[<Border.EDGE: 'edge'>\\] is "
                                        "not of type \\[<enum 'Border'>\\]."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)


if __name__ == "__main__":
    test_pad_op()
    test_pad_grayscale()
    test_pad_md5()
    test_pad_operation_01()
    test_pad_operation_02()
    test_pad_operation_03()
    test_pad_operation_04()
    test_pad_operation_05()
    test_pad_operation_06()
    test_pad_exception_01()
    test_pad_exception_02()
