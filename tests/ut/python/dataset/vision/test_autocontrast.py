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
Testing AutoContrast op in DE
"""
import cv2
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import visualize_one_channel_dataset, save_and_check_md5

DATA_DIR = "../data/dataset/testImageNetData/train/"
MNIST_DATA_DIR = "../data/dataset/testMnistData"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

GENERATE_GOLDEN = False


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








def test_auto_contrast_mnist_c(plot=False):
    """
    Feature: AutoContrast op
    Description: Test AutoContrast Cpp implementation with MnistDataset (grayscale images)
    Expectation: The dataset is processed as expected
    """
    logger.info("Test AutoContrast C implementation With MNIST Images")
    data_set = ds.MnistDataset(
        dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)
    ds_auto_contrast_c = data_set.map(operations=vision.AutoContrast(
        cutoff=1, ignore=(0, 255)), input_columns="image")
    ds_orig = ds.MnistDataset(
        dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)

    images = []
    images_trans = []
    labels = []
    for _, (data_orig, data_trans) in enumerate(zip(ds_orig, ds_auto_contrast_c)):
        image_orig, label_orig = data_orig
        image_trans, _ = data_trans
        images.append(image_orig.asnumpy())
        labels.append(label_orig.asnumpy())
        images_trans.append(image_trans.asnumpy())

    # Compare with expected md5 from images
    filename = "autocontrast_mnist_result_c.npz"
    save_and_check_md5(ds_auto_contrast_c, filename,
                       generate_golden=GENERATE_GOLDEN)

    if plot:
        visualize_one_channel_dataset(images, images_trans, labels)


def test_auto_contrast_invalid_ignore_param_c():
    """
    Feature: AutoContrast op
    Description: Test AutoContrast Cpp implementation with invalid ignore parameter
    Expectation: Error is raised as expected
    """
    logger.info(
        "Test AutoContrast C implementation with invalid ignore parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[vision.Decode(),
                                            vision.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])
        # invalid ignore
        data_set = data_set.map(operations=vision.AutoContrast(
            ignore=255.5), input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Argument ignore with value 255.5 is not of type" in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[vision.Decode(), vision.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])
        # invalid ignore
        data_set = data_set.map(operations=vision.AutoContrast(
            ignore=(10, 100)), input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Argument ignore with value (10,100) is not of type" in str(
            error)


def test_auto_contrast_invalid_cutoff_param_c():
    """
    Feature: AutoContrast op
    Description: Test AutoContrast Cpp implementation with invalid cutoff parameter
    Expectation: Error is raised as expected
    """
    logger.info(
        "Test AutoContrast C implementation with invalid cutoff parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[vision.Decode(),
                                            vision.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])
        # invalid ignore
        data_set = data_set.map(operations=vision.AutoContrast(
            cutoff=-10.0), input_columns="image")
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Input cutoff is not within the required interval of [0, 50)." in str(
            error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[vision.Decode(),
                                            vision.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])
        # invalid ignore
        data_set = data_set.map(operations=vision.AutoContrast(
            cutoff=120.0), input_columns="image")
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Input cutoff is not within the required interval of [0, 50)." in str(
            error)






def test_auto_contrast_operation_01():
    """
    Feature: AutoContrast operation
    Description: Testing the normal functionality of the AutoContrast operator
    Expectation: The Output is equal to the expected output
    """
    # AutoContrast Normal Functions: Test Pipeline Mode
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = [10, 20]
    auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast(cutoff=cutoff, ignore=ignore)]
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # AutoContrast Normal Function: cutoff set to 0.0
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 0.0
    auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast(cutoff=cutoff)]
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # AutoContrast Normal Function: cutoff set to 49.99
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 49.99
    auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast(cutoff=cutoff)]
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # AutoContrast Normal Function: cutoff set to 10
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10
    auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast(cutoff=cutoff)]
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # AutoContrast Normal Function: ignore set to 10
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = 10
    auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast(cutoff=cutoff, ignore=ignore)]
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # AutoContrast Normal Function: ignore set to 0
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = 0
    auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast(cutoff=cutoff, ignore=ignore)]
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # AutoContrast Normal Function: ignore set to 255
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = 255
    auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast(cutoff=cutoff, ignore=ignore)]
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass


def test_auto_contrast_operation_02():
    """
    Feature: AutoContrast operation
    Description: Testing the normal functionality of the AutoContrast operator
    Expectation: The Output is equal to the expected output
    """
    # AutoContrast Normal Function: ignore set to (10, 20)
    dataset = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = (10, 20)
    auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast(cutoff=cutoff, ignore=ignore)]
    dataset = dataset.map(input_columns=["image"], operations=auto_contrast_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # AutoContrast Normal Function: ignore set to [10, 20]
    dataset = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = [10, 20]
    auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast(cutoff=cutoff, ignore=ignore)]
    dataset = dataset.map(input_columns=["image"], operations=auto_contrast_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # AutoContrast Normal Function: No parameters passed
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast()]
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # AutoContrast Normal Function: cutoff = 0.8, ignore = [0, 20, 15]
    with Image.open(dir_data()[1]) as image:
        cutoff = 8.8
        ignore = [0, 20, 15]
        auto_contrast_op = vision.AutoContrast(cutoff=cutoff, ignore=ignore)
        _ = auto_contrast_op(image)

    # AutoContrast normal function: cutoff set to 0.01, ignore set to None
    image = cv2.imread(dir_data()[1])
    cutoff = 0.01
    ignore = None
    auto_contrast_op = vision.AutoContrast(cutoff=cutoff, ignore=ignore)
    _ = auto_contrast_op(image)

    # AutoContrast Normal Function: cutoff set to 0, ignore set to 6
    image = cv2.imread(dir_data()[1])
    cutoff = 0
    ignore = 6
    auto_contrast_op = vision.AutoContrast(cutoff=cutoff, ignore=ignore)
    _ = auto_contrast_op(image)

    # AutoContrast Normal Function: cutoff set to 49.9, ignore set to (8, 255, 254, 0)
    image = cv2.imread(dir_data()[1])
    cutoff = 49.9
    ignore = (8, 255, 254, 0)
    auto_contrast_op = vision.AutoContrast(cutoff=cutoff, ignore=ignore)
    _ = auto_contrast_op(image)

    # AutoContrast normal function: ignore values are [11, 16]
    image = cv2.imread(dir_data()[1])
    ignore = [11, 16]
    auto_contrast_op = vision.AutoContrast(ignore=ignore)
    _ = auto_contrast_op(image)




def test_auto_contrast_exception_01():
    """
    Feature: AutoContrast operation
    Description: Testing the AutoContrast Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # AutoContrast Anomaly Scenario: cutoff set to 100.1
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 100.1
    with pytest.raises(ValueError, match="Input cutoff is not within the required interval"):
        auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast(cutoff=cutoff)]
        dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # AutoContrast Anomaly Scenario: cutoff set to -0.1
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = -0.1
    with pytest.raises(ValueError, match="Input cutoff is not within the required interval"):
        auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast(cutoff=cutoff)]
        dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # AutoContrast Anomaly Scenario: cutoff set to [10.0]
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = [10.0]
    with pytest.raises(TypeError, match="Argument cutoff"):
        auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast(cutoff=cutoff)]
        dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # AutoContrast Anomaly Scenario: cutoff set to ""
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = ""
    with pytest.raises(TypeError, match="Argument cutoff"):
        auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast(cutoff=cutoff)]
        dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # AutoContrast Anomaly Scenario: ignore set to 256
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = 256
    with pytest.raises(ValueError, match="Input ignore is not within the required interval"):
        auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast(cutoff=cutoff, ignore=ignore)]
        dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # AutoContrast Anomaly Scenario: ignore set to -1
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = -1
    with pytest.raises(ValueError, match="Input ignore is not within the required interval"):
        auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast(cutoff=cutoff, ignore=ignore)]
        dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # AutoContrast Anomaly Scenario: ignore set to ""
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 1.5
    ignore = ""
    with pytest.raises(TypeError, match="Argument ignore"):
        auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast(cutoff=cutoff, ignore=ignore)]
        dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass
