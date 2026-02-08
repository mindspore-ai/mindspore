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
import mindspore.dataset.transforms
import mindspore.dataset.transforms.transforms as t_trans
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import visualize_list, visualize_one_channel_dataset, diff_mse, save_and_check_md5, save_and_check_md5_pil

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


def test_auto_contrast_py(plot=False):
    """
    Feature: AutoContrast op
    Description: Test AutoContrast Python implementation
    Expectation: The dataset is processed as expected
    """
    logger.info("Test AutoContrast Python implementation")

    # Original Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_original = mindspore.dataset.transforms.Compose([vision.Decode(True),
                                                                vision.Resize((224, 224)),
                                                                vision.ToTensor()])

    ds_original = data_set.map(
        operations=transforms_original, input_columns="image")

    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = np.transpose(image.asnumpy(), (0, 2, 3, 1))
        else:
            images_original = np.append(images_original,
                                        np.transpose(
                                            image.asnumpy(), (0, 2, 3, 1)),
                                        axis=0)

    # AutoContrast Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)

    transforms_auto_contrast = \
        mindspore.dataset.transforms.Compose([vision.Decode(True),
                                              vision.Resize((224, 224)),
                                              vision.AutoContrast(cutoff=10.0, ignore=[10, 20]),
                                              vision.ToTensor()])

    ds_auto_contrast = data_set.map(
        operations=transforms_auto_contrast, input_columns="image")

    ds_auto_contrast = ds_auto_contrast.batch(512)

    for idx, (image, _) in enumerate(ds_auto_contrast):
        if idx == 0:
            images_auto_contrast = np.transpose(image.asnumpy(), (0, 2, 3, 1))
        else:
            images_auto_contrast = np.append(images_auto_contrast,
                                             np.transpose(
                                                 image.asnumpy(), (0, 2, 3, 1)),
                                             axis=0)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_auto_contrast[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))

    # Compare with expected md5 from images
    filename = "autocontrast_01_result_py.npz"
    save_and_check_md5_pil(ds_auto_contrast, filename,
                           generate_golden=GENERATE_GOLDEN)

    if plot:
        visualize_list(images_original, images_auto_contrast)


def test_auto_contrast_c(plot=False):
    """
    Feature: AutoContrast op
    Description: Test AutoContrast Cpp implementation
    Expectation: The dataset is processed as expected
    """
    logger.info("Test AutoContrast C implementation")

    # AutoContrast Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data_set = data_set.map(operations=[vision.Decode(
    ), vision.Resize((224, 224))], input_columns=["image"])
    python_op = vision.AutoContrast(cutoff=10.0, ignore=[10, 20])
    c_op = vision.AutoContrast(cutoff=10.0, ignore=[10, 20])
    transforms_op = mindspore.dataset.transforms.Compose([lambda img: vision.ToPIL()(img.astype(np.uint8)),
                                                          python_op,
                                                          np.array])

    ds_auto_contrast_py = data_set.map(
        operations=transforms_op, input_columns="image")

    ds_auto_contrast_py = ds_auto_contrast_py.batch(512)

    for idx, (image, _) in enumerate(ds_auto_contrast_py):
        if idx == 0:
            images_auto_contrast_py = image.asnumpy()
        else:
            images_auto_contrast_py = np.append(images_auto_contrast_py,
                                                image.asnumpy(),
                                                axis=0)

    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data_set = data_set.map(operations=[vision.Decode(
    ), vision.Resize((224, 224))], input_columns=["image"])

    ds_auto_contrast_c = data_set.map(operations=c_op, input_columns="image")

    ds_auto_contrast_c = ds_auto_contrast_c.batch(512)

    for idx, (image, _) in enumerate(ds_auto_contrast_c):
        if idx == 0:
            images_auto_contrast_c = image.asnumpy()
        else:
            images_auto_contrast_c = np.append(images_auto_contrast_c,
                                               image.asnumpy(),
                                               axis=0)

    num_samples = images_auto_contrast_c.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_auto_contrast_c[i],
                          images_auto_contrast_py[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))
    np.testing.assert_equal(np.mean(mse), 0.0)

    # Compare with expected md5 from images
    filename = "autocontrast_01_result_c.npz"
    save_and_check_md5(ds_auto_contrast_c, filename,
                       generate_golden=GENERATE_GOLDEN)

    if plot:
        visualize_list(images_auto_contrast_c,
                       images_auto_contrast_py, visualize_mode=2)


def test_auto_contrast_one_channel_c(plot=False):
    """
    Feature: AutoContrast op
    Description: Test AutoContrast Cpp implementation with one channel images
    Expectation: The dataset is processed as expected
    """
    logger.info("Test AutoContrast C implementation With One Channel Images")

    # AutoContrast Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data_set = data_set.map(operations=[vision.Decode(
    ), vision.Resize((224, 224))], input_columns=["image"])
    python_op = vision.AutoContrast()
    c_op = vision.AutoContrast()
    # not using vision.ToTensor() since it converts to floats
    transforms_op = mindspore.dataset.transforms.Compose(
        [lambda img: (np.array(img)[:, :, 0]).astype(np.uint8),
         vision.ToPIL(),
         python_op,
         np.array])

    ds_auto_contrast_py = data_set.map(
        operations=transforms_op, input_columns="image")

    ds_auto_contrast_py = ds_auto_contrast_py.batch(512)

    for idx, (image, _) in enumerate(ds_auto_contrast_py):
        if idx == 0:
            images_auto_contrast_py = image.asnumpy()
        else:
            images_auto_contrast_py = np.append(images_auto_contrast_py,
                                                image.asnumpy(),
                                                axis=0)

    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    data_set = data_set.map(operations=[vision.Decode(), vision.Resize((224, 224)), lambda img: np.array(img[:, :, 0])],
                            input_columns=["image"])

    ds_auto_contrast_c = data_set.map(operations=c_op, input_columns="image")

    ds_auto_contrast_c = ds_auto_contrast_c.batch(512)

    for idx, (image, _) in enumerate(ds_auto_contrast_c):
        if idx == 0:
            images_auto_contrast_c = image.asnumpy()
        else:
            images_auto_contrast_c = np.append(images_auto_contrast_c,
                                               image.asnumpy(),
                                               axis=0)

    num_samples = images_auto_contrast_c.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(np.squeeze(images_auto_contrast_c[i]),
                          images_auto_contrast_py[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))
    np.testing.assert_equal(np.mean(mse), 0.0)

    if plot:
        visualize_list(images_auto_contrast_c,
                       images_auto_contrast_py, visualize_mode=2)


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


def test_auto_contrast_invalid_ignore_param_py():
    """
    Feature: AutoContrast op
    Description: Test AutoContrast Python implementation with invalid ignore parameter
    Expectation: Error is raised as expected
    """
    logger.info(
        "Test AutoContrast Python implementation with invalid ignore parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[mindspore.dataset.transforms.Compose([vision.Decode(True),
                                                                                  vision.Resize((224, 224)),
                                                                                  vision.AutoContrast(ignore=255.5),
                                                                                  vision.ToTensor()])],
                                input_columns=["image"])
    except TypeError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Argument ignore with value 255.5 is not of type" in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[mindspore.dataset.transforms.Compose([vision.Decode(True),
                                                                                  vision.Resize((224, 224)),
                                                                                  vision.AutoContrast(ignore=(10, 100)),
                                                                                  vision.ToTensor()])],
                                input_columns=["image"])
    except TypeError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Argument ignore with value (10,100) is not of type" in str(
            error)


def test_auto_contrast_invalid_cutoff_param_py():
    """
    Feature: AutoContrast op
    Description: Test AutoContrast Python implementation with invalid cutoff parameter
    Expectation: Error is raised as expected
    """
    logger.info(
        "Test AutoContrast Python implementation with invalid cutoff parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(operations=[mindspore.dataset.transforms.Compose([vision.Decode(True),
                                                                                  vision.Resize((224, 224)),
                                                                                  vision.AutoContrast(cutoff=-10.0),
                                                                                  vision.ToTensor()])],
                                input_columns=["image"])
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Input cutoff is not within the required interval of [0, 50)." in str(
            error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(
            operations=[mindspore.dataset.transforms.Compose([vision.Decode(True),
                                                              vision.Resize((224, 224)),
                                                              vision.AutoContrast(cutoff=120.0),
                                                              vision.ToTensor()])],
            input_columns=["image"])
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


def test_auto_contrast_operation_03():
    """
    Feature: AutoContrast operation
    Description: Testing the normal functionality of the AutoContrast operator
    Expectation: The Output is equal to the expected output
    """
    # AutoContrast Normal Function: ignore set to 10
    dataset = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = 10
    transforms = [
        vision.Decode(to_pil=True),
        vision.AutoContrast(cutoff=cutoff, ignore=ignore),
        vision.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    dataset = dataset.map(input_columns=["image"], operations=transform)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # AutoContrast Normal Function: ignore set to (10, 20)
    ds1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    transforms = [
        vision.Decode(to_pil=True),
        vision.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)

    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = (10, 20)
    transforms1 = [
        vision.Decode(to_pil=True),
        vision.AutoContrast(cutoff=cutoff, ignore=ignore),
        vision.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for _ in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        pass

    # AutoContrast Normal Function: ignore set to (100, 150, 200, 100, 255, 0)
    with Image.open(dir_data()[2]) as image:
        cutoff = 49
        ignore = (100, 150, 200, 100, 255, 0)
        auto_contrast_op = vision.AutoContrast(cutoff=cutoff, ignore=ignore)
        out = auto_contrast_op(image)

    # AutoContrast normal function: ignore.len is 100000
    with Image.open(dir_data()[2]) as image:
        cutoff = 38.653
        ignore = np.random.randint(0, 255, (100000,)).astype(np.uint8).tolist()
        auto_contrast_op = vision.AutoContrast(cutoff=cutoff, ignore=ignore)
        out = auto_contrast_op(image)

    # AutoContrast normal function: No parameters passed
    with Image.open(dir_data()[2]) as image:
        auto_contrast_op = vision.AutoContrast()
        out = auto_contrast_op(image)
        assert (np.array(image) == np.array(out)).all()


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


def test_auto_contrast_exception_02():
    """
    Feature: AutoContrast operation
    Description: Testing the AutoContrast Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # AutoContrast Anomaly Scenario: Multiple Parameters Transmitted
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 0.0
    ignore = 0
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        auto_contrast_op = [vision.Decode(to_pil=False), vision.AutoContrast(cutoff, ignore, more_para)]
        dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # AutoContrast Anomaly Scenario: cutoff set to -0.1, ignore range [11, 16]
    image = cv2.imread(dir_data()[1])
    cutoff = -0.1
    ignore = [11, 16]
    with pytest.raises(ValueError, match="Input cutoff is not within the required interval of"):
        auto_contrast_op = vision.AutoContrast(cutoff=cutoff, ignore=ignore)
        auto_contrast_op(image)

    # AutoContrast Anomaly Scenario: cutoff set to 100.01, ignore set to [16]
    image = cv2.imread(dir_data()[1])
    cutoff = 100.01
    ignore = [16]
    with pytest.raises(ValueError, match="Input cutoff is not within the required interval of"):
        auto_contrast_op = vision.AutoContrast(cutoff=cutoff, ignore=ignore)
        auto_contrast_op(image)

    # AutoContrast Anomaly Scenario: cutoff set to [10.0], ignore set to [16]
    image = cv2.imread(dir_data()[1])
    cutoff = [10.0]
    ignore = [16]
    with pytest.raises(TypeError, match="is not of type"):
        auto_contrast_op = vision.AutoContrast(cutoff=cutoff, ignore=ignore)
        auto_contrast_op(image)

    # AutoContrast Anomaly Scenario: cutoff set to (8.2,), ignore set to [16]
    image = cv2.imread(dir_data()[1])
    cutoff = (8.2,)
    ignore = [16]
    with pytest.raises(TypeError, match="is not of type"):
        auto_contrast_op = vision.AutoContrast(cutoff=cutoff, ignore=ignore)
        auto_contrast_op(image)

    # AutoContrast Anomaly Scenario: cutoff set to 8.2, ignore range [16.1, 8]
    image = cv2.imread(dir_data()[1])
    cutoff = 8.2
    ignore = [16.1, 8]
    with pytest.raises(TypeError, match="Argument item with value 16.1 is not of type"):
        auto_contrast_op = vision.AutoContrast(cutoff=cutoff, ignore=ignore)
        auto_contrast_op(image)

    # AutoContrast Anomaly Scenario: cutoff set to 8.2, ignore set to -1
    image = cv2.imread(dir_data()[1])
    cutoff = 8.2
    ignore = -1
    with pytest.raises(ValueError, match="Input ignore is not within the required interval of"):
        auto_contrast_op = vision.AutoContrast(cutoff=cutoff, ignore=ignore)
        auto_contrast_op(image)

    # AutoContrast Anomaly Scenarios: cutoff set to 20, ignore values set to (8, 255, 254, 0)
    image = cv2.imread(dir_data()[1])
    image = np.array(image).astype(np.float64)
    cutoff = 20
    ignore = (8, 255, 254, 0)
    auto_contrast_op = vision.AutoContrast(cutoff=cutoff, ignore=ignore)
    with pytest.raises(RuntimeError):
        auto_contrast_op(image)

    # AutoContrast Anomaly Scenario: ignore set to 256
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    with pytest.raises(ValueError, match="Input ignore is not within the required interval"):
        ignore = 256
        transforms1 = [
            vision.Decode(to_pil=True),
            vision.AutoContrast(ignore=ignore),
            vision.ToTensor()
        ]
        transform1 = t_trans.Compose(transforms1)
        ds2 = ds2.map(input_columns=["image"], operations=transform1)
        for _ in ds2:
            pass


def test_auto_contrast_exception_03():
    """
    Feature: AutoContrast operation
    Description: Testing the AutoContrast Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # AutoContrast Anomaly Scenario: Multiple Parameters Passed
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = [10, 20]
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        transforms1 = [
            vision.Decode(to_pil=True),
            vision.AutoContrast(cutoff, ignore, more_para),
            vision.ToTensor()
        ]
        transform1 = t_trans.Compose(transforms1)
        ds2 = ds2.map(input_columns=["image"], operations=transform1)

        for _ in ds2.create_dict_iterator(output_numpy=True):
            pass

    # AutoContrast Anomaly Scenario: Input data is a PNG image
    with Image.open(dir_data()[4]) as image:
        auto_contrast_op = vision.AutoContrast()
        with pytest.raises(OSError, match="not supported for .*"):
            auto_contrast_op(image)

    # AutoContrast Anomaly Scenario: Input data is a list
    image = np.array(Image.open(dir_data()[4])).tolist()
    auto_contrast_op = vision.AutoContrast()
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        auto_contrast_op(image)

    # AutoContrast Anomaly Scenario: cutoff set to -1
    cutoff = -1
    with pytest.raises(ValueError, match="Input cutoff is not within the required interval of \\[0, 50\\)."):
        vision.AutoContrast(cutoff=cutoff)

    # AutoContrast Anomaly Scenario: cutoff set to -100.1
    cutoff = 100.1
    with pytest.raises(ValueError, match="Input cutoff is not within the required interval of \\[0, 50\\)."):
        vision.AutoContrast(cutoff=cutoff)

    # AutoContrast Anomaly Scenario: cutoff set to [20]
    cutoff = [20]
    with pytest.raises(TypeError,
                       match="Argument cutoff with value \\[20\\] is not of type \\[<class 'int'>, <class 'float'>\\]"):
        vision.AutoContrast(cutoff=cutoff)

    # AutoContrast Anomaly Scenario: cutoff set to True
    cutoff = True
    with pytest.raises(TypeError,
                       match="Argument cutoff with value True is not of type \\(<class 'int'>, <class 'float'>\\)"):
        vision.AutoContrast(cutoff=cutoff)

    # AutoContrast Anomaly Scenario: ignore set to {5, 6}
    cutoff = 10
    ignore = {5, 6}
    with pytest.raises(TypeError, match="Argument ignore with value {5, 6} is not of type \\[<class"
                                        " 'list'>, <class 'tuple'>, <class 'int'>\\]."):
        vision.AutoContrast(cutoff=cutoff, ignore=ignore)

    # AutoContrast Anomaly Scenario: ignore set to -1
    cutoff = 10
    ignore = -1
    with pytest.raises(ValueError, match="Input ignore is not within the required interval of \\[0, 255\\]."):
        vision.AutoContrast(cutoff=cutoff, ignore=ignore)

    # AutoContrast Anomaly Scenario: ignore set to 20.6
    cutoff = 10
    ignore = 20.6
    with pytest.raises(TypeError, match="Argument ignore with value 20.6 is not of "
                                        "type \\[<class 'list'>, <class 'tuple'>, <class 'int'>\\]."):
        vision.AutoContrast(cutoff=cutoff, ignore=ignore)

    # AutoContrast Anomaly Scenario: ignore set to True
    cutoff = 10
    ignore = True
    with pytest.raises(TypeError, match="Argument ignore with value True is not of "
                                        "type \\(<class 'list'>, <class 'tuple'>, <class 'int'>\\)."):
        vision.AutoContrast(cutoff=cutoff, ignore=ignore)



if __name__ == "__main__":
    test_auto_contrast_py(plot=True)
    test_auto_contrast_c(plot=True)
    test_auto_contrast_one_channel_c(plot=True)
    test_auto_contrast_mnist_c(plot=True)
    test_auto_contrast_invalid_ignore_param_c()
    test_auto_contrast_invalid_ignore_param_py()
    test_auto_contrast_invalid_cutoff_param_c()
    test_auto_contrast_invalid_cutoff_param_py()
    test_auto_contrast_operation_01()
    test_auto_contrast_operation_02()
    test_auto_contrast_operation_03()
    test_auto_contrast_exception_01()
    test_auto_contrast_exception_02()
    test_auto_contrast_exception_03()
