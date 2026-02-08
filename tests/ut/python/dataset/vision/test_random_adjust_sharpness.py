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
Testing RandomAdjustSharpness in DE
"""
import cv2
import numpy as np
import os
import pytest
from PIL import Image

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import helper_random_op_pipeline, visualize_list, visualize_image, diff_mse

image_file = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
data_dir = "../data/dataset/testImageNetData/train/"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


def test_random_adjust_sharpness_pipeline(plot=False):
    """
    Feature: RandomAdjustSharpness op
    Description: Test RandomAdjustSharpness pipeline
    Expectation: Passes the test
    """
    logger.info("Test RandomAdjustSharpness pipeline")

    # Original Images
    images_original = helper_random_op_pipeline(data_dir)

    # Randomly Adjust Sharpness Images
    images_random_adjust_sharpness = helper_random_op_pipeline(
        data_dir, vision.RandomAdjustSharpness(2.0, 0.6))

    if plot:
        visualize_list(images_original, images_random_adjust_sharpness)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_adjust_sharpness[i],
                          images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_random_adjust_sharpness_eager():
    """
    Feature: RandomAdjustSharpness op
    Description: Test RandomAdjustSharpness eager
    Expectation: Passes the equality test
    """
    img = np.fromfile(image_file, dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = vision.Decode()(img)
    img_sharped = vision.RandomSharpness((2.0, 2.0))(img)
    img_random_sharped = vision.RandomAdjustSharpness(2.0, 1.0)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(
        type(img_random_sharped), img_random_sharped.shape))

    assert img_random_sharped.all() == img_sharped.all()


def test_random_adjust_sharpness_comp(plot=False):
    """
    Feature: RandomAdjustSharpness op
    Description: Test RandomAdjustSharpness op compared with Sharpness op
    Expectation: Resulting outputs from both operations are expected to be equal
    """
    random_adjust_sharpness_op = vision.RandomAdjustSharpness(
        degree=2.0, prob=1.0)
    sharpness_op = vision.RandomSharpness((2.0, 2.0))

    dataset1 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    for item in dataset1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item['image']
    dataset1.map(operations=random_adjust_sharpness_op,
                 input_columns=['image'])
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset.map(operations=sharpness_op, input_columns=['image'])

    for item1, item2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_random_sharpness = item1['image']
        image_sharpness = item2['image']

    mse = diff_mse(image_sharpness, image_random_sharpness)
    assert mse == 0
    logger.info("mse: {}".format(mse))
    if plot:
        visualize_image(image, image_random_sharpness, mse, image_sharpness)


def test_random_adjust_sharpness_invalid_prob():
    """
    Feature: RandomAdjustSharpness op
    Description: Test invalid prob where prob is out of range
    Expectation: Error is raised as expected
    """
    logger.info("test_random_adjust_sharpness_invalid_prob")
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        random_adjust_sharpness_op = vision.RandomAdjustSharpness(2.0, 1.5)
        dataset = dataset.map(
            operations=random_adjust_sharpness_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(
            e)


def test_random_adjust_sharpness_invalid_degree():
    """
    Feature: RandomAdjustSharpness op
    Description: Test invalid prob where prob is out of range
    Expectation: Error is raised as expected
    """
    logger.info("test_random_adjust_sharpness_invalid_prob")
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        random_adjust_sharpness_op = vision.RandomAdjustSharpness(-1.0, 1.5)
        dataset = dataset.map(
            operations=random_adjust_sharpness_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "interval" in str(e)


def test_random_adjust_sharpness_four_dim():
    """
    Feature: RandomAdjustSharpness
    Description: test with four dimension images
    Expectation: raise errors as expected
    """
    logger.info("test_random_adjust_sharpness_four_dim")

    c_op = vision.RandomAdjustSharpness(2.0, 0.5)

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[vision.Decode(), vision.Resize((224, 224)),
                                            lambda img: np.array(img[2, 200, 10, 32])], input_columns=["image"])

        data_set = data_set.map(operations=c_op, input_columns="image")

    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "image shape is not <H,W,C>" in str(e)


def test_random_adjust_sharpness_invalid_input():
    """
    Feature: RandomAdjustSharpness
    Description: test with images in uint32 type
    Expectation: raise errors as expected
    """
    logger.info("test_random_adjust_sharpness_invalid_input")

    c_op = vision.RandomAdjustSharpness(2.0, 0.5)

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[vision.Decode(), vision.Resize((224, 224)),
                                            lambda img: np.array(img[2, 32, 3], dtype=uint32)], input_columns=["image"])
        data_set = data_set.map(operations=c_op, input_columns="image")

    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Cannot convert from OpenCV type, unknown CV type" in str(e)


def test_random_adjust_sharpness_operation_01():
    """
    Feature: RandomAdjustSharpness operation
    Description: Testing the normal functionality of the RandomAdjustSharpness operator
    Expectation: The Output is equal to the expected output
    """
    # RandomAdjustSharpness operator, normal testing, pipeline mode, numpy image, degree=0.0
    data_dir_1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir_1, shuffle=False, decode=True)
    random_sharpness_op = vision.RandomAdjustSharpness(degree=0.0, prob=1)
    dataset = dataset.map(input_columns=["image"], operations=random_sharpness_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # RandomAdjustSharpness operator, normal testing, pipeline mode, PIL image, degree=1.0
    data_dir_1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir_1, shuffle=False, decode=True)
    random_sharpness_op = [vision.ToPIL(), vision.RandomAdjustSharpness(degree=1.0, prob=1)]
    dataset = dataset.map(input_columns=["image"], operations=random_sharpness_op)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # RandomAdjustSharpness operator, normal test, degree=49.9
    data_dir_1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir_1, shuffle=False, decode=True)
    random_sharpness_op = vision.RandomAdjustSharpness(degree=49.9)
    dataset = dataset.map(input_columns=["image"], operations=random_sharpness_op)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # RandomAdjustSharpness operator, normal testing, eager mode, degree and prob include two decimal places
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        random_sharpness_op = vision.RandomAdjustSharpness(degree=1.55, prob=0.55)
        _ = random_sharpness_op(image)

    # RandomAdjustSharpness operator, normal testing, degree and prob contain 7 decimal places
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        random_sharpness_op = vision.RandomAdjustSharpness(degree=49.9999999, prob=0.5555555)
        _ = random_sharpness_op(image)

    # RandomAdjustSharpness operator, normal test, degree=1, prob=1
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        random_sharpness_op = vision.RandomAdjustSharpness(degree=1, prob=1)
        out = random_sharpness_op(image)
        assert (out == image).all()

    # RandomAdjustSharpness operator, normal test, prob=0.0
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        random_sharpness_op = vision.RandomAdjustSharpness(degree=0.5, prob=0.0)
        out = random_sharpness_op(image)
        assert (out == image).all()

    # RandomAdjustSharpness operator, normal test, prob=1.0
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        random_sharpness_op = vision.RandomAdjustSharpness(degree=0.5, prob=1.0)
        _ = random_sharpness_op(image)

    # RandomAdjustSharpness operator, standard testing with input data being an image read via cv2
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image = cv2.imread(image_jpg)
    random_sharpness_op = vision.RandomAdjustSharpness(degree=0.5, prob=0.5)
    _ = random_sharpness_op(image)

    # RandomAdjustSharpness operator, normal testing, input data is a single-channel numpy array
    image = np.random.randint(0, 255, (128, 128, 1)).astype(np.uint8)
    random_sharpness_op = vision.RandomAdjustSharpness(degree=0.5, prob=1)
    _ = random_sharpness_op(image)

    # RandomAdjustSharpness operator, standard testing with input data being a two-channel numpy array
    image = np.random.randint(0, 255, (128, 128, 2)).astype(np.uint8)
    random_sharpness_op = vision.RandomAdjustSharpness(degree=1, prob=1)
    out = random_sharpness_op(image)
    assert (out == image).all()

    # RandomAdjustSharpness operator, normal testing, input data is a numpy four-channel array
    image = np.random.randint(0, 255, (128, 128, 4)).astype(np.uint8)
    random_sharpness_op = vision.RandomAdjustSharpness(degree=100, prob=1)
    _ = random_sharpness_op(image)


def test_random_adjust_sharpness_operation_02():
    """
    Feature: RandomAdjustSharpness operation
    Description: Testing the normal functionality of the RandomAdjustSharpness operator
    Expectation: The Output is equal to the expected output
    """
    # RandomAdjustSharpness operator, normal testing, input data is a Git PIL image
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_gif) as image:
        random_sharpness_op = vision.RandomAdjustSharpness(degree=0.5)
        _ = random_sharpness_op(image)

    # RandomAdjustSharpness operator, standard testing with input data being a numpy two-dimensional array
    image = np.random.randint(-255, 255, (256, 128)).astype(np.uint8)
    random_sharpness_op = vision.RandomAdjustSharpness(degree=200, prob=1)
    _ = random_sharpness_op(image)


def test_random_adjust_sharpness_exception_01():
    """
    Feature: RandomAdjustSharpness operation
    Description: Testing the RandomAdjustSharpness Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # RandomAdjustSharpness Operator, Abnormal Test, degree=16777216.1
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        with pytest.raises(ValueError, match="Input degree is not within the required interval of \\[0, 16777216\\]"):
            random_sharpness_op = vision.RandomAdjustSharpness(degree=16777216.1)
            random_sharpness_op(image)

    # RandomAdjustSharpness Operator, Abnormal Test, degree=-0.1
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        with pytest.raises(ValueError, match="Input degree is not within the required interval of \\[0, 16777216\\]"):
            random_sharpness_op = vision.RandomAdjustSharpness(degree=-0.1)
            random_sharpness_op(image)

    # RandomAdjustSharpness Operator, Abnormal Test, degree is str
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        with pytest.raises(TypeError, match="Argument degree with value test is not of type \\[<class 'float'>, " + \
                                            "<class 'int'>\\], but got <class 'str'>"):
            random_sharpness_op = vision.RandomAdjustSharpness(degree='test')
            random_sharpness_op(image)

    # RandomAdjustSharpness Operator, Abnormal Test, degree is tuple
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        with pytest.raises(TypeError,
                           match="Argument degree with value \\(0.1, 0.5\\) is not of type \\[<class 'float'>," + \
                                 " <class 'int'>\\], but got <class 'tuple'>"):
            random_sharpness_op = vision.RandomAdjustSharpness(degree=(0.1, 0.5))
            random_sharpness_op(image)

    # RandomAdjustSharpness Operator, Abnormal Test, degree is list
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        with pytest.raises(TypeError,
                           match="Argument degree with value \\[0.1, 0.5\\] is not of type " + \
                                 "\\[<class 'float'>, <class 'int'>\\], but got <class 'list'>"):
            random_sharpness_op = vision.RandomAdjustSharpness(degree=[0.1, 0.5])
            random_sharpness_op(image)

    # RandomAdjustSharpness Operator, Abnormal Test, degree is set
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        with pytest.raises(TypeError, match="Argument degree with value {0.1, 0.5} is not of "
                                            "type \\[<class 'float'>, <class 'int'>\\], but got <class 'set'>"):
            random_sharpness_op = vision.RandomAdjustSharpness(degree={0.1, 0.5})
            random_sharpness_op(image)

    # RandomAdjustSharpness Operator, Abnormal Test, prob=-0.1
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        with pytest.raises(ValueError, match="Input prob is not within the required interval of \\[0.0, 1.0\\]."):
            random_sharpness_op = vision.RandomAdjustSharpness(degree=0.5, prob=-0.1)
            _ = random_sharpness_op(image)

    # RandomAdjustSharpness Operator, Abnormal Test, prob=1.1
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        with pytest.raises(ValueError, match="Input prob is not within the required interval of \\[0.0, 1.0\\]."):
            random_sharpness_op = vision.RandomAdjustSharpness(degree=0.5, prob=1.1)
            _ = random_sharpness_op(image)

    # RandomAdjustSharpness Operator, Abnormal Test, prob is str
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        with pytest.raises(TypeError,
                           match="Argument prob with value test is not of type \\[<class 'float'>, <class 'int'>\\]" + \
                                 ", but got <class 'str'>."):
            random_sharpness_op = vision.RandomAdjustSharpness(degree=0.5, prob='test')
            _ = random_sharpness_op(image)

    # RandomAdjustSharpness Operator, Abnormal Test, prob is list
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        with pytest.raises(TypeError,
                           match="Argument prob with value \\[0.5, 0.8\\] is not of type " + \
                                 "\\[<class 'float'>, <class 'int'>\\], but got <class 'list'>."):
            random_sharpness_op = vision.RandomAdjustSharpness(degree=0.5, prob=[0.5, 0.8])
            _ = random_sharpness_op(image)

    # RandomAdjustSharpness Operator, Abnormal Test, Parameter test redundancy
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        with pytest.raises(TypeError, match="got an unexpected keyword argument 'test'"):
            random_sharpness_op = vision.RandomAdjustSharpness(degree=0.5, prob=0.5, test=0.5)
            _ = random_sharpness_op(image)

    # RandomAdjustSharpness Operator, Abnormal Test, input data is list
    image = np.random.randn(128, 128, 4).astype(np.uint8).tolist()
    random_sharpness_op = vision.RandomAdjustSharpness(degree=0.5)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        random_sharpness_op(image)


def test_random_adjust_sharpness_exception_02():
    """
    Feature: RandomAdjustSharpness operation
    Description: Testing the RandomAdjustSharpness Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # RandomAdjustSharpness Operator, Abnormal Test, Input data is one-dimensional NumPy data.
    image = np.random.randn(200,).astype(np.uint8)
    random_sharpness_op = vision.RandomAdjustSharpness(degree=0.5)
    with pytest.raises(RuntimeError, match="RandomAdjustSharpness: image shape is "
                                           "not <H,W,C> or <H,W>, got rank: 1"):
        random_sharpness_op(image)

    # RandomAdjustSharpness Operator, Abnormal Test, Input data is four-dimensional NumPy data.
    image = np.random.randint(0, 255, (8, 100, 200, 3)).astype(np.uint8)
    random_sharpness_op = vision.RandomAdjustSharpness(degree=0.5)
    with pytest.raises(RuntimeError, match="RandomAdjustSharpness: image shape is "
                                           "not <H,W,C> or <H,W>, got rank: 4"):
        random_sharpness_op(image)

    # RandomAdjustSharpness Operator, Abnormal Test, Input data is NumPy data
    image = np.random.randn(128, 128, 3).astype("U")
    random_sharpness_op = vision.RandomAdjustSharpness(degree=0.5)
    with pytest.raises(RuntimeError, match=r"RandomAdjustSharpness: Cannot convert "
                                           r"from OpenCV type, unknown CV type. Currently supported data "
                                           r"type: \[int8, uint8, int16, uint16, int32, float16, float32, float64\]."):
        random_sharpness_op(image)

    # RandomAdjustSharpness Operator, Abnormal Test, input data is Tensor
    image = ms.Tensor(np.random.randn(5, 5, 3))
    random_sharpness_op = vision.RandomAdjustSharpness(degree=0.5)
    with pytest.raises(TypeError,
                       match="Input should be NumPy or PIL image, got <class 'mindspore.common.tensor.Tensor'>."):
        random_sharpness_op(image)

    # RandomAdjustSharpness Operator, Abnormal Test, degree=None
    with pytest.raises(TypeError, match="Argument degree with value None is not of type \\[<class 'float'>, " + \
                                        "<class 'int'>\\], but got <class 'NoneType'>"):
        vision.RandomAdjustSharpness(degree=None)

    # RandomAdjustSharpness Operator, Abnormal Test, prob=None
    with pytest.raises(TypeError, match="Argument prob with value None is not of type \\[<class 'float'>," + \
                                        " <class 'int'>\\], but got <class 'NoneType'>"):
        vision.RandomAdjustSharpness(degree=0.5, prob=None)

    # RandomAdjustSharpness Operator, Abnormal Test, input data is int
    image = 10
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'int'>"):
        random_sharpness_op = vision.RandomAdjustSharpness(degree=0.5)
        random_sharpness_op(image)

    # RandomAdjustSharpness Operator, Abnormal Test, input data is tuple
    image = (10,)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>"):
        random_sharpness_op = vision.RandomAdjustSharpness(degree=0.5)
        random_sharpness_op(image)

    # RandomAdjustSharpness Operator, Abnormal Test, No parameters passed
    with pytest.raises(TypeError, match="missing a required argument: 'degree'"):
        vision.RandomAdjustSharpness()


if __name__ == "__main__":
    test_random_adjust_sharpness_pipeline(plot=True)
    test_random_adjust_sharpness_eager()
    test_random_adjust_sharpness_comp(plot=True)
    test_random_adjust_sharpness_invalid_prob()
    test_random_adjust_sharpness_invalid_degree()
    test_random_adjust_sharpness_four_dim()
    test_random_adjust_sharpness_invalid_input()
    test_random_adjust_sharpness_operation_01()
    test_random_adjust_sharpness_operation_02()
    test_random_adjust_sharpness_exception_01()
    test_random_adjust_sharpness_exception_02()
