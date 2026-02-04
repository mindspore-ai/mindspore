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
Testing GaussianBlur Python API
"""
import cv2
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms as t_trans
import mindspore.dataset.vision.transforms as vision

from mindspore import log as logger
from util import visualize_image, diff_mse

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
IMAGE_FILE = "../data/dataset/apple.jpg"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


def test_gaussian_blur_pipeline(plot=False):
    """
    Feature: GaussianBlur
    Description: Test GaussianBlur of Cpp implementation
    Expectation: Output is the same as expected output
    """
    logger.info("test_gaussian_blur_pipeline")

    # First dataset
    dataset1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = vision.Decode()
    gaussian_blur_op = vision.GaussianBlur(3, 3)
    dataset1 = dataset1.map(operations=decode_op, input_columns=["image"])
    dataset1 = dataset1.map(operations=gaussian_blur_op, input_columns=["image"])

    # Second dataset
    dataset2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    dataset2 = dataset2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        gaussian_blur_ms = data1["image"]
        original = data2["image"]
        gaussian_blur_cv = cv2.GaussianBlur(original, (3, 3), 3)
        mse = diff_mse(gaussian_blur_ms, gaussian_blur_cv)
        logger.info("gaussian_blur_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1
        if plot:
            visualize_image(original, gaussian_blur_ms, mse, gaussian_blur_cv)


def test_gaussian_blur_eager():
    """
    Feature: GaussianBlur
    Description: Test GaussianBlur in eager mode
    Expectation: Output is the same as expected output
    """
    logger.info("test_gaussian_blur_eager")
    img = cv2.imread(IMAGE_FILE)

    img_ms = vision.GaussianBlur((3, 5), (3.5, 3.5))(img)
    img_cv = cv2.GaussianBlur(img, (3, 5), 3.5, 3.5)
    mse = diff_mse(img_ms, img_cv)
    assert mse == 0


def test_gaussian_blur_exception():
    """
    Feature: GaussianBlur
    Description: Test GaussianBlur with invalid parameters
    Expectation: Error is raised as expected
    """
    logger.info("test_gaussian_blur_exception")
    try:
        _ = vision.GaussianBlur([2, 2])
    except ValueError as e:
        logger.info("Got an exception in GaussianBlur: {}".format(str(e)))
        assert "not an odd value" in str(e)
    try:
        _ = vision.GaussianBlur(3.0, [3, 3])
    except TypeError as e:
        logger.info("Got an exception in GaussianBlur: {}".format(str(e)))
        assert "not of type [<class 'int'>, <class 'list'>, <class 'tuple'>]" in str(e)
    try:
        _ = vision.GaussianBlur(3, -3)
    except ValueError as e:
        logger.info("Got an exception in GaussianBlur: {}".format(str(e)))
        assert "not within the required interval" in str(e)
    try:
        _ = vision.GaussianBlur(3, [3, 3, 3])
    except TypeError as e:
        logger.info("Got an exception in GaussianBlur: {}".format(str(e)))
        assert "should be a single number or a list/tuple of length 2" in str(e)


def test_gaussian_blur_operation_01():
    """
    Feature: GaussianBlur operation
    Description: Testing the normal functionality of the GaussianBlur operator
    Expectation: The Output is equal to the expected output
    """
    # GaussianBlur operator: Test sigma is 102.6
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train", "")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    horizontal_flip_op = vision.GaussianBlur([11, 127], 102.6)
    dataset2 = dataset2.map(input_columns=["image"], operations=horizontal_flip_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # GaussianBlur operator: Test sigma is 2
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with Image.open(image_png) as image:
        gaussianblur_op = vision.GaussianBlur(9, 2)
        _ = gaussianblur_op(image)

    # GaussianBlur operator: Test sigma is 120
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        gaussianblur_op = vision.GaussianBlur((101, 1007), 120)
        _ = gaussianblur_op(image)

    # GaussianBlur operator: Test sigma is [1.2, 3.5]
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_gif) as image:
        gaussianblur_op = vision.GaussianBlur((3, 7), [1.2, 3.5])
        _ = gaussianblur_op(image)

    # GaussianBlur operator: Test sigma is [16.0, 0.1]
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    with Image.open(image_bmp) as image:
        gaussianblur_op = vision.GaussianBlur([165, 1613], [16.0, 0.1])
        _ = gaussianblur_op(image)

    # GaussianBlur operator: Test sigma is 0
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_gif) as image:
        gaussianblur_op = vision.GaussianBlur(1, 0)
        _ = gaussianblur_op(image)

    # GaussianBlur operator: Test sigma is none
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with Image.open(image_png) as image:
        gaussianblur_op = vision.GaussianBlur(11)
        out = gaussianblur_op(image)
        gaussianblur_op = vision.GaussianBlur(11, 2.0)
        out2 = gaussianblur_op(image)
        assert (out == out2).all()

    # GaussianBlur operator: Test input.shape is (50, 50, 8)
    image = np.random.randint(0, 255, (50, 50, 8)).astype(np.uint8)
    gaussianblur_op = vision.GaussianBlur(1, 100.0)
    _ = gaussianblur_op(image)

    # GaussianBlur operator: Test input is 3d list
    image = list(np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8))
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>"):
        gaussianblur_op = vision.GaussianBlur(1, 0)
        _ = gaussianblur_op(image)

    # GaussianBlur operator: Test input is 3d tuple
    image = tuple(np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8))
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>"):
        gaussianblur_op = vision.GaussianBlur(1, 0)
        _ = gaussianblur_op(image)

    # GaussianBlur operator: Test PIL data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train", "")
    dataset = ds.ImageFolderDataset(data_dir)
    transforms = [
        vision.Decode(to_pil=True),
        vision.GaussianBlur(1, 0),
        vision.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    dataset = dataset.map(input_columns=["image"], operations=transform)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_gaussian_blur_exception_01():
    """
    Feature: GaussianBlur operation
    Description: Testing the GaussianBlur Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # GaussianBlur operator: Test input is 1d
    image = np.random.randint(0, 255, (50,)).astype(np.uint8)
    gaussianblur_op = vision.GaussianBlur(1, 0.8)
    with pytest.raises(RuntimeError, match=r"GaussianBlur: input tensor is not in shape of " + \
                                           "<H,W> or <H,W,C>, but got rank: 1"):
        gaussianblur_op(image)

    # GaussianBlur operator: Test input is 4d
    image = np.random.randint(0, 255, (50, 50, 3, 3)).astype(np.uint8)
    gaussianblur_op = vision.GaussianBlur(1, 0.8)
    with pytest.raises(RuntimeError, match=r"GaussianBlur: input tensor is not in shape of " + \
                                           "<H,W> or <H,W,C>, but got rank: 4"):
        gaussianblur_op(image)

    # GaussianBlur operator: Test input is int
    image = 10
    gaussianblur_op = vision.GaussianBlur(1, 0.8)
    with pytest.raises(TypeError, match=r"Input should be NumPy or PIL image, got <class 'int'>"):
        gaussianblur_op(image)

    # GaussianBlur operator: Test no Parameters
    with pytest.raises(TypeError, match="missing a required argument: 'kernel_size'"):
        vision.GaussianBlur()

    # GaussianBlur operator: Test more Parameters
    with pytest.raises(TypeError, match="too many positional arguments"):
        vision.GaussianBlur(1, 1.0, 3)

    # GaussianBlur operator: Test pepiline kernel_size is 2
    with pytest.raises(ValueError, match="Input kernel_size is not an odd value."):
        vision.GaussianBlur(2, 1.0)

    # GaussianBlur operator: Test kernel_size is 1-list
    with pytest.raises(TypeError, match="Kernel size should be a single integer or a "
                                        "list/tuple \\(kernel_width, kernel_height\\) of length 2."):
        vision.GaussianBlur([1], 1.0)

    # GaussianBlur operator: Test kernel_size is -1
    with pytest.raises(ValueError, match="Input kernel_size is not within the required interval of \\[1, 16777216\\]."):
        vision.GaussianBlur(-1, 1.0)

    # GaussianBlur operator: Test kernel_size is 3-tuple
    with pytest.raises(TypeError, match="Kernel size should be a single integer or a "
                                        "list/tuple \\(kernel_width, kernel_height\\) of length 2."):
        vision.GaussianBlur((1, 3, 5), 1.0)

    # GaussianBlur operator: Test kernel_size is np
    with pytest.raises(TypeError, match="Argument kernel_size with value \\[1 3\\] is not of type \\[\\<class "
                                        "'int'\\>, \\<class 'list'\\>, \\<class 'tuple'\\>\\], but got \\<class "
                                        "'numpy.ndarray'\\>."):
        vision.GaussianBlur(np.array([1, 3]), 1.0)

    # GaussianBlur operator: Test kernel_size is float
    with pytest.raises(TypeError, match="Argument kernel_size\\[1\\] with value 3.0 is not "
                                        "of type \\[\\<class 'int'\\>\\], but got \\<class 'float'\\>."):
        vision.GaussianBlur([3, 3.0], 1.0)

    # GaussianBlur operator: Test kernel_size > 16777216
    with pytest.raises(ValueError, match="Input kernel_size is not within the required interval of \\[1, 16777216\\]."):
        vision.GaussianBlur(16777217, 1.0)

    # GaussianBlur operator: Test kernel_size is str
    with pytest.raises(TypeError, match="Argument kernel_size with value 3 is not of type \\[\\<class 'int'\\>, "
                                        "\\<class 'list'\\>, \\<class 'tuple'\\>\\], but got \\<class 'str'\\>."):
        vision.GaussianBlur("3", 1.0)

    # GaussianBlur operator: Test sigma < 0
    with pytest.raises(ValueError, match="Input sigma is not within the required interval of \\[0, 16777216\\]."):
        vision.GaussianBlur(3, -1.0)

    # GaussianBlur operator: Test sigma > 16777216
    with pytest.raises(ValueError, match="Input sigma is not within the required interval of \\[0, 16777216\\]."):
        vision.GaussianBlur(3, 16777216.1)

    # GaussianBlur operator: Test sigma is str
    with pytest.raises(TypeError, match="Argument sigma with value 3 is not of type \\[\\<class 'numbers.Number'\\>, "
                                        "\\<class 'list'\\>, \\<class 'tuple'\\>\\], but got \\<class 'str'\\>."):
        vision.GaussianBlur(3, "3")

    # GaussianBlur operator: Test sigma is 1-list
    with pytest.raises(TypeError,
                       match="Sigma should be a single number or a list/tuple of length 2 for width and height."):
        vision.GaussianBlur(3, [0.5])

    # GaussianBlur operator: Test sigma is 3-tuple
    with pytest.raises(TypeError,
                       match="Sigma should be a single number or a list/tuple of length 2 for width and height."):
        vision.GaussianBlur(3, (0.1, 0.8, 0.6))

    # GaussianBlur operator: Test sigma is np
    with pytest.raises(TypeError, match="Argument sigma with value \\[0.1 0.8\\] is not of type \\[\\<class "
                                        "'numbers.Number'\\>, \\<class 'list'\\>, \\<class 'tuple'\\>\\], but "
                                        "got \\<class 'numpy.ndarray'\\>."):
        vision.GaussianBlur(3, np.array([0.1, 0.8]))

    # GaussianBlur operator: Test no input
    gaussianblur_op = vision.GaussianBlur(1, 0)
    with pytest.raises(RuntimeError, match="Input Tensor is not valid"):
        gaussianblur_op()


if __name__ == "__main__":
    test_gaussian_blur_pipeline(plot=False)
    test_gaussian_blur_eager()
    test_gaussian_blur_exception()
