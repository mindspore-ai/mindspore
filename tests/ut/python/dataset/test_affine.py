# Copyright 2022 Huawei Technologies Co., Ltd
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
Testing Affine op in DE
"""
import numpy as np

from mindspore import log as logger
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from util import visualize_list, diff_mse

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
MNIST_DATA_DIR = "../data/dataset/testMnistData"


def test_affine_exception_degrees_type():
    """
    Feature: Test Affine degrees type
    Description: Input the type of degrees is list
    Expectation: Got an exception to raise TyoeError
    """
    logger.info("test_affine_exception_degrees_type")
    try:
        _ = vision.Affine(degrees=[15.0], translate=[-1, 1], scale=1.0, shear=[1, 1])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Argument degrees with value [15.0] is not of type [<class 'int'>, <class 'float'>], " \
                         "but got <class 'list'>."


def test_affine_exception_scale_value():
    """
    Feature: Test Affine(scale is not valid)
    Description: Input scale is not valid
    Expectation: Got an exception to raise ValueError
    """
    logger.info("test_affine_exception_scale_value")
    try:
        _ = vision.Affine(degrees=15, translate=[1, 1], scale=0.0, shear=10)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input scale must be greater than 0."

    try:
        _ = vision.Affine(degrees=15, translate=[1, 1], scale=-0.2, shear=10)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input scale must be greater than 0."


def test_affine_exception_shear_size():
    """
    Feature: Test Affine(shear is not list or a tuple of length 2)
    Description: Input shear is not list or a tuple of length 2
    Expectation: Got an exception to raise TypeError
    """
    logger.info("test_affine_shear_size")
    try:
        _ = vision.Affine(degrees=15, translate=[1, 1], scale=1.5, shear=[1.5, 3.5, 3.5])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "The length of shear should be 2."


def test_affine_exception_translate_size():
    """
    Feature: Test Affine(translate is not list or a tuple of length 2)
    Description: Input translate is not list or a tuple of length 2
    Expectation: Got an exception to raise TypeError
    """
    logger.info("test_affine_exception_translate_size")
    try:
        _ = vision.Affine(degrees=15, translate=[1, 1, 1], scale=1.9, shear=[10.1])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "The length of translate should be 2."


def test_affine_exception_translate_value():
    """
    Feature: Test Affine(translate value)
    Description: Input translate is not a sequence
    Expectation: Got an exception to raise TypeError
    """
    logger.info("test_affine_exception_translate_value")
    try:
        _ = vision.Affine(degrees=15, translate=(0.1,), scale=2.1, shear=[1.5, 1.5], resample=vision.Inter.BILINEAR)
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "The length of translate should be 2."


def test_affine_pipeline(plot=False):
    """
    Feature: Affine
    Description: Test Affine in pipeline mode
    Expectation: The dataset is processed as expected
    """
    # First dataset
    transforms_list = transforms.Compose([vision.Decode(),
                                          vision.Resize([64, 64])])
    dataset = ds.TFRecordDataset(DATA_DIR,
                                 SCHEMA_DIR,
                                 columns_list=["image"],
                                 shuffle=False)
    dataset = dataset.map(operations=transforms_list, input_columns=["image"])

    # Second dataset
    affine_transforms_list = transforms.Compose([vision.Decode(),
                                                 vision.Resize([64, 64]),
                                                 vision.Affine(degrees=15, translate=[0.2, 0.2],
                                                               scale=1.1, shear=[10.0, 10.0])])
    affine_dataset = ds.TFRecordDataset(DATA_DIR,
                                        SCHEMA_DIR,
                                        columns_list=["image"],
                                        shuffle=False)
    affine_dataset = affine_dataset.map(operations=affine_transforms_list, input_columns=["image"])

    num_image = 0
    image_list = []
    affine_image_list = []
    for image, affine_image in zip(dataset.create_dict_iterator(num_epochs=1, output_numpy=True),
                                   affine_dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_image += 1
        image_list.append(image["image"])
        affine_image_list.append(affine_image["image"])

    assert num_image == 3

    if plot:
        visualize_list(image_list, affine_image_list)


def test_affine_eager():
    """
    Feature: Affine op
    Description: Test eager support for Affine Cpp implementation
    Expectation: The output data is the same as the result of cv2.warpAffine
    """
    img_in = np.array([[[211, 192, 16], [146, 176, 190], [103, 86, 18], [23, 194, 246]],
                       [[17, 86, 38], [180, 162, 43], [197, 198, 224], [109, 3, 195]],
                       [[172, 197, 74], [33, 52, 136], [120, 185, 76], [105, 23, 221]],
                       [[197, 50, 36], [82, 187, 119], [124, 193, 164], [181, 8, 11]]], dtype=np.uint8)

    affine_op1 = vision.Affine(degrees=30, translate=[0.5, 0.5], scale=1.0, shear=[0, 0])
    img_out1 = affine_op1(img_in)
    exp1 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [211, 192, 16]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [211, 192, 16]],
                     [[0, 0, 0], [0, 0, 0], [172, 197, 74], [180, 162, 43]]], dtype=np.uint8)

    affine_op2 = vision.Affine(degrees=30, translate=[0.5, 0.5], scale=1.0, shear=[10, 10])
    img_out2 = affine_op2(img_in)
    exp2 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [211, 192, 16]],
                     [[0, 0, 0], [0, 0, 0], [172, 197, 74], [180, 162, 43]]], dtype=np.uint8)

    affine_op3 = vision.Affine(degrees=30, translate=[0.5, 0.5], scale=1.2, shear=5)
    img_out3 = affine_op3(img_in)
    exp3 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [211, 192, 16]],
                     [[0, 0, 0], [0, 0, 0], [17, 86, 38], [17, 86, 38]],
                     [[0, 0, 0], [172, 197, 74], [172, 197, 74], [180, 162, 43]]], dtype=np.uint8)

    mse1 = diff_mse(img_out1, exp1)
    mse2 = diff_mse(img_out2, exp2)
    mse3 = diff_mse(img_out3, exp3)
    assert mse1 < 0.001 and mse2 < 0.001 and mse3 < 0.001


if __name__ == "__main__":
    test_affine_exception_degrees_type()
    test_affine_exception_scale_value()
    test_affine_exception_shear_size()
    test_affine_exception_translate_size()
    test_affine_exception_translate_value()
    test_affine_pipeline(plot=False)
    test_affine_eager()
