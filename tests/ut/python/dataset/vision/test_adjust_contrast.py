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
Testing AdjustContrast op in DE
"""
import numpy as np
from numpy.testing import assert_allclose

import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms as t_trans
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import diff_mse

DATA_DIR = "../data/dataset/testImageNetData/train/"

DATA_DIR_2 = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

TEST_DATA_DATASET_FUNC = "../data/dataset/"


def generate_numpy_random_rgb(shape):
    """
    Only generate floating points that are fractions like n / 256, since they
    are RGB pixels. Some low-precision floating point types in this test can't
    handle arbitrary precision floating points well.
    """
    return np.random.randint(0, 256, shape) / 255.


def test_adjust_contrast_eager(plot=False):
    """
    Feature: AdjustContrast op
    Description: Test AdjustContrast in eager mode
    Expectation: Output is the same as expected output
    """
    # Eager 3-channel
    image_file = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
    img = np.fromfile(image_file, dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = vision.Decode()(img)
    img_adjustcontrast = vision.AdjustContrast(1)(img)
    if plot:
        visualize_image(img, img_adjustcontrast)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_adjustcontrast),
                                                         img_adjustcontrast.shape))
    mse = diff_mse(img_adjustcontrast, img)
    logger.info("MSE= {}".format(str(mse)))
    assert mse == 0


def test_adjust_contrast_invalid_contrast_factor_param():
    """
    Feature: AdjustContrast op
    Description: Test improper parameters for AdjustContrast implementation
    Expectation: Throw ValueError exception and TypeError exception
    """
    logger.info("Test AdjustContrast Python implementation with invalid ignore parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        trans = t_trans.Compose([
            vision.Decode(True),
            vision.Resize((224, 224)),
            vision.AdjustContrast(contrast_factor=-10.0),
            vision.ToTensor()
        ])
        data_set = data_set.map(operations=[trans], input_columns=["image"])
    except ValueError as error:
        logger.info("Got an exception in AdjustContrast: {}".format(str(error)))
        assert "Input contrast_factor is not within the required interval of " in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        trans = t_trans.Compose([
            vision.Decode(True),
            vision.Resize((224, 224)),
            vision.AdjustContrast(contrast_factor=[1, 2]),
            vision.ToTensor()
        ])
        data_set = data_set.map(operations=[trans], input_columns=["image"])
    except TypeError as error:
        logger.info("Got an exception in AdjustContrast: {}".format(str(error)))
        assert "is not of type [<class 'float'>, <class 'int'>], but got" in str(error)


def test_adjust_contrast_pipeline():
    """
    Feature: AdjustContrast op
    Description: Test AdjustContrast in pipeline mode
    Expectation: Output is the same as expected output
    """
    # First dataset
    transforms1 = [vision.Decode(True), vision.Resize([64, 64]), vision.ToTensor()]
    transforms1 = t_trans.Compose(transforms1)
    ds1 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds1 = ds1.map(operations=transforms1, input_columns=["image"])

    # Second dataset
    transforms2 = [
        vision.Decode(True),
        vision.Resize([64, 64]),
        vision.AdjustContrast(1.0),
        vision.ToTensor()
    ]
    transform2 = t_trans.Compose(transforms2)
    ds2 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds2 = ds2.map(operations=transform2, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(ds1.create_dict_iterator(num_epochs=1),
                            ds2.create_dict_iterator(num_epochs=1)):
        num_iter += 1
        ori_img = data1["image"].asnumpy()
        cvt_img = data2["image"].asnumpy()
        assert_allclose(ori_img.flatten(),
                        cvt_img.flatten(),
                        rtol=1e-5,
                        atol=0)
        mse = diff_mse(ori_img, cvt_img)
        logger.info("MSE= {}".format(str(mse)))
        assert mse == 0


def test_adjust_contrast_operation_01():
    """
    Feature: AdjustContrast operation
    Description: Testing the normal functionality of the AdjustContrast operator
    Expectation: The Output is equal to the expected output
    """
    # Test AdjustContrast in pipeline mode
    # First dataset
    transforms1 = [vision.Decode(True), vision.Resize([64, 64]), vision.ToTensor()]
    transforms1 = t_trans.Compose(transforms1)
    ds1 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds1 = ds1.map(operations=transforms1, input_columns=["image"])

    # Second dataset
    transforms2 = [
        vision.Decode(True),
        vision.Resize([64, 64]),
        vision.AdjustContrast(1.0),
        vision.ToTensor()
    ]
    transform2 = t_trans.Compose(transforms2)
    ds2 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds2 = ds2.map(operations=transform2, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(ds1.create_dict_iterator(num_epochs=1),
                            ds2.create_dict_iterator(num_epochs=1)):
        num_iter += 1
        ori_img = data1["image"].asnumpy()
        cvt_img = data2["image"].asnumpy()
        assert_allclose(ori_img.flatten(),
                        cvt_img.flatten(),
                        rtol=1e-5,
                        atol=0)
        mse = diff_mse(ori_img, cvt_img)
        assert mse == 0


def test_adjust_contrast_exception_01():
    """
    Feature: AdjustContrast operation
    Description: Testing the AdjustContrast Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Testing abnormal parameters in the AdjustContrast operator, where contrast_factor is of type list
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        trans = t_trans.Compose([
            vision.Decode(True),
            vision.Resize((224, 224)),
            vision.AdjustContrast(contrast_factor=[1, 2]),
            vision.ToTensor()
        ])
        data_set.map(operations=[trans], input_columns=["image"])
    except TypeError as error:
        assert "is not of type [<class 'float'>, <class 'int'>], but got" in str(error)

    # Testing abnormal parameters in the AdjustContrast operator, where contrast_factor is -10.
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        trans = t_trans.Compose([
            vision.Decode(True),
            vision.Resize((224, 224)),
            vision.AdjustContrast(contrast_factor=-10.0),
            vision.ToTensor()
        ])
        data_set.map(operations=[trans], input_columns=["image"])
    except ValueError as error:
        assert "Input contrast_factor is not within the required interval of " in str(error)


if __name__ == "__main__":
    test_adjust_contrast_eager()
    test_adjust_contrast_invalid_contrast_factor_param()
    test_adjust_contrast_pipeline()
    test_adjust_contrast_operation_01()
    test_adjust_contrast_exception_01()
