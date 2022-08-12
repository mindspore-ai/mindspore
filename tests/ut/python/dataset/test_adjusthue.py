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
Testing AdjustHue op in DE
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import diff_mse

DATA_DIR = "../data/dataset/testImageNetData/train/"
MNIST_DATA_DIR = "../data/dataset/testMnistData"

DATA_DIR_2 = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def generate_numpy_random_rgb(shape):
    """
    Only generate floating points that are fractions like n / 256, since they
    are RGB pixels. Some low-precision floating point types in this test can't
    handle arbitrary precision floating points well.
    """
    return np.random.randint(0, 256, shape) / 255.


def test_adjust_hue_eager(plot=False):
    """
    Feature: AdjustHue op
    Description: Test eager support for AdjustHue implementation
    Expectation: Output is the same as expected output
    """
    # Eager 3-channel
    image_file = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
    img = np.fromfile(image_file, dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = vision.Decode()(img)
    img_adjusthue = vision.AdjustHue(0)(img)

    if plot:
        visualize_image(img, img_adjusthue)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_adjusthue),
                                                         img_adjusthue.shape))
    mse = diff_mse(img_adjusthue, img)
    logger.info("MSE= {}".format(str(mse)))
    assert mse < 0.001


def test_adjust_hue_invalid_hue_factor_param():
    """
    Feature: AdjustHue op
    Description: Test improper parameters for AdjustHue implementation
    Expectation: Throw ValueError exception and TypeError exception
    """
    logger.info("Test AdjustHue implementation with invalid ignore parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        trans = mindspore.dataset.transforms.transforms.Compose([
            vision.Decode(True),
            vision.Resize((224, 224)),
            vision.AdjustHue(hue_factor=-1.0),
            vision.ToTensor()
        ])
        data_set = data_set.map(operations=[trans], input_columns=["image"])
    except ValueError as error:
        logger.info("Got an exception in AdjustHue: {}".format(str(error)))
        assert "Input hue_factor is not within the required interval of " in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        trans = mindspore.dataset.transforms.transforms.Compose([
            vision.Decode(True),
            vision.Resize((224, 224)),
            vision.AdjustHue(hue_factor=[1, 2]),
            vision.ToTensor()
        ])
        data_set = data_set.map(operations=[trans], input_columns=["image"])
    except TypeError as error:
        logger.info("Got an exception in AdjustHue: {}".format(str(error)))
        assert "is not of type [<class 'float'>, <class 'int'>], but got" in str(error)


def test_adjust_hue_pipeline():
    """
    Feature: AdjustHue op
    Description: Test AdjustHue implementation Pipeline
    Expectation: Output is equal to the expected output
    """
    # First dataset
    transforms1 = [vision.Decode(), vision.Resize([64, 64]), vision.ToTensor()]
    transforms1 = mindspore.dataset.transforms.transforms.Compose(
        transforms1)
    ds1 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds1 = ds1.map(operations=transforms1, input_columns=["image"])

    # Second dataset
    transforms2 = [
        vision.Decode(),
        vision.Resize([64, 64]),
        vision.AdjustHue(0),
        vision.ToTensor()
    ]
    transform2 = mindspore.dataset.transforms.transforms.Compose(
        transforms2)
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
        mse = diff_mse(ori_img, cvt_img)
        logger.info("MSE= {}".format(str(mse)))
        assert mse < 0.001


if __name__ == "__main__":
    test_adjust_hue_eager()
    test_adjust_hue_invalid_hue_factor_param()
    test_adjust_hue_pipeline()
