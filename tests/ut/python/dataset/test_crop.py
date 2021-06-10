# Copyright 2021 Huawei Technologies Co., Ltd
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
Testing Crop op in DE
"""
import cv2

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_vision

from mindspore import log as logger
from util import visualize_image, diff_mse

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
IMAGE_FILE = "../data/dataset/apple.jpg"


def test_crop_pipeline(plot=False):
    """
    Test Crop of c_transforms
    """
    logger.info("test_crop_pipeline")

    # First dataset
    dataset1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = c_vision.Decode()
    crop_op = c_vision.Crop((0, 0), (20, 25))
    dataset1 = dataset1.map(operations=decode_op, input_columns=["image"])
    dataset1 = dataset1.map(operations=crop_op, input_columns=["image"])

    # Second dataset
    dataset2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    dataset2 = dataset2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        crop_ms = data1["image"]
        original = data2["image"]
        crop_expect = original[0:20, 0:25]
        mse = diff_mse(crop_ms, crop_expect)
        logger.info("crop_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1
        if plot:
            visualize_image(original, crop_ms, mse, crop_expect)


def test_crop_eager():
    """
    Test Crop with eager mode
    """
    logger.info("test_crop_eager")
    img = cv2.imread(IMAGE_FILE)

    img_ms = c_vision.Crop((20, 50), (30, 50))(img)
    img_expect = img[20:50, 50:100]
    mse = diff_mse(img_ms, img_expect)
    assert mse == 0


def test_crop_exception():
    """
    Test Crop with invalid parameters
    """
    logger.info("test_crop_exception")
    try:
        _ = c_vision.Crop([-10, 0], [20])
    except ValueError as e:
        logger.info("Got an exception in Crop: {}".format(str(e)))
        assert "not within the required interval of [0, 2147483647]" in str(e)
    try:
        _ = c_vision.Crop([0, 5.2], [10, 10])
    except TypeError as e:
        logger.info("Got an exception in Crop: {}".format(str(e)))
        assert "not of type [<class 'int'>]" in str(e)
    try:
        _ = c_vision.Crop([0], [28])
    except TypeError as e:
        logger.info("Got an exception in Crop: {}".format(str(e)))
        assert "Coordinates should be a list/tuple (y, x) of length 2." in str(e)
    try:
        _ = c_vision.Crop((0, 0), -1)
    except ValueError as e:
        logger.info("Got an exception in Crop: {}".format(str(e)))
        assert "not within the required interval of [1, 16777216]" in str(e)
    try:
        _ = c_vision.Crop((0, 0), (10.5, 15))
    except TypeError as e:
        logger.info("Got an exception in Crop: {}".format(str(e)))
        assert "not of type [<class 'int'>]" in str(e)
    try:
        _ = c_vision.Crop((0, 0), (0, 10, 20))
    except TypeError as e:
        logger.info("Got an exception in Crop: {}".format(str(e)))
        assert "Size should be a single integer or a list/tuple (h, w) of length 2." in str(e)


if __name__ == "__main__":
    test_crop_pipeline(plot=False)
    test_crop_eager()
    test_crop_exception()
