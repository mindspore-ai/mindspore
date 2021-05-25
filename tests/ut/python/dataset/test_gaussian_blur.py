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
Testing GaussianBlur Python API
"""
import cv2

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_vision

from mindspore import log as logger
from util import visualize_image, diff_mse

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
IMAGE_FILE = "../data/dataset/apple.jpg"


def test_gaussian_blur_pipeline(plot=False):
    """
    Test GaussianBlur of c_transforms
    """
    logger.info("test_gaussian_blur_pipeline")

    # First dataset
    dataset1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = c_vision.Decode()
    gaussian_blur_op = c_vision.GaussianBlur(3, 3)
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
    Test GaussianBlur with eager mode
    """
    logger.info("test_gaussian_blur_eager")
    img = cv2.imread(IMAGE_FILE)

    img_ms = c_vision.GaussianBlur((3, 5), (3.5, 3.5))(img)
    img_cv = cv2.GaussianBlur(img, (3, 5), 3.5, 3.5)
    mse = diff_mse(img_ms, img_cv)
    assert mse == 0


def test_gaussian_blur_exception():
    """
    Test GaussianBlur with invalid parameters
    """
    logger.info("test_gaussian_blur_exception")
    try:
        _ = c_vision.GaussianBlur([2, 2])
    except ValueError as e:
        logger.info("Got an exception in GaussianBlur: {}".format(str(e)))
        assert "not an odd value" in str(e)
    try:
        _ = c_vision.GaussianBlur(3.0, [3, 3])
    except TypeError as e:
        logger.info("Got an exception in GaussianBlur: {}".format(str(e)))
        assert "not of type [<class 'int'>, <class 'list'>, <class 'tuple'>]" in str(e)
    try:
        _ = c_vision.GaussianBlur(3, -3)
    except ValueError as e:
        logger.info("Got an exception in GaussianBlur: {}".format(str(e)))
        assert "not within the required interval" in str(e)
    try:
        _ = c_vision.GaussianBlur(3, [3, 3, 3])
    except TypeError as e:
        logger.info("Got an exception in GaussianBlur: {}".format(str(e)))
        assert "should be a single number or a list/tuple of length 2" in str(e)


if __name__ == "__main__":
    test_gaussian_blur_pipeline(plot=False)
    test_gaussian_blur_eager()
    test_gaussian_blur_exception()
