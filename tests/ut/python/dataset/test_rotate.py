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
Testing Rotate Python API
"""
import cv2

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore import log as logger
from mindspore.dataset.vision.utils import Inter
from util import visualize_image, diff_mse

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
IMAGE_FILE = "../data/dataset/apple.jpg"


def test_rotate_pipeline_with_expanding(plot=False):
    """
    Test Rotate of c_transforms with expanding
    """
    logger.info("test_rotate_pipeline_with_expanding")

    # First dataset
    dataset1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = c_vision.Decode()
    rotate_op = c_vision.Rotate(90, expand=True)
    dataset1 = dataset1.map(operations=decode_op, input_columns=["image"])
    dataset1 = dataset1.map(operations=rotate_op, input_columns=["image"])

    # Second dataset
    dataset2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    dataset2 = dataset2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        rotate_ms = data1["image"]
        original = data2["image"]
        rotate_cv = cv2.rotate(original, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mse = diff_mse(rotate_ms, rotate_cv)
        logger.info("rotate_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1
        if plot:
            visualize_image(original, rotate_ms, mse, rotate_cv)


def test_rotate_pipeline_without_expanding():
    """
    Test Rotate of c_transforms without expanding
    """
    logger.info("test_rotate_pipeline_without_expanding")

    # Create a Dataset then decode and rotate the image
    dataset = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = c_vision.Decode()
    resize_op = c_vision.Resize((64, 128))
    rotate_op = c_vision.Rotate(30)
    dataset = dataset.map(operations=decode_op, input_columns=["image"])
    dataset = dataset.map(operations=resize_op, input_columns=["image"])
    dataset = dataset.map(operations=rotate_op, input_columns=["image"])

    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        rotate_img = data["image"]
        assert rotate_img.shape == (64, 128, 3)


def test_rotate_eager():
    """
    Test Rotate with eager mode
    """
    logger.info("test_rotate_eager")
    img = cv2.imread(IMAGE_FILE)
    resize_img = c_vision.Resize((32, 64))(img)
    rotate_img = c_vision.Rotate(-90, expand=True)(resize_img)
    assert rotate_img.shape == (64, 32, 3)


def test_rotate_exception():
    """
    Test Rotate with invalid parameters
    """
    logger.info("test_rotate_exception")
    try:
        _ = c_vision.Rotate("60")
    except TypeError as e:
        logger.info("Got an exception in Rotate: {}".format(str(e)))
        assert "not of type [<class 'float'>, <class 'int'>]" in str(e)
    try:
        _ = c_vision.Rotate(30, Inter.BICUBIC, False, (0, 0, 0))
    except ValueError as e:
        logger.info("Got an exception in Rotate: {}".format(str(e)))
        assert "Value center needs to be a 2-tuple." in str(e)
    try:
        _ = c_vision.Rotate(-120, Inter.NEAREST, False, (-1, -1), (255, 255))
    except TypeError as e:
        logger.info("Got an exception in Rotate: {}".format(str(e)))
        assert "fill_value should be a single integer or a 3-tuple." in str(e)


if __name__ == "__main__":
    test_rotate_pipeline_with_expanding(False)
    test_rotate_pipeline_without_expanding()
    test_rotate_eager()
    test_rotate_exception()
