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
Testing HorizontalFlip Python API
"""
import cv2

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_vision

from mindspore import log as logger
from util import visualize_image, diff_mse

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
IMAGE_FILE = "../data/dataset/apple.jpg"


def test_horizontal_flip_pipeline(plot=False):
    """
    Test HorizontalFlip of c_transforms
    """
    logger.info("test_horizontal_flip_pipeline")

    # First dataset
    dataset1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = c_vision.Decode()
    horizontal_flip_op = c_vision.HorizontalFlip()
    dataset1 = dataset1.map(operations=decode_op, input_columns=["image"])
    dataset1 = dataset1.map(operations=horizontal_flip_op, input_columns=["image"])

    # Second dataset
    dataset2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    dataset2 = dataset2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        horizontal_flip_ms = data1["image"]
        original = data2["image"]
        horizontal_flip_cv = cv2.flip(original, 1)
        mse = diff_mse(horizontal_flip_ms, horizontal_flip_cv)
        logger.info("horizontal_flip_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1
        if plot:
            visualize_image(original, horizontal_flip_ms, mse, horizontal_flip_cv)


def test_horizontal_flip_eager():
    """
    Test HorizontalFlip with eager mode
    """
    logger.info("test_horizontal_flip_eager")
    img = cv2.imread(IMAGE_FILE)

    img_ms = c_vision.HorizontalFlip()(img)
    img_cv = cv2.flip(img, 1)
    mse = diff_mse(img_ms, img_cv)
    assert mse == 0


if __name__ == "__main__":
    test_horizontal_flip_pipeline(plot=False)
    test_horizontal_flip_eager()
