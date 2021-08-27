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
Testing ConvertColor op in DE
"""
import cv2

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.utils as mode
from mindspore import log as logger
from util import visualize_image, diff_mse

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
IMAGE_FILE = "../data/dataset/apple.jpg"


def convert_color(ms_convert, cv_convert, plot=False):
    """
    ConvertColor with different mode.
    """
    # First dataset
    dataset1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = c_vision.Decode()
    convertcolor_op = c_vision.ConvertColor(ms_convert)
    dataset1 = dataset1.map(operations=decode_op, input_columns=["image"])
    dataset1 = dataset1.map(operations=convertcolor_op, input_columns=["image"])

    # Second dataset
    dataset2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    dataset2 = dataset2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        convertcolor_ms = data1["image"]
        original = data2["image"]
        convertcolor_cv = cv2.cvtColor(original, cv_convert)
        mse = diff_mse(convertcolor_ms, convertcolor_cv)
        logger.info("convertcolor_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1
        if plot:
            visualize_image(original, convertcolor_ms, mse, convertcolor_cv)


def test_convertcolor_pipeline(plot=False):
    """
    Test ConvertColor of c_transforms
    """
    logger.info("test_convertcolor_pipeline")
    convert_color(mode.ConvertMode.COLOR_BGR2GRAY, cv2.COLOR_BGR2GRAY, plot)
    convert_color(mode.ConvertMode.COLOR_BGR2RGB, cv2.COLOR_BGR2RGB, plot)
    convert_color(mode.ConvertMode.COLOR_BGR2BGRA, cv2.COLOR_BGR2BGRA, plot)


def test_convertcolor_eager():
    """
    Test ConvertColor with eager mode
    """
    logger.info("test_convertcolor")
    img = cv2.imread(IMAGE_FILE)

    img_ms = c_vision.ConvertColor(mode.ConvertMode.COLOR_BGR2GRAY)(img)
    img_expect = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mse = diff_mse(img_ms, img_expect)
    assert mse == 0


if __name__ == "__main__":
    test_convertcolor_pipeline(plot=False)
    test_convertcolor_eager()
