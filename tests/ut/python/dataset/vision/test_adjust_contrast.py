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
