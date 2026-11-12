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
Testing RandomResize op in DE
"""
import numpy as np
import os
from PIL import Image


DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

GENERATE_GOLDEN = False


def generator_mc(maxid=3):
    """ return five image """
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_jpg) as image:
        image = np.array(image)
    with Image.open(image_bmp) as image1:
        image1 = np.array(image1)
    with Image.open(image_gif) as image2:
        image2 = np.array(image2)
    for _ in range(maxid):
        yield image, image, image1, image1, image2
