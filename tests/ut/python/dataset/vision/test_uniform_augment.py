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
Testing UniformAugment in DE
"""
import os


DATA_DIR = "../data/dataset/testImageNetData/train/"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")


def py_func(x):
    '''Python custom methods'''
    return x
