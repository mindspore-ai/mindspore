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
Testing AutoContrast op in DE
"""
import os


DATA_DIR = "../data/dataset/testImageNetData/train/"
MNIST_DATA_DIR = "../data/dataset/testMnistData"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

GENERATE_GOLDEN = False


def dir_data():
    """Obtain the dataset"""
    data_list = []
    data_dir1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    data_dir2 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_1.jpg")
    data_dir3 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    data_dir4 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    data_dir5 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    data_dir6 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    data_list.append(data_dir1)
    data_list.append(data_dir2)
    data_list.append(data_dir3)
    data_list.append(data_dir4)
    data_list.append(data_dir5)
    data_list.append(data_dir6)
    return data_list
