# Copyright 2019 Huawei Technologies Co., Ltd
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
import pytest

import mindspore.dataset as ds
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images2/train-0000-of-0001.data",
            "../data/dataset/test_tf_file_3_images2/train-0000-of-0002.data",
            "../data/dataset/test_tf_file_3_images2/train-0000-of-0003.data",
            "../data/dataset/test_tf_file_3_images2/train-0000-of-0004.data"]

SCHEMA_DIR = "../data/dataset/test_tf_file_3_images2/datasetSchema.json"


def test_tf_file_normal():
    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    data1 = data1.repeat(1)
    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 12


if __name__ == '__main__':
    logger.info('=======test normal=======')
    test_tf_file_normal()
