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
"""
Testing the one_hot op in DE
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as data_trans
import mindspore.dataset.transforms.vision.c_transforms as vision
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def one_hot(index, depth):
    """
    Apply the one_hot
    """
    arr = np.zeros([1, depth], dtype=np.int32)
    arr[0, index] = 1
    return arr


def test_one_hot():
    """
    Test one_hot
    """
    logger.info("Test one_hot")

    depth = 10

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    one_hot_op = data_trans.OneHot(depth)
    data1 = data1.map(input_columns=["label"], operations=one_hot_op, columns_order=["label"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["label"], shuffle=False)

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        assert len(item1) == len(item2)
        label1 = item1["label"]
        label2 = one_hot(item2["label"][0], depth)
        mse = np.sum(label1 - label2)
        logger.info("DE one_hot: {}, Numpy one_hot: {}, diff: {}".format(label1, label2, mse))
        num_iter += 1


if __name__ == "__main__":
    test_one_hot()
