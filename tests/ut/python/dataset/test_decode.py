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
Testing Decode op in DE
"""
import cv2
import mindspore.dataset.transforms.vision.c_transforms as vision
import numpy as np
import mindspore.dataset as ds
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def diff_mse(in1, in2):
    mse = (np.square(in1.astype(float) / 255 - in2.astype(float) / 255)).mean()
    return mse * 100


def test_decode_op():
    """
    Test Decode op
    """
    logger.info("test_decode_op")

    # Decode with rgb format set to True
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    # Serialize and Load dataset requires using vision.Decode instead of vision.Decode().
    data1 = data1.map(input_columns=["image"], operations=[vision.Decode(True)])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        actual = item1["image"]
        expected = cv2.imdecode(item2["image"], cv2.IMREAD_COLOR)
        expected = cv2.cvtColor(expected, cv2.COLOR_BGR2RGB)
        assert actual.shape == expected.shape
        diff = actual - expected
        mse = np.sum(np.power(diff, 2))
        assert mse == 0


def test_decode_op_tf_file_dataset():
    """
    Test Decode op with tf_file dataset
    """
    logger.info("test_decode_op_tf_file_dataset")

    # Decode with rgb format set to True
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=ds.Shuffle.FILES)
    data1 = data1.map(input_columns=["image"], operations=vision.Decode(True))

    for item in data1.create_dict_iterator():
        logger.info('decode == {}'.format(item['image']))

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        actual = item1["image"]
        expected = cv2.imdecode(item2["image"], cv2.IMREAD_COLOR)
        expected = cv2.cvtColor(expected, cv2.COLOR_BGR2RGB)
        assert actual.shape == expected.shape
        diff = actual - expected
        mse = np.sum(np.power(diff, 2))
        assert mse == 0


if __name__ == "__main__":
    test_decode_op()
    test_decode_op_tf_file_dataset()
