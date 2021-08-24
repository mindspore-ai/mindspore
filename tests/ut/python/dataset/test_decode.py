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
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore import log as logger
from util import diff_mse

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_decode_op():
    """
    Test Decode op
    """
    logger.info("test_decode_op")

    # Decode with rgb format set to True
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    # Serialize and Load dataset requires using vision.Decode instead of vision.Decode().
    data1 = data1.map(operations=[vision.Decode(True)], input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        actual = item1["image"]
        expected = cv2.imdecode(item2["image"], cv2.IMREAD_COLOR)
        expected = cv2.cvtColor(expected, cv2.COLOR_BGR2RGB)
        assert actual.shape == expected.shape
        mse = diff_mse(actual, expected)
        assert mse == 0


def test_decode_op_tf_file_dataset():
    """
    Test Decode op with tf_file dataset
    """
    logger.info("test_decode_op_tf_file_dataset")

    # Decode with rgb format set to True
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=ds.Shuffle.FILES)
    data1 = data1.map(operations=vision.Decode(True), input_columns=["image"])

    for item in data1.create_dict_iterator(num_epochs=1):
        logger.info('decode == {}'.format(item['image']))

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        actual = item1["image"]
        expected = cv2.imdecode(item2["image"], cv2.IMREAD_COLOR)
        expected = cv2.cvtColor(expected, cv2.COLOR_BGR2RGB)
        assert actual.shape == expected.shape
        mse = diff_mse(actual, expected)
        assert mse == 0


class ImageDataset:
    def __init__(self, data_path, data_type="numpy"):
        self.data = [data_path]
        self.label = np.random.sample((1, 1))
        self.data_type = data_type

    def __getitem__(self, index):
        # use file open and read method
        f = open(self.data[index], 'rb')
        img_bytes = f.read()
        f.close()
        if self.data_type == "numpy":
            img_bytes = np.frombuffer(img_bytes, dtype=np.uint8)

        # return bytes directly
        return (img_bytes, self.label[index])

    def __len__(self):
        return len(self.data)


def test_read_image_decode_op():
    data_path = "../data/dataset/testPK/data/class1/0.jpg"
    dataset1 = ds.GeneratorDataset(ImageDataset(data_path, data_type="numpy"), ["data", "label"])
    dataset2 = ds.GeneratorDataset(ImageDataset(data_path, data_type="bytes"), ["data", "label"])
    decode_op = py_vision.Decode()
    to_tensor = py_vision.ToTensor(output_type=np.int32)
    dataset1 = dataset1.map(operations=[decode_op, to_tensor], input_columns=["data"])
    dataset2 = dataset2.map(operations=[decode_op, to_tensor], input_columns=["data"])

    for item1, item2 in zip(dataset1, dataset2):
        assert np.count_nonzero(item1[0].asnumpy() - item2[0].asnumpy()) == 0


if __name__ == "__main__":
    test_decode_op()
    test_decode_op_tf_file_dataset()
    test_read_image_decode_op()
