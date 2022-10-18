# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
import time

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TF_FILES = ["../data/dataset/testTFTestAllTypes/test.data"]
TF_SCHEMA_FILE = "../data/dataset/testTFTestAllTypes/datasetSchema.json"


def test_case_0():
    """
    Feature: Device_que op
    Description: Test device_que op after a repeat op
    Expectation: Runs successfully
    """
    # apply dataset operations
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    # define parameters
    repeat_count = 2
    data = data.repeat(repeat_count)

    data = data.device_que()
    data.send()
    time.sleep(0.1)
    data.stop_send()


def test_case_1():
    """
    Feature: Device_que op
    Description: Test device_que op after a batch op
    Expectation: Runs successfully
    """
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    # define data augmentation parameters
    resize_height, resize_width = 224, 224

    # define map operations
    decode_op = vision.Decode()
    resize_op = vision.Resize((resize_height, resize_width))

    # apply map operations on images
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=resize_op, input_columns=["image"])

    batch_size = 3
    data = data.batch(batch_size, drop_remainder=True)

    data = data.device_que()
    data.send()
    time.sleep(0.1)
    data.stop_send()


def test_case_2():
    """
    Feature: Device_que op
    Description: Test device_que op after a batch op then a repeat op
    Expectation: Runs successfully
    """
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    # define data augmentation parameters
    resize_height, resize_width = 224, 224

    # define map operations
    decode_op = vision.Decode()
    resize_op = vision.Resize((resize_height, resize_width))

    # apply map operations on images
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=resize_op, input_columns=["image"])

    batch_size = 2
    data = data.batch(batch_size, drop_remainder=True)

    data = data.repeat(2)

    data = data.device_que()
    assert data.get_repeat_count() == 2
    data.send()
    time.sleep(0.1)
    data.stop_send()


def test_case_3():
    """
    Feature: Device_que op
    Description: Test device_que op after a repeat op then a batch op
    Expectation: Runs successfully
    """
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    # define data augmentation parameters
    resize_height, resize_width = 224, 224

    # define map operations
    decode_op = vision.Decode()
    resize_op = vision.Resize((resize_height, resize_width))

    # apply map operations on images
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=resize_op, input_columns=["image"])

    data = data.repeat(2)

    batch_size = 2
    data = data.batch(batch_size, drop_remainder=True)

    data = data.device_que()
    data.send()
    time.sleep(0.1)
    data.stop_send()


def test_case_tf_file():
    """
    Feature: Device_que op
    Description: Test device_que op using TFRecordDataset
    Expectation: Runs successfully
    """
    data = ds.TFRecordDataset(TF_FILES, TF_SCHEMA_FILE, shuffle=ds.Shuffle.FILES)

    data = data.device_que()
    data.send()
    time.sleep(0.1)
    data.stop_send()


if __name__ == '__main__':
    logger.info('===========now test Repeat============')
    test_case_0()

    logger.info('===========now test Batch============')
    test_case_1()

    logger.info('===========now test Batch & Repeat============')
    test_case_2()

    logger.info('===========now test Repeat & Batch============')
    test_case_3()

    logger.info('===========now test tf file============')
    test_case_tf_file()
