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
import mindspore.dataset.transforms.vision.c_transforms as vision
import mindspore.dataset.transforms.c_transforms as data_trans
import pytest

import mindspore.dataset as ds
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_case_repeat():
    """
    a simple repeat operation.
    """
    logger.info("Test Simple Repeat")
    # define parameters
    repeat_count = 2

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is: {}".format(item["image"]))
        logger.info("label is: {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))


def test_case_shuffle():
    """
        a simple shuffle operation.
    """
    logger.info("Test Simple Shuffle")
    # define parameters
    buffer_size = 8
    seed = 10

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    ds.config.set_seed(seed)
    data1 = data1.shuffle(buffer_size=buffer_size)

    for item in data1.create_dict_iterator():
        logger.info("image is: {}".format(item["image"]))
        logger.info("label is: {}".format(item["label"]))


def test_case_0():
    """
    Test Repeat then Shuffle
    """
    logger.info("Test Repeat then Shuffle")
    # define parameters
    repeat_count = 2
    buffer_size = 7
    seed = 9

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    data1 = data1.repeat(repeat_count)
    ds.config.set_seed(seed)
    data1 = data1.shuffle(buffer_size=buffer_size)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is: {}".format(item["image"]))
        logger.info("label is: {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))


def test_case_0_reverse():
    """
    Test Shuffle then Repeat
    """
    logger.info("Test Shuffle then Repeat")
    # define parameters
    repeat_count = 2
    buffer_size = 10
    seed = 9

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    ds.config.set_seed(seed)
    data1 = data1.shuffle(buffer_size=buffer_size)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is: {}".format(item["image"]))
        logger.info("label is: {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))


def test_case_3():
    """
    Test Map
    """
    logger.info("Test Map Rescale and Resize, then Shuffle")
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    # define data augmentation parameters
    rescale = 1.0 / 255.0
    shift = 0.0
    resize_height, resize_width = 224, 224

    # define map operations
    decode_op = vision.Decode()
    rescale_op = vision.Rescale(rescale, shift)
    # resize_op = vision.Resize(resize_height, resize_width,
    #                            InterpolationMode.DE_INTER_LINEAR)  # Bilinear mode
    resize_op = vision.Resize((resize_height, resize_width))

    # apply map operations on images
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=rescale_op)
    data1 = data1.map(input_columns=["image"], operations=resize_op)

    # # apply ont-hot encoding on labels
    num_classes = 4
    one_hot_encode = data_trans.OneHot(num_classes)  # num_classes is input argument
    data1 = data1.map(input_columns=["label"], operations=one_hot_encode)
    #
    # # apply Datasets
    buffer_size = 100
    seed = 10
    batch_size = 2
    ds.config.set_seed(seed)
    data1 = data1.shuffle(buffer_size=buffer_size)  # 10000 as in imageNet train script
    data1 = data1.batch(batch_size, drop_remainder=True)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is: {}".format(item["image"]))
        logger.info("label is: {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))


if __name__ == '__main__':
    logger.info('===========now test Repeat============')
    # logger.info('Simple Repeat')
    test_case_repeat()
    logger.info('\n')

    logger.info('===========now test Shuffle===========')
    # logger.info('Simple Shuffle')
    test_case_shuffle()
    logger.info('\n')

    # Note: cannot work with different shapes, hence not for image
    # logger.info('===========now test Batch=============')
    # # logger.info('Simple Batch')
    # test_case_batch()
    # logger.info('\n')

    logger.info('===========now test case 0============')
    # logger.info('Repeat then Shuffle')
    test_case_0()
    logger.info('\n')

    logger.info('===========now test case 0 reverse============')
    # # logger.info('Shuffle then  Repeat')
    test_case_0_reverse()
    logger.info('\n')

    # logger.info('===========now test case 1============')
    # # logger.info('Repeat with Batch')
    # test_case_1()
    # logger.info('\n')

    # logger.info('===========now test case 2============')
    # # logger.info('Batch with Shuffle')
    # test_case_2()
    # logger.info('\n')

    # for image augmentation only
    logger.info('===========now test case 3============')
    logger.info('Map then Shuffle')
    test_case_3()
    logger.info('\n')
