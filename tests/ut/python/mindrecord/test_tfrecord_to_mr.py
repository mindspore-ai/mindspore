# Copyright 2020 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""test tfrecord to mindrecord tool"""
import collections
from importlib import import_module
import os

import numpy as np
import pytest
from mindspore import log as logger
from mindspore.mindrecord import FileReader
from mindspore.mindrecord import TFRecordToMR

SupportedTensorFlowVersion = '2.1.0'

try:
    tf = import_module("tensorflow")    # just used to convert tfrecord to mindrecord
except ModuleNotFoundError:
    logger.warning("tensorflow module not found.")
    tf = None

TFRECORD_DATA_DIR = "../data/mindrecord/testTFRecordData"
TFRECORD_FILE_NAME = "test.tfrecord"
MINDRECORD_FILE_NAME = "test.mindrecord"
PARTITION_NUM = 1

def verify_data(transformer, reader):
    """Verify the data by read from mindrecord"""
    tf_iter = transformer.tfrecord_iterator()
    mr_iter = reader.get_next()

    count = 0
    for tf_item, mr_item in zip(tf_iter, mr_iter):
        count = count + 1
        assert len(tf_item) == 6
        assert len(mr_item) == 6
        for key, value in tf_item.items():
            logger.info("key: {}, tfrecord: value: {}, mindrecord: value: {}".format(key, value, mr_item[key]))
            if isinstance(value, np.ndarray):
                assert (value == mr_item[key]).all()
            else:
                assert value == mr_item[key]
    assert count == 10

def generate_tfrecord():
    def create_int_feature(values):
        if isinstance(values, list):
            feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))  # values: [int, int, int]
        else:
            feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))      # values: int
        return feature

    def create_float_feature(values):
        if isinstance(values, list):
            feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))  # values: [float, float]
        else:
            feature = tf.train.Feature(float_list=tf.train.FloatList(value=[values]))      # values: float
        return feature

    def create_bytes_feature(values):
        if isinstance(values, bytes):
            feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))      # values: bytes
        else:
            # values: string
            feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(values, encoding='utf-8')]))
        return feature

    writer = tf.io.TFRecordWriter(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME))

    example_count = 0
    for i in range(10):
        file_name = "000" + str(i) + ".jpg"
        image_bytes = bytes(str("aaaabbbbcccc" + str(i)), encoding="utf-8")
        int64_scalar = i
        float_scalar = float(i)
        int64_list = [i, i+1, i+2, i+3, i+4, i+1234567890]
        float_list = [float(i), float(i+1), float(i+2.8), float(i+3.2),
                      float(i+4.4), float(i+123456.9), float(i+98765432.1)]

        features = collections.OrderedDict()
        features["file_name"] = create_bytes_feature(file_name)
        features["image_bytes"] = create_bytes_feature(image_bytes)
        features["int64_scalar"] = create_int_feature(int64_scalar)
        features["float_scalar"] = create_float_feature(float_scalar)
        features["int64_list"] = create_int_feature(int64_list)
        features["float_list"] = create_float_feature(float_list)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        example_count += 1
    writer.close()
    logger.info("Write {} rows in tfrecord.".format(example_count))

def test_tfrecord_to_mindrecord():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    generate_tfrecord()
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME))

    feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                    "float_list": tf.io.FixedLenFeature([7], tf.float32),
                    }

    if os.path.exists(MINDRECORD_FILE_NAME):
        os.remove(MINDRECORD_FILE_NAME)
    if os.path.exists(MINDRECORD_FILE_NAME + ".db"):
        os.remove(MINDRECORD_FILE_NAME + ".db")

    tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME),
                                        MINDRECORD_FILE_NAME, feature_dict, ["image_bytes"])
    tfrecord_transformer.transform()

    assert os.path.exists(MINDRECORD_FILE_NAME)
    assert os.path.exists(MINDRECORD_FILE_NAME + ".db")

    fr_mindrecord = FileReader(MINDRECORD_FILE_NAME)
    verify_data(tfrecord_transformer, fr_mindrecord)

    os.remove(MINDRECORD_FILE_NAME)
    os.remove(MINDRECORD_FILE_NAME + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME))

def test_tfrecord_to_mindrecord_scalar_with_1():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    generate_tfrecord()
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME))

    feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([1], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([1], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                    "float_list": tf.io.FixedLenFeature([7], tf.float32),
                    }

    if os.path.exists(MINDRECORD_FILE_NAME):
        os.remove(MINDRECORD_FILE_NAME)
    if os.path.exists(MINDRECORD_FILE_NAME + ".db"):
        os.remove(MINDRECORD_FILE_NAME + ".db")

    tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME),
                                        MINDRECORD_FILE_NAME, feature_dict, ["image_bytes"])
    tfrecord_transformer.transform()

    assert os.path.exists(MINDRECORD_FILE_NAME)
    assert os.path.exists(MINDRECORD_FILE_NAME + ".db")

    fr_mindrecord = FileReader(MINDRECORD_FILE_NAME)
    verify_data(tfrecord_transformer, fr_mindrecord)

    os.remove(MINDRECORD_FILE_NAME)
    os.remove(MINDRECORD_FILE_NAME + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME))

def test_tfrecord_to_mindrecord_scalar_with_1_list_small_len_exception():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    generate_tfrecord()
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME))

    feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([1], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([1], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                    "float_list": tf.io.FixedLenFeature([2], tf.float32),
                    }

    if os.path.exists(MINDRECORD_FILE_NAME):
        os.remove(MINDRECORD_FILE_NAME)
    if os.path.exists(MINDRECORD_FILE_NAME + ".db"):
        os.remove(MINDRECORD_FILE_NAME + ".db")

    with pytest.raises(ValueError):
        tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME),
                                            MINDRECORD_FILE_NAME, feature_dict, ["image_bytes"])
        tfrecord_transformer.transform()

    if os.path.exists(MINDRECORD_FILE_NAME):
        os.remove(MINDRECORD_FILE_NAME)
    if os.path.exists(MINDRECORD_FILE_NAME + ".db"):
        os.remove(MINDRECORD_FILE_NAME + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME))

def test_tfrecord_to_mindrecord_list_with_diff_type_exception():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    generate_tfrecord()
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME))

    feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([1], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([1], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.float32),
                    "float_list": tf.io.FixedLenFeature([7], tf.float32),
                    }

    if os.path.exists(MINDRECORD_FILE_NAME):
        os.remove(MINDRECORD_FILE_NAME)
    if os.path.exists(MINDRECORD_FILE_NAME + ".db"):
        os.remove(MINDRECORD_FILE_NAME + ".db")

    with pytest.raises(ValueError):
        tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME),
                                            MINDRECORD_FILE_NAME, feature_dict, ["image_bytes"])
        tfrecord_transformer.transform()

    if os.path.exists(MINDRECORD_FILE_NAME):
        os.remove(MINDRECORD_FILE_NAME)
    if os.path.exists(MINDRECORD_FILE_NAME + ".db"):
        os.remove(MINDRECORD_FILE_NAME + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME))

def test_tfrecord_to_mindrecord_list_without_bytes_type():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    generate_tfrecord()
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME))

    feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([1], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([1], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                    "float_list": tf.io.FixedLenFeature([7], tf.float32),
                    }

    if os.path.exists(MINDRECORD_FILE_NAME):
        os.remove(MINDRECORD_FILE_NAME)
    if os.path.exists(MINDRECORD_FILE_NAME + ".db"):
        os.remove(MINDRECORD_FILE_NAME + ".db")

    tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME),
                                        MINDRECORD_FILE_NAME, feature_dict)
    tfrecord_transformer.transform()

    assert os.path.exists(MINDRECORD_FILE_NAME)
    assert os.path.exists(MINDRECORD_FILE_NAME + ".db")

    fr_mindrecord = FileReader(MINDRECORD_FILE_NAME)
    verify_data(tfrecord_transformer, fr_mindrecord)

    os.remove(MINDRECORD_FILE_NAME)
    os.remove(MINDRECORD_FILE_NAME + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME))

def test_tfrecord_to_mindrecord_scalar_with_2_exception():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    generate_tfrecord()
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME))

    feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([2], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([1], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                    "float_list": tf.io.FixedLenFeature([7], tf.float32),
                    }

    if os.path.exists(MINDRECORD_FILE_NAME):
        os.remove(MINDRECORD_FILE_NAME)
    if os.path.exists(MINDRECORD_FILE_NAME + ".db"):
        os.remove(MINDRECORD_FILE_NAME + ".db")

    tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME),
                                        MINDRECORD_FILE_NAME, feature_dict, ["image_bytes"])
    with pytest.raises(ValueError):
        tfrecord_transformer.transform()

    if os.path.exists(MINDRECORD_FILE_NAME):
        os.remove(MINDRECORD_FILE_NAME)
    if os.path.exists(MINDRECORD_FILE_NAME + ".db"):
        os.remove(MINDRECORD_FILE_NAME + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME))

def test_tfrecord_to_mindrecord_scalar_string_with_1_exception():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    generate_tfrecord()
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME))

    feature_dict = {"file_name": tf.io.FixedLenFeature([1], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([1], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([1], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                    "float_list": tf.io.FixedLenFeature([7], tf.float32),
                    }

    if os.path.exists(MINDRECORD_FILE_NAME):
        os.remove(MINDRECORD_FILE_NAME)
    if os.path.exists(MINDRECORD_FILE_NAME + ".db"):
        os.remove(MINDRECORD_FILE_NAME + ".db")

    with pytest.raises(ValueError):
        tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME),
                                            MINDRECORD_FILE_NAME, feature_dict, ["image_bytes"])
        tfrecord_transformer.transform()

    if os.path.exists(MINDRECORD_FILE_NAME):
        os.remove(MINDRECORD_FILE_NAME)
    if os.path.exists(MINDRECORD_FILE_NAME + ".db"):
        os.remove(MINDRECORD_FILE_NAME + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME))

def test_tfrecord_to_mindrecord_scalar_bytes_with_10_exception():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    generate_tfrecord()
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME))

    feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([10], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([1], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([1], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                    "float_list": tf.io.FixedLenFeature([7], tf.float32),
                    }

    if os.path.exists(MINDRECORD_FILE_NAME):
        os.remove(MINDRECORD_FILE_NAME)
    if os.path.exists(MINDRECORD_FILE_NAME + ".db"):
        os.remove(MINDRECORD_FILE_NAME + ".db")

    with pytest.raises(ValueError):
        tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME),
                                            MINDRECORD_FILE_NAME, feature_dict, ["image_bytes"])
        tfrecord_transformer.transform()

    if os.path.exists(MINDRECORD_FILE_NAME):
        os.remove(MINDRECORD_FILE_NAME)
    if os.path.exists(MINDRECORD_FILE_NAME + ".db"):
        os.remove(MINDRECORD_FILE_NAME + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, TFRECORD_FILE_NAME))
