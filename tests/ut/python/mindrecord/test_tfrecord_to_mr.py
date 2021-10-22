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
from string import punctuation

import numpy as np
import pytest
from mindspore import log as logger
from mindspore.mindrecord import FileReader
from mindspore.mindrecord import TFRecordToMR

SupportedTensorFlowVersion = '1.13.0-rc1'

try:
    tf = import_module("tensorflow")    # just used to convert tfrecord to mindrecord
except ModuleNotFoundError:
    logger.warning("tensorflow module not found.")
    tf = None

TFRECORD_DATA_DIR = "../data/mindrecord/testTFRecordData"
PARTITION_NUM = 1

def cast_name(key):
    """
    Cast schema names which containing special characters to valid names.

    Here special characters means any characters in
    '!"#$%&\'()*+,./:;<=>?@[\\]^`{|}~
    Valid names can only contain a-z, A-Z, and 0-9 and _

    Args:
        key (str): original key that might contains special characters.

    Returns:
        str, casted key that replace the special characters with "_". i.e. if
            key is "a b" then returns "a_b".
    """
    special_symbols = set('{}{}'.format(punctuation, ' '))
    special_symbols.remove('_')
    new_key = ['_' if x in special_symbols else x for x in key]
    casted_key = ''.join(new_key)
    return casted_key

def verify_data(transformer, reader):
    """
    Verify the data by read from mindrecord
    If in 1.x.x version, use old version to receive that iteration
    """

    if tf.__version__ < '2.0.0':
        tf_iter = transformer.tfrecord_iterator_oldversion()
    else:
        tf_iter = transformer.tfrecord_iterator()
    mr_iter = reader.get_next()

    count = 0
    for tf_item, mr_item in zip(tf_iter, mr_iter):
        count = count + 1
        assert len(tf_item) == len(mr_item)
        for key, value in tf_item.items():
            logger.info("key: {}, tfrecord: value: {}, mindrecord: value: {}".format(key, value,
                                                                                     mr_item[cast_name(key)]))
            if isinstance(value, np.ndarray):
                assert (value == mr_item[cast_name(key)]).all()
            else:
                assert value == mr_item[cast_name(key)]
    assert count == 10

def generate_tfrecord(tfrecord_file_name):
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

    writer = tf.io.TFRecordWriter(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

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

def generate_tfrecord_with_special_field_name(tfrecord_file_name):
    def create_int_feature(values):
        if isinstance(values, list):
            feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))  # values: [int, int, int]
        else:
            feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))      # values: int
        return feature

    def create_bytes_feature(values):
        if isinstance(values, bytes):
            feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))      # values: bytes
        else:
            # values: string
            feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(values, encoding='utf-8')]))
        return feature

    writer = tf.io.TFRecordWriter(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

    example_count = 0
    for i in range(10):
        label = i
        image_bytes = bytes(str("aaaabbbbcccc" + str(i)), encoding="utf-8")

        features = collections.OrderedDict()
        features["image/class/label"] = create_int_feature(label)
        features["image/encoded"] = create_bytes_feature(image_bytes)

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

    file_name_ = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    mindrecord_file_name = file_name_ + '.mindrecord'
    tfrecord_file_name = file_name_ + '.tfrecord'
    generate_tfrecord(tfrecord_file_name)

    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

    feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                    "float_list": tf.io.FixedLenFeature([7], tf.float32),
                    }

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name),
                                        mindrecord_file_name, feature_dict, ["image_bytes"])
    tfrecord_transformer.transform()

    assert os.path.exists(mindrecord_file_name)
    assert os.path.exists(mindrecord_file_name + ".db")

    fr_mindrecord = FileReader(mindrecord_file_name)
    verify_data(tfrecord_transformer, fr_mindrecord)

    os.remove(mindrecord_file_name)
    os.remove(mindrecord_file_name + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

def test_tfrecord_to_mindrecord_scalar_with_1():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    file_name_ = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    mindrecord_file_name = file_name_ + '.mindrecord'
    tfrecord_file_name = file_name_ + '.tfrecord'
    generate_tfrecord(tfrecord_file_name)
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

    feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([1], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([1], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                    "float_list": tf.io.FixedLenFeature([7], tf.float32),
                    }

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name),
                                        mindrecord_file_name, feature_dict, ["image_bytes"])
    tfrecord_transformer.transform()

    assert os.path.exists(mindrecord_file_name)
    assert os.path.exists(mindrecord_file_name + ".db")

    fr_mindrecord = FileReader(mindrecord_file_name)
    verify_data(tfrecord_transformer, fr_mindrecord)

    os.remove(mindrecord_file_name)
    os.remove(mindrecord_file_name + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

def test_tfrecord_to_mindrecord_scalar_with_1_list_small_len_exception():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    file_name_ = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    mindrecord_file_name = file_name_ + '.mindrecord'
    tfrecord_file_name = file_name_ + '.tfrecord'
    generate_tfrecord(tfrecord_file_name)
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

    feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([1], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([1], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                    "float_list": tf.io.FixedLenFeature([2], tf.float32),
                    }

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    with pytest.raises(ValueError):
        tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name),
                                            mindrecord_file_name, feature_dict, ["image_bytes"])
        tfrecord_transformer.transform()

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

def test_tfrecord_to_mindrecord_list_with_diff_type_exception():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    file_name_ = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    mindrecord_file_name = file_name_ + '.mindrecord'
    tfrecord_file_name = file_name_ + '.tfrecord'
    generate_tfrecord(tfrecord_file_name)
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

    feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([1], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([1], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.float32),
                    "float_list": tf.io.FixedLenFeature([7], tf.float32),
                    }

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    with pytest.raises(ValueError):
        tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name),
                                            mindrecord_file_name, feature_dict, ["image_bytes"])
        tfrecord_transformer.transform()

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

def test_tfrecord_to_mindrecord_list_without_bytes_type():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    file_name_ = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    mindrecord_file_name = file_name_ + '.mindrecord'
    tfrecord_file_name = file_name_ + '.tfrecord'
    generate_tfrecord(tfrecord_file_name)
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

    feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([1], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([1], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                    "float_list": tf.io.FixedLenFeature([7], tf.float32),
                    }

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name),
                                        mindrecord_file_name, feature_dict)
    tfrecord_transformer.transform()

    assert os.path.exists(mindrecord_file_name)
    assert os.path.exists(mindrecord_file_name + ".db")

    fr_mindrecord = FileReader(mindrecord_file_name)
    verify_data(tfrecord_transformer, fr_mindrecord)

    os.remove(mindrecord_file_name)
    os.remove(mindrecord_file_name + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

def test_tfrecord_to_mindrecord_scalar_with_2_exception():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    file_name_ = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    mindrecord_file_name = file_name_ + '.mindrecord'
    tfrecord_file_name = file_name_ + '.tfrecord'
    generate_tfrecord(tfrecord_file_name)
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

    feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([2], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([1], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                    "float_list": tf.io.FixedLenFeature([7], tf.float32),
                    }

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name),
                                        mindrecord_file_name, feature_dict, ["image_bytes"])
    with pytest.raises(ValueError):
        tfrecord_transformer.transform()

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

def test_tfrecord_to_mindrecord_scalar_string_with_1_exception():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    file_name_ = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    mindrecord_file_name = file_name_ + '.mindrecord'
    tfrecord_file_name = file_name_ + '.tfrecord'
    generate_tfrecord(tfrecord_file_name)
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

    feature_dict = {"file_name": tf.io.FixedLenFeature([1], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([1], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([1], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                    "float_list": tf.io.FixedLenFeature([7], tf.float32),
                    }

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    with pytest.raises(ValueError):
        tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name),
                                            mindrecord_file_name, feature_dict, ["image_bytes"])
        tfrecord_transformer.transform()

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

def test_tfrecord_to_mindrecord_scalar_bytes_with_10_exception():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    file_name_ = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    mindrecord_file_name = file_name_ + '.mindrecord'
    tfrecord_file_name = file_name_ + '.tfrecord'
    generate_tfrecord(tfrecord_file_name)
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

    feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([10], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([1], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([1], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                    "float_list": tf.io.FixedLenFeature([7], tf.float32),
                    }

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    with pytest.raises(ValueError):
        tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name),
                                            mindrecord_file_name, feature_dict, ["image_bytes"])
        tfrecord_transformer.transform()

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

def test_tfrecord_to_mindrecord_exception_bytes_fields_is_not_string_type():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    file_name_ = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    mindrecord_file_name = file_name_ + '.mindrecord'
    tfrecord_file_name = file_name_ + '.tfrecord'
    generate_tfrecord(tfrecord_file_name)
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

    feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                    "float_list": tf.io.FixedLenFeature([7], tf.float32),
                    }

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    with pytest.raises(ValueError):
        tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name),
                                            mindrecord_file_name, feature_dict, ["int64_list"])
        tfrecord_transformer.transform()

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

def test_tfrecord_to_mindrecord_exception_bytes_fields_is_not_list():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    file_name_ = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    mindrecord_file_name = file_name_ + '.mindrecord'
    tfrecord_file_name = file_name_ + '.tfrecord'
    generate_tfrecord(tfrecord_file_name)
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

    feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                    "float_list": tf.io.FixedLenFeature([7], tf.float32),
                    }

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    with pytest.raises(ValueError):
        tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name),
                                            mindrecord_file_name, feature_dict, "")
        tfrecord_transformer.transform()

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

def test_tfrecord_to_mindrecord_with_special_field_name():
    """test transform tfrecord to mindrecord."""
    if not tf or tf.__version__ < SupportedTensorFlowVersion:
        # skip the test
        logger.warning("Module tensorflow is not found or version wrong, \
            please use pip install it / reinstall version >= {}.".format(SupportedTensorFlowVersion))
        return

    file_name_ = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    mindrecord_file_name = file_name_ + '.mindrecord'
    tfrecord_file_name = file_name_ + '.tfrecord'
    generate_tfrecord_with_special_field_name(tfrecord_file_name)
    assert os.path.exists(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))

    feature_dict = {"image/class/label": tf.io.FixedLenFeature([], tf.int64),
                    "image/encoded": tf.io.FixedLenFeature([], tf.string),
                    }

    if os.path.exists(mindrecord_file_name):
        os.remove(mindrecord_file_name)
    if os.path.exists(mindrecord_file_name + ".db"):
        os.remove(mindrecord_file_name + ".db")

    tfrecord_transformer = TFRecordToMR(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name),
                                        mindrecord_file_name, feature_dict, ["image/encoded"])
    tfrecord_transformer.transform()

    assert os.path.exists(mindrecord_file_name)
    assert os.path.exists(mindrecord_file_name + ".db")

    fr_mindrecord = FileReader(mindrecord_file_name)
    verify_data(tfrecord_transformer, fr_mindrecord)

    os.remove(mindrecord_file_name)
    os.remove(mindrecord_file_name + ".db")

    os.remove(os.path.join(TFRECORD_DATA_DIR, tfrecord_file_name))
