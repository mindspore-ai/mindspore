# Copyright 2020 Huawei Technologies Co., Ltd
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
"""
TFRecord convert tool for MindRecord
"""

from importlib import import_module
from string import punctuation
import numpy as np

from mindspore import log as logger
from ..filewriter import FileWriter
from ..shardutils import check_filename

try:
    tf = import_module("tensorflow")    # just used to convert tfrecord to mindrecord
except ModuleNotFoundError:
    tf = None

__all__ = ['TFRecordToMR']

SupportedTensorFlowVersion = '2.1.0'

def _cast_type(value):
    """
    Cast complex data type to basic datatype for MindRecord to recognize.

    Args:
        value: the TFRecord data type

    Returns:
        str, which is MindRecord field type.
    """
    tf_type_to_mr_type = {tf.string: "string",
                          tf.int8: "int32",
                          tf.int16: "int32",
                          tf.int32: "int32",
                          tf.int64: "int64",
                          tf.uint8: "int32",
                          tf.uint16: "int32",
                          tf.uint32: "int64",
                          tf.uint64: "int64",
                          tf.float16: "float32",
                          tf.float32: "float32",
                          tf.float64: "float64",
                          tf.double: "float64",
                          tf.bool: "int32"}
    unsupport_tf_type_to_mr_type = {tf.complex64: "None",
                                    tf.complex128: "None"}

    if value in tf_type_to_mr_type:
        return tf_type_to_mr_type[value]

    raise ValueError("Type " + value + " is not supported in MindRecord.")

def _cast_string_type_to_np_type(value):
    """Cast string type like: int32/int64/float32/float64 to np.int32/np.int64/np.float32/np.float64"""
    string_type_to_np_type = {"int32": np.int32,
                              "int64": np.int64,
                              "float32": np.float32,
                              "float64": np.float64}

    if value in string_type_to_np_type:
        return string_type_to_np_type[value]

    raise ValueError("Type " + value + " is not supported cast to numpy type in MindRecord.")

def _cast_name(key):
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

class TFRecordToMR:
    """
    Class is for tranformation from TFRecord to MindRecord.

    Args:
        source (str): the TFRecord file to be transformed.
        destination (str): the MindRecord file path to tranform into.
        feature_dict (dict): a dictionary than states the feature type, i.e.
            feature_dict = {"xxxx": tf.io.FixedLenFeature([], tf.string), \
                            "yyyy": tf.io.FixedLenFeature([], tf.int64)}

            **Follow case which uses VarLenFeature not support**

            feature_dict = {"context": {"xxxx": tf.io.FixedLenFeature([], tf.string), \
                                        "yyyy": tf.io.VarLenFeature(tf.int64)}, \
                            "sequence": {"zzzz": tf.io.FixedLenSequenceFeature([], tf.float32)}}
        bytes_fields (list): the bytes fields which are in feature_dict.

    Raises:
        ValueError: If parameter is invalid.
        Exception: when tensorflow module not found or version is not correct.
    """
    def __init__(self, source, destination, feature_dict, bytes_fields=None):
        if not tf:
            raise Exception("Module tensorflow is not found, please use pip install it.")

        if tf.__version__ < SupportedTensorFlowVersion:
            raise Exception("Module tensorflow version must be greater or equal {}.".format(SupportedTensorFlowVersion))

        if not isinstance(source, str):
            raise ValueError("Parameter source must be string.")
        check_filename(source)

        if not isinstance(destination, str):
            raise ValueError("Parameter destination must be string.")
        check_filename(destination)

        self.source = source
        self.destination = destination

        if feature_dict is None or not isinstance(feature_dict, dict):
            raise ValueError("Parameter feature_dict is None or not dict.")

        for key, val in feature_dict.items():
            if not isinstance(val, tf.io.FixedLenFeature):
                raise ValueError("Parameter feature_dict: {} only support FixedLenFeature.".format(feature_dict))

        self.feature_dict = feature_dict

        bytes_fields_list = []
        if bytes_fields:
            if not isinstance(bytes_fields, list):
                raise ValueError("Parameter bytes_fields: {} must be list(str).".format(bytes_fields))
            for item in bytes_fields:
                if not isinstance(item, str):
                    raise ValueError("Parameter bytes_fields's item: {} is not str.".format(item))

                if item not in self.feature_dict:
                    raise ValueError("Parameter bytes_fields's item: {} is not in feature_dict: {}."
                                     .format(item, self.feature_dict))

                if not isinstance(self.feature_dict[item].shape, list):
                    raise ValueError("Parameter feature_dict[{}].shape should be a list.".format(item))

                casted_bytes_field = _cast_name(item)
                bytes_fields_list.append(casted_bytes_field)

        self.bytes_fields_list = bytes_fields_list
        self.scalar_set = set()
        self.list_set = set()

        mindrecord_schema = {}
        for key, val in self.feature_dict.items():
            if not val.shape:
                self.scalar_set.add(_cast_name(key))
                if key in self.bytes_fields_list:
                    mindrecord_schema[_cast_name(key)] = {"type": "bytes"}
                else:
                    mindrecord_schema[_cast_name(key)] = {"type": _cast_type(val.dtype)}
            else:
                if len(val.shape) != 1:
                    raise ValueError("Parameter len(feature_dict[{}].shape) should be 1.")
                if val.shape[0] < 1:
                    raise ValueError("Parameter feature_dict[{}].shape[0] should > 0".format(key))
                if val.dtype == tf.string:
                    raise ValueError("Parameter feautre_dict[{}].dtype is tf.string which shape[0] \
                        is not None. It is not supported.".format(key))
                self.list_set.add(_cast_name(key))
                mindrecord_schema[_cast_name(key)] = {"type": _cast_type(val.dtype), "shape": [val.shape[0]]}
        self.mindrecord_schema = mindrecord_schema

    def _parse_record(self, example):
        """Returns features for a single example"""
        features = tf.io.parse_single_example(example, features=self.feature_dict)
        return features

    def _get_data_when_scalar_field(self, ms_dict, cast_key, key, val):
        """put data in ms_dict when field type is string"""
        if isinstance(val.numpy(), (np.ndarray, list)):
            raise ValueError("The response key: {}, value: {} from TFRecord should be a scalar.".format(key, val))
        if self.feature_dict[key].dtype == tf.string:
            if cast_key in self.bytes_fields_list:
                ms_dict[cast_key] = val.numpy()
            else:
                ms_dict[cast_key] = str(val.numpy(), encoding="utf-8")
        elif _cast_type(self.feature_dict[key].dtype).startswith("int"):
            ms_dict[cast_key] = int(val.numpy())
        else:
            ms_dict[cast_key] = float(val.numpy())

    def tfrecord_iterator(self):
        """Yield a dict with key to be fields in schema, and value to be data."""
        dataset = tf.data.TFRecordDataset(self.source)
        dataset = dataset.map(self._parse_record)
        iterator = dataset.__iter__()
        index_id = 0
        try:
            for features in iterator:
                ms_dict = {}
                index_id = index_id + 1
                for key, val in features.items():
                    cast_key = _cast_name(key)
                    if key in self.scalar_set:
                        self._get_data_when_scalar_field(ms_dict, cast_key, key, val)
                    else:
                        if not isinstance(val.numpy(), np.ndarray) and not isinstance(val.numpy(), list):
                            raise ValueError("he response key: {}, value: {} from TFRecord should be a ndarray or list."
                                             .format(key, val))
                        # list set
                        ms_dict[cast_key] = \
                            np.asarray(val, _cast_string_type_to_np_type(self.mindrecord_schema[cast_key]["type"]))
                yield ms_dict
        except tf.errors.InvalidArgumentError:
            raise ValueError("TFRecord feature_dict parameter error.")

    def transform(self):
        """
        Executes transform from TFRecord to MindRecord.

        Returns:
            SUCCESS/FAILED, whether successfuly written into MindRecord.
        """
        writer = FileWriter(self.destination)
        logger.info("Transformed MindRecord schema is: {}, TFRecord feature dict is: {}"
                    .format(self.mindrecord_schema, self.feature_dict))

        writer.add_schema(self.mindrecord_schema, "TFRecord to MindRecord")

        tf_iter = self.tfrecord_iterator()
        batch_size = 256

        transform_count = 0
        while True:
            data_list = []
            try:
                for _ in range(batch_size):
                    data_list.append(tf_iter.__next__())
                    transform_count += 1

                writer.write_raw_data(data_list)
                logger.info("Transformed {} records...".format(transform_count))
            except StopIteration:
                if data_list:
                    writer.write_raw_data(data_list)
                    logger.info("Transformed {} records...".format(transform_count))
                break
        return writer.commit()
