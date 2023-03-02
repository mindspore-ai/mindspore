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
from ..shardutils import check_filename, ExceptionThread

__all__ = ['TFRecordToMR']

SupportedTensorFlowVersion = '1.13.0-rc1'


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
    A class to transform from TFRecord to MindRecord.

    Note:
        For details about Examples, please refer to `Converting TFRecord Dataset <https://
        www.mindspore.cn/tutorials/en/master/advanced/dataset/record.html#converting-tfrecord-dataset>`_ .

    Args:
        source (str): TFRecord file to be transformed.
        destination (str): MindRecord file path to transform into, ensure that the directory is created in advance and
            no file with the same name exists in the directory.
        feature_dict (dict[str, FixedLenFeature]): Dictionary that states the feature type, and
            `FixedLenFeature <https://www.tensorflow.org/api_docs/python/tf/io/FixedLenFeature>`_ is supported.
        bytes_fields (list[str], optional): The bytes fields which are in `feature_dict` and can be images bytes.
            Default: None, means that there is no byte dtype field such as image.

    Raises:
        ValueError: If parameter is invalid.
        Exception: when tensorflow module is not found or version is not correct.
    """

    def __init__(self, source, destination, feature_dict, bytes_fields=None):
        try:
            self.tf = import_module("tensorflow")  # just used to convert tfrecord to mindrecord
        except ModuleNotFoundError:
            raise Exception("Module tensorflow is not found, please use pip install it.")

        if self.tf.__version__ < SupportedTensorFlowVersion:
            raise Exception("Module tensorflow version must be greater or equal {}.".format(SupportedTensorFlowVersion))

        self._check_input(source, destination, feature_dict)
        self.source = source
        self.destination = destination
        self.feature_dict = feature_dict

        bytes_fields_list = []
        if bytes_fields is not None:
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

                if self.feature_dict[item].dtype != self.tf.string:
                    raise ValueError("Parameter bytes_field: {} should be tf.string in feature_dict.".format(item))

                casted_bytes_field = _cast_name(item)
                bytes_fields_list.append(casted_bytes_field)

        self.bytes_fields_list = bytes_fields_list
        self.scalar_set = set()
        self.list_set = set()
        self.mindrecord_schema = self._parse_mindrecord_schema_from_feature_dict()

    def _check_input(self, source, destination, feature_dict):
        """Validation check for inputs of init method"""
        if not isinstance(source, str):
            raise ValueError("Parameter source must be string.")
        check_filename(source, "source")

        if not isinstance(destination, str):
            raise ValueError("Parameter destination must be string.")
        check_filename(destination, "destination")

        if feature_dict is None or not isinstance(feature_dict, dict):
            raise ValueError("Parameter feature_dict is None or not dict.")

        for _, val in feature_dict.items():
            if not isinstance(val, self.tf.io.FixedLenFeature):
                raise ValueError("Parameter feature_dict: {} only support FixedLenFeature.".format(feature_dict))

    def _parse_mindrecord_schema_from_feature_dict(self):
        """get mindrecord schema from feature dict"""
        mindrecord_schema = {}
        for key, val in self.feature_dict.items():
            if not val.shape:
                self.scalar_set.add(_cast_name(key))
                if _cast_name(key) in self.bytes_fields_list:
                    mindrecord_schema[_cast_name(key)] = {"type": "bytes"}
                else:
                    mindrecord_schema[_cast_name(key)] = {"type": self._cast_type(val.dtype)}
            else:
                if len(val.shape) != 1:
                    raise ValueError("Parameter len(feature_dict[{}].shape) should be 1.")
                if not isinstance(val.shape[0], int):
                    raise ValueError("Invalid parameter feature_dict, value and type mismatch, " +
                                     "please check item of feature_dict[\"{}\"].".format(key))
                if val.shape[0] < 1:
                    raise ValueError("Parameter feature_dict[{}].shape[0] should > 0".format(key))
                if val.dtype == self.tf.string:
                    raise ValueError("Parameter feature_dict[{}].dtype is tf.string which shape[0] "
                                     "is not None. It is not supported.".format(key))
                self.list_set.add(_cast_name(key))
                mindrecord_schema[_cast_name(key)] = {"type": self._cast_type(val.dtype), "shape": [val.shape[0]]}
        return mindrecord_schema

    def _parse_record(self, example):
        """Returns features for a single example"""
        features = self.tf.io.parse_single_example(example, features=self.feature_dict)
        return features

    def _get_data_when_scalar_field(self, ms_dict, cast_key, key, val):
        """put data in ms_dict when field type is string"""
        if isinstance(val.numpy(), (np.ndarray, list)):
            raise ValueError("The response key: {}, value: {} from TFRecord should be a scalar.".format(key, val))
        if self.feature_dict[key].dtype == self.tf.string:
            if cast_key in self.bytes_fields_list:
                ms_dict[cast_key] = val.numpy()
            else:
                ms_dict[cast_key] = str(val.numpy(), encoding="utf-8")
        elif self._cast_type(self.feature_dict[key].dtype).startswith("int"):
            ms_dict[cast_key] = int(val.numpy())
        else:
            ms_dict[cast_key] = float(val.numpy())

    def _get_data_when_scalar_field_oldversion(self, ms_dict, cast_key, key, val):
        """
        put data in ms_dict when field type is string
        However, we have to make change due to the different structure of old version
        """
        if isinstance(val, (bytes, str)):
            if isinstance(val, (np.ndarray, list)):
                raise ValueError("The response key: {}, value: {} from TFRecord should be a scalar.".format(key, val))
            if self.feature_dict[key].dtype == self.tf.string:
                if cast_key in self.bytes_fields_list:
                    ms_dict[cast_key] = val
                else:
                    ms_dict[cast_key] = val.decode("utf-8")
            else:
                ms_dict[cast_key] = val
        else:
            if self._cast_type(self.feature_dict[key].dtype).startswith("int"):
                ms_dict[cast_key] = int(val)
            else:
                ms_dict[cast_key] = float(val)

    def tfrecord_iterator_oldversion(self):
        """
        Yield a dict with key to be fields in schema, and value to be data.
        This function is for old version tensorflow whose version number < 2.1.0.

        Returns:
            dict, data dictionary whose keys are the same as columns.
        """
        logger.warning("This interface will be deleted or invisible in the future.")

        dataset = self.tf.data.TFRecordDataset(self.source)
        dataset = dataset.map(self._parse_record)
        iterator = dataset.make_one_shot_iterator()
        with self.tf.Session() as sess:
            while True:
                try:
                    ms_dict = {}
                    sample = iterator.get_next()
                    sample = sess.run(sample)
                    for key, val in sample.items():
                        cast_key = _cast_name(key)
                        if cast_key in self.scalar_set:
                            self._get_data_when_scalar_field_oldversion(ms_dict, cast_key, key, val)
                        else:
                            if not isinstance(val, np.ndarray) and not isinstance(val, list):
                                raise ValueError("The response key: {}, value: {} from "
                                                 "TFRecord should be a ndarray or "
                                                 "list.".format(key, val))
                            # list set
                            ms_dict[cast_key] = \
                                np.asarray(val, _cast_string_type_to_np_type(self.mindrecord_schema[cast_key]["type"]))
                    yield ms_dict
                except self.tf.errors.OutOfRangeError:
                    break
                except self.tf.errors.InvalidArgumentError:
                    raise ValueError("TFRecord feature_dict parameter error.")

    def _get_data_from_tfrecord_sample(self, iterator):
        """convert tfrecord sample to mindrecord sample"""
        ms_dict = {}
        sample = iterator.get_next()
        for key, val in sample.items():
            cast_key = _cast_name(key)
            if cast_key in self.scalar_set:
                self._get_data_when_scalar_field(ms_dict, cast_key, key, val)
            else:
                if not isinstance(val.numpy(), np.ndarray) and not isinstance(val.numpy(), list):
                    raise ValueError("The response key: {}, value: {} from TFRecord should be a ndarray or list."
                                     .format(key, val))
                # list set
                ms_dict[cast_key] = \
                    np.asarray(val, _cast_string_type_to_np_type(self.mindrecord_schema[cast_key]["type"]))
        return ms_dict


    def tfrecord_iterator(self):
        """
        Yield a dictionary whose keys are fields in schema.

        Returns:
            dict, data dictionary whose keys are the same as columns.
        """
        logger.warning("This interface will be deleted or invisible in the future.")

        dataset = self.tf.data.TFRecordDataset(self.source)
        dataset = dataset.map(self._parse_record)
        iterator = dataset.__iter__()
        while True:
            try:
                yield self._get_data_from_tfrecord_sample(iterator)
            except self.tf.errors.OutOfRangeError:
                break
            except self.tf.errors.InvalidArgumentError:
                raise ValueError("TFRecord feature_dict parameter error.")

    def run(self):
        """
        Execute transformation from TFRecord to MindRecord.

        Returns:
            MSRStatus, SUCCESS or FAILED.
        """
        writer = FileWriter(self.destination)
        logger.info("Transformed MindRecord schema is: {}, TFRecord feature dict is: {}"
                    .format(self.mindrecord_schema, self.feature_dict))

        writer.add_schema(self.mindrecord_schema, "TFRecord to MindRecord")
        if self.tf.__version__ < '2.0.0':
            tf_iter = self.tfrecord_iterator_oldversion()
        else:
            tf_iter = self.tfrecord_iterator()
        batch_size = 256
        transform_count = 0
        while True:
            data_list = []
            try:
                for _ in range(batch_size):
                    data_list.append(tf_iter.__next__())
                    transform_count += 1

                writer.write_raw_data(data_list, True)
                logger.info("Transformed {} records...".format(transform_count))
            except StopIteration:
                if data_list:
                    writer.write_raw_data(data_list, True)
                    logger.info("Transformed {} records...".format(transform_count))
                break
        return writer.commit()

    def transform(self):
        """
        Encapsulate the :func:`mindspore.mindrecord.TFRecordToMR.run` function to exit normally.

        Returns:
            MSRStatus, SUCCESS or FAILED.
        """

        t = ExceptionThread(target=self.run)
        t.daemon = True
        t.start()
        t.join()
        if t.exitcode != 0:
            raise t.exception
        return t.res

    def _cast_type(self, value):
        """
        Cast complex data type to basic datatype for MindRecord to recognize.

        Args:
            value: the TFRecord data type

        Returns:
            str, which is MindRecord field type.
        """
        tf_type_to_mr_type = {self.tf.string: "string",
                              self.tf.int8: "int32",
                              self.tf.int16: "int32",
                              self.tf.int32: "int32",
                              self.tf.int64: "int64",
                              self.tf.uint8: "int32",
                              self.tf.uint16: "int32",
                              self.tf.uint32: "int64",
                              self.tf.uint64: "int64",
                              self.tf.float16: "float32",
                              self.tf.float32: "float32",
                              self.tf.float64: "float64",
                              self.tf.double: "float64",
                              self.tf.bool: "int32"}

        if value in tf_type_to_mr_type:
            return tf_type_to_mr_type[value]

        raise ValueError("Type " + value + " is not supported in MindRecord.")
