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
This module is to write data into mindrecord.
"""
import os
import sys
import threading
import traceback

from inspect import signature
from functools import wraps

import numpy as np
import mindspore._c_mindrecord as ms
from .common.exceptions import ParamValueError, MRMUnsupportedSchemaError

SUCCESS = ms.MSRStatus.SUCCESS
FAILED = ms.MSRStatus.FAILED
DATASET_NLP = ms.ShardType.NLP
DATASET_CV = ms.ShardType.CV

MIN_HEADER_SIZE = ms.MIN_HEADER_SIZE
MAX_HEADER_SIZE = ms.MAX_HEADER_SIZE
MIN_PAGE_SIZE = ms.MIN_PAGE_SIZE
MAX_PAGE_SIZE = ms.MAX_PAGE_SIZE
MIN_SHARD_COUNT = ms.MIN_SHARD_COUNT
MAX_SHARD_COUNT = ms.MAX_SHARD_COUNT
MIN_CONSUMER_COUNT = ms.MIN_CONSUMER_COUNT
MAX_CONSUMER_COUNT = ms.get_max_thread_num

VALUE_TYPE_MAP = {"int": ["int32", "int64"], "float": ["float32", "float64"], "str": "string", "bytes": "bytes",
                  "int32": "int32", "int64": "int64", "float32": "float32", "float64": "float64",
                  "ndarray": ["int32", "int64", "float32", "float64"]}

VALID_ATTRIBUTES = ["int32", "int64", "float32", "float64", "string", "bytes"]
VALID_ARRAY_ATTRIBUTES = ["int32", "int64", "float32", "float64"]


class ExceptionThread(threading.Thread):
    """ class to pass exception"""
    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.res = SUCCESS
        self.exitcode = 0
        self.exception = None
        self.exc_traceback = ''

    def run(self):
        try:
            if self._target:
                self.res = self._target(*self._args, **self._kwargs)
        except Exception as e:  # pylint: disable=W0703
            self.exitcode = 1
            self.exception = e
            self.exc_traceback = ''.join(traceback.format_exception(*sys.exc_info()))


def check_filename(path, arg_name=""):
    """
    check the filename in the path.

    Args:
        path (str): the path.

    Raises:
        ParamValueError: If path is not string.
        FileNameError: If path contains invalid character.

    Returns:
        Bool, whether filename is valid.
    """
    if arg_name == "":
        arg_name = "File path"
    else:
        arg_name = "'{}'".format(arg_name)
    if not path:
        raise ParamValueError('{} is not allowed None or empty!'.format(arg_name))
    if not isinstance(path, str):
        raise ParamValueError("File path: {} is not string.".format(path))
    if path.endswith("/"):
        raise ParamValueError("File path can not end with '/'")
    file_name = os.path.basename(path)

    # '#', ':', '|', ' ', '}', '"', '+', '!', ']', '[', '\\', '`',
    # '&', '.', '/', '@', "'", '^', ',', '_', '<', ';', '~', '>',
    # '*', '(', '%', ')', '-', '=', '{', '?', '$'
    forbidden_symbols = set(r'\/:*?"<>|`&\';')

    if set(file_name) & forbidden_symbols:
        raise ParamValueError(r"File name should not contains \/:*?\"<>|`&;\'")

    if file_name.startswith(' ') or file_name.endswith(' '):
        raise ParamValueError("File name should not start/end with space.")

    return True


def check_parameter(func):
    """
    decorator for parameter check
    """
    sig = signature(func)

    @wraps(func)
    def wrapper(*args, **kw):
        bound = sig.bind(*args, **kw)
        for name, value in bound.arguments.items():
            if name == 'file_name':
                if isinstance(value, list):
                    for f in value:
                        check_filename(f)
                else:
                    check_filename(value)
            if name == 'num_consumer':
                if value is None:
                    raise ParamValueError("Parameter num_consumer is None.")
                if isinstance(value, int):
                    if value < MIN_CONSUMER_COUNT or value > MAX_CONSUMER_COUNT():
                        raise ParamValueError("Parameter num_consumer: {} should between {} and {}."
                                              .format(value, MIN_CONSUMER_COUNT, MAX_CONSUMER_COUNT()))
                else:
                    raise ParamValueError("Parameter num_consumer is not int.")
        return func(*args, **kw)

    return wrapper


def populate_data(raw, blob, columns, blob_fields, schema):
    """
    Reconstruct data form raw and blob data.

    Args:
        raw (Dict): Data contain primitive data like "int32", "int64", "float32", "float64", "string", "bytes".
        blob (Bytes): Data contain bytes and ndarray data.
        columns(List): List of column name which will be populated.
        blob_fields (List): Refer to the field which data stored in blob.
        schema(Dict): Dict of Schema

    Raises:
        MRMUnsupportedSchemaError: If schema is invalid.
    """
    if raw:
        # remove dummy fields
        raw = {k: v for k, v in raw.items() if k in schema}
    else:
        raw = {}
    if not blob_fields:
        return raw

    loaded_columns = []
    if columns:
        for column in columns:
            if column in blob_fields:
                loaded_columns.append(column)
    else:
        loaded_columns = blob_fields

    def _render_raw(field, blob_data):
        data_type = schema[field]['type']
        data_shape = schema[field]['shape'] if 'shape' in schema[field] else []
        if data_shape:
            try:
                raw[field] = np.reshape(np.frombuffer(blob_data, dtype=data_type), data_shape)
            except ValueError:
                raise MRMUnsupportedSchemaError('Shape in schema is illegal.')
        else:
            raw[field] = blob_data

    for i, blob_field in enumerate(loaded_columns):
        _render_raw(blob_field, bytes(blob[i]))
    return raw
