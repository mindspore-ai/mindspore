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


def check_filename(path):
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
    if not path:
        raise ParamValueError('File path is not allowed None or empty!')
    if not isinstance(path, str):
        raise ParamValueError("File path: {} is not string.".format(path))
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
        # remove dummy fileds
        raw = {k: v for k, v in raw.items() if k in schema}
    else:
        raw = {}
    if not blob_fields:
        return raw

    # Get the order preserving sequence of columns in blob
    ordered_columns = []
    if columns:
        for blob_field in blob_fields:
            if blob_field in columns:
                ordered_columns.append(blob_field)
    else:
        ordered_columns = blob_fields

    blob_bytes = bytes(blob)

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

    if len(blob_fields) == 1:
        if len(ordered_columns) == 1:
            _render_raw(blob_fields[0], blob_bytes)
            return raw
        return raw

    def _int_from_bytes(xbytes: bytes) -> int:
        return int.from_bytes(xbytes, 'big')

    def _blob_at_position(pos):
        start = 0
        for _ in range(pos):
            n_bytes = _int_from_bytes(blob_bytes[start : start + 8])
            start += 8 + n_bytes
        n_bytes = _int_from_bytes(blob_bytes[start : start + 8])
        start += 8
        return blob_bytes[start : start + n_bytes]

    for i, blob_field in enumerate(ordered_columns):
        _render_raw(blob_field, _blob_at_position(i))
    return raw
