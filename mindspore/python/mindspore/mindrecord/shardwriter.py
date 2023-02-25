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
import numpy as np
import mindspore._c_mindrecord as ms
from mindspore import log as logger
from .common.exceptions import MRMOpenError, MRMOpenForAppendError, MRMInvalidHeaderSizeError, \
    MRMInvalidPageSizeError, MRMSetHeaderError, MRMWriteDatasetError, MRMCommitError

__all__ = ['ShardWriter']


class ShardWriter:
    """
    Wrapper class which is represent shardWrite class in c++ module.

    The class would write MindRecord File series.
    """

    def __init__(self):
        self._writer = ms.ShardWriter()
        self._header = None
        self._is_open = False

    @property
    def is_open(self):
        """getter function"""
        return self._is_open

    @staticmethod
    def convert_np_types(val):
        """convert numpy type to python primitive type"""
        if isinstance(val, (np.int32, np.int64, np.float32, np.float64)):
            return val.item()
        return val

    def open(self, paths, override):
        """
        Open a new MindRecord File and prepare to write raw data.

        Args:
             paths (list[str]): List of file path.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMOpenError: If failed to open MindRecord File.
        """
        ret = self._writer.open(paths, False, override)
        if ret != ms.MSRStatus.SUCCESS:
            logger.critical("Failed to open paths")
            raise MRMOpenError
        self._is_open = True
        return ret

    def open_for_append(self, path):
        """
        Open a existed MindRecord File and prepare to append raw data.

        Args:
            path (str): String of file path.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMOpenForAppendError: If failed to append MindRecord File.
        """
        ret = self._writer.open_for_append(path)
        if ret != ms.MSRStatus.SUCCESS:
            logger.critical("Failed to open path to append.")
            raise MRMOpenForAppendError
        self._is_open = True
        return ret

    def set_header_size(self, header_size):
        """
        Set the size of header.

        Args:
           header_size (int): Size of header, between 16KB and 128MB.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMInvalidHeaderSizeError: If failed to set header size.
        """
        ret = self._writer.set_header_size(header_size)
        if ret != ms.MSRStatus.SUCCESS:
            logger.critical("Failed to set header size.")
            raise MRMInvalidHeaderSizeError
        return ret

    def set_page_size(self, page_size):
        """
        Set the size of page.

        Args:
           page_size (int): Size of page, between 16KB and 128MB.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMInvalidPageSizeError: If failed to set page size.
        """
        ret = self._writer.set_page_size(page_size)
        if ret != ms.MSRStatus.SUCCESS:
            logger.critical("Failed to set page size.")
            raise MRMInvalidPageSizeError
        return ret

    def set_shard_header(self, shard_header):
        """
        Set header which contains schema and index before write raw data.

        Args:
           shard_header (ShardHeader): Object of ShardHeader.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMSetHeaderError: If failed to set header.
        """
        self._header = shard_header
        ret = self._writer.set_shard_header(shard_header.header)
        if ret != ms.MSRStatus.SUCCESS:
            logger.critical("Failed to set header.")
            raise MRMSetHeaderError
        return ret

    def get_shard_header(self):
        return self._header

    def write_raw_data(self, data, validate=True, parallel_writer=False):
        """
        Write raw data of cv dataset.

        Filter data according to schema and separate blob data from raw data.
        Support data verify according to schema and remove the invalid data.

        Args:
           data (list[dict]): List of raw data.
           validate (bool, optional): verify data according schema if it equals to True.
           parallel_writer (bool, optional): Load data parallel if it equals to True.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMWriteCVError: If failed to write cv type dataset.
        """
        blob_data = []
        raw_data = []
        # slice data to blob data and raw data
        for item in data:
            row_blob = self._merge_blob({field: item[field] for field in self._header.blob_fields})
            if row_blob:
                blob_data.append(row_blob)
            # filter raw data according to schema
            row_raw = {field: self.convert_np_types(item[field])
                       for field in self._header.schema.keys() - self._header.blob_fields if field in item}
            if row_raw:
                raw_data.append(row_raw)
        raw_data = {0: raw_data} if raw_data else {}
        ret = self._writer.write_raw_data(raw_data, blob_data, validate, parallel_writer)
        if ret != ms.MSRStatus.SUCCESS:
            logger.critical("Failed to write dataset.")
            raise MRMWriteDatasetError
        return ret

    def commit(self):
        """
        Flush data to disk.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMCommitError: If failed to flush data to disk.
        """
        ret = self._writer.commit()
        if ret != ms.MSRStatus.SUCCESS:
            logger.critical("Failed to commit.")
            raise MRMCommitError
        return ret

    def _merge_blob(self, blob_data):
        """
        Merge multiple blob data whose type is bytes or ndarray

        Args:
           blob_data (dict): Dict of blob data

        Returns:
            bytes, merged blob data
        """
        if len(blob_data) == 1:
            values = [v for v in blob_data.values()]
            return bytes(values[0])

        # convert int to bytes
        def int_to_bytes(x: int) -> bytes:
            return x.to_bytes(8, 'big')

        merged = bytes()
        for field, v in blob_data.items():
            # convert ndarray to bytes
            if isinstance(v, np.ndarray):
                v = v.astype(self._header.schema[field]["type"]).tobytes()
            merged += int_to_bytes(len(v))
            merged += v
        return merged
