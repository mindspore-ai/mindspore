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
This module is to read page from mindrecord.
"""
import mindspore._c_mindrecord as ms
from mindspore import log as logger
from .shardutils import populate_data, SUCCESS
from .shardheader import ShardHeader

__all__ = ['ShardSegment']


class ShardSegment:
    """
    Wrapper class which is represent ShardSegment class in c++ module.

    The class would query data from MindRecord File in pagination.

    """
    def __init__(self):
        self._segment = ms.ShardSegment()
        self._header = None
        self._columns = None

    def open(self, file_name, num_consumer=4, columns=None, operator=None):
        """
        Initialize the ShardSegment.

        Args:
            file_name (str, list[str]): File names of MindRecord File.
            num_consumer (int): Number of worker threads which load data in parallel. Default: 4.
            columns (list[str]): List of fields which correspond data would be read.
            operator(int): Reserved parameter for operators. Default: ``None``.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMOpenError: If failed to open MindRecord File.
        """
        self._columns = columns if columns else []
        operator = operator if operator else []
        if isinstance(file_name, list):
            load_dataset = False
        else:
            load_dataset = True
            file_name = [file_name]
        ret = self._segment.open(file_name, load_dataset, num_consumer, self._columns, operator)
        if ret != SUCCESS:
            logger.critical("Failed to open {}.".format(file_name))
            raise MRMOpenError
        self._header = ShardHeader(self._segment.get_header())
        return ret

    def get_category_fields(self):
        """
        Get candidate category fields.

        Returns:
            list[str], by which data could be grouped.

        """
        return self._segment.get_category_fields()

    def set_category_field(self, category_field):
        """Select one category field to use."""
        return self._segment.set_category_field(category_field)

    def read_category_info(self):
        """
        Get the group info by the current category field.

        Returns:
            str, description fo group information.

        """
        return self._segment.read_category_info()

    def read_at_page_by_id(self, category_id, page, num_row):
        """
        Get the data of some page by category id.

        Args:
            category_id (int): Category id, referred to the return of read_category_info.
            page (int): Index of page.
            num_row (int): Number of rows in a page.

        Returns:
            list[dict]

        Raises:
            MRMUnsupportedSchemaError: If schema is invalid.
        """
        data = self._segment.read_at_page_by_id(category_id, page, num_row)
        return [populate_data(raw, blob, self._columns, self._header.blob_fields,
                              self._header.schema) for blob, raw in data]

    def read_at_page_by_name(self, category_name, page, num_row):
        """
        Get the data of some page by category name.

        Args:
            category_name (str): Category name, referred to the return of read_category_info.
            page (int): Index of page.
            num_row (int): Number of rows in a page.

        Returns:
            list[dict]

        Raises:
            MRMUnsupportedSchemaError: If schema is invalid.
        """
        data = self._segment.read_at_page_by_name(category_name, page, num_row)
        return [populate_data(raw, blob, self._columns, self._header.blob_fields,
                              self._header.schema) for blob, raw in data]
