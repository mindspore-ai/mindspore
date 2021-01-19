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
This module is to support reading page from mindrecord.
"""

from mindspore import log as logger
from .shardsegment import ShardSegment
from .shardutils import MIN_CONSUMER_COUNT, MAX_CONSUMER_COUNT, check_filename
from .common.exceptions import ParamValueError, ParamTypeError, MRMDefineCategoryError

__all__ = ['MindPage']


class MindPage:
    """
    Class to read MindRecord File series in pagination.

    Args:
        file_name (str): One of MindRecord File or a file list.
        num_consumer(int, optional): The number of consumer threads which load data to memory (default=4).
            It should not be smaller than 1 or larger than the number of CPUs.

    Raises:
        ParamValueError: If `file_name`, `num_consumer` or columns is invalid.
        MRMInitSegmentError: If failed to initialize ShardSegment.
    """

    def __init__(self, file_name, num_consumer=4):
        if isinstance(file_name, list):
            for f in file_name:
                check_filename(f)
        else:
            check_filename(file_name)

        if num_consumer is not None:
            if isinstance(num_consumer, int):
                if num_consumer < MIN_CONSUMER_COUNT or num_consumer > MAX_CONSUMER_COUNT():
                    raise ParamValueError("Consumer number should between {} and {}."
                                          .format(MIN_CONSUMER_COUNT, MAX_CONSUMER_COUNT()))
            else:
                raise ParamValueError("Consumer number is illegal.")
        else:
            raise ParamValueError("Consumer number is illegal.")

        self._segment = ShardSegment()
        self._segment.open(file_name, num_consumer)
        self._category_field = None
        self._candidate_fields = [field[:field.rfind('_')] for field in self._segment.get_category_fields()]

    @property
    def candidate_fields(self):
        """
        Return candidate category fields.

        Returns:
            list[str], by which data could be grouped.
        """
        return self._candidate_fields

    def get_category_fields(self):
        """
        Return candidate category fields.

        Returns:
            list[str], by which data could be grouped.
        """
        logger.warning("WARN_DEPRECATED: The usage of get_category_fields is deprecated."
                       " Please use candidate_fields")
        return self.candidate_fields

    def set_category_field(self, category_field):
        """
        Set category field for reading.

        Note:
            Should be a candidate category field.

        Args:
            category_field (str): String of category field name.

        Returns:
            MSRStatus, SUCCESS or FAILED.
        """
        logger.warning("WARN_DEPRECATED: The usage of set_category_field is deprecated."
                       " Please use category_field")
        if not category_field or not isinstance(category_field, str):
            raise ParamTypeError('category_fields', 'str')
        if category_field not in self._candidate_fields:
            raise MRMDefineCategoryError("Field '{}' is not a candidate category field.".format(category_field))
        return self._segment.set_category_field(category_field)

    @property
    def category_field(self):
        """
        Getter function for category fields.

        Returns:
            list[str], by which data could be grouped.
        """
        return self._category_field

    @category_field.setter
    def category_field(self, category_field):
        """
        Setter function for category field.

        Returns:
            MSRStatus, SUCCESS or FAILED.
        """
        if not category_field or not isinstance(category_field, str):
            raise ParamTypeError('category_fields', 'str')
        if category_field not in self._candidate_fields:
            raise MRMDefineCategoryError("Field '{}' is not a candidate category field.".format(category_field))
        self._category_field = category_field
        return self._segment.set_category_field(self._category_field)

    def read_category_info(self):
        """
        Return category information when data is grouped by indicated category field.

        Returns:
            str, description of group information.

        Raises:
            MRMReadCategoryInfoError: If failed to read category information.
        """
        return self._segment.read_category_info()

    def read_at_page_by_id(self, category_id, page, num_row):
        """
        Query by category id in pagination.

        Args:
             category_id (int): Category id, referred to the return of `read_category_info`.
             page (int): Index of page.
             num_row (int): Number of rows in a page.

        Returns:
            list[dict], data queried by category id.

        Raises:
            ParamValueError: If any parameter is invalid.
            MRMFetchDataError: If failed to fetch data by category.
            MRMUnsupportedSchemaError: If schema is invalid.
        """
        if not isinstance(category_id, int) or category_id < 0:
            raise ParamValueError("Category id should be int and greater than or equal to 0.")
        if not isinstance(page, int) or page < 0:
            raise ParamValueError("Page should be int and greater than or equal to 0.")
        if not isinstance(num_row, int) or num_row <= 0:
            raise ParamValueError("num_row should be int and greater than 0.")
        return self._segment.read_at_page_by_id(category_id, page, num_row)

    def read_at_page_by_name(self, category_name, page, num_row):
        """
        Query by category name in pagination.

        Args:
            category_name (str): String of category field's value,
                referred to the return of `read_category_info`.
            page (int): Index of page.
            num_row (int): Number of row in a page.

        Returns:
            list[dict], data queried by category name.
        """
        if not isinstance(category_name, str):
            raise ParamValueError("Category name should be str.")
        if not isinstance(page, int) or page < 0:
            raise ParamValueError("Page should be int and greater than or equal to 0.")
        if not isinstance(num_row, int) or num_row <= 0:
            raise ParamValueError("num_row should be int and greater than 0.")
        return self._segment.read_at_page_by_name(category_name, page, num_row)
