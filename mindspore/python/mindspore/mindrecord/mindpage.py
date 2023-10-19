# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
This module is to support reading page from MindRecord.
"""

from .shardsegment import ShardSegment
from .shardutils import check_parameter
from .common.exceptions import ParamValueError, ParamTypeError, MRMDefineCategoryError

__all__ = ['MindPage']


class MindPage:
    """
    Class to read MindRecord files in pagination.

    Args:
        file_name (Union[str, list[str]]): One of MindRecord files or a file list.
        num_consumer (int, optional): The number of reader workers which load data. Default: ``4`` .
            It should not be smaller than 1 or larger than the number of processor cores.

    Raises:
        ParamValueError: If `file_name` is not type str or list[str].
        ParamValueError: If `num_consumer` is not type int.

    Examples:
        >>> from mindspore.mindrecord import MindPage
        >>>
        >>> mindrecord_file = "/path/to/mindrecord/file"
        >>> mind_page = MindPage(mindrecord_file)
        >>>
        >>> # get all the index fields
        >>> fields = mind_page.candidate_fields
        >>>
        >>> # set the field to be retrieved
        >>> mind_page.category_field = "file_name"
        >>>
        >>> # get all the group info
        >>> info = mind_page.read_category_info()
        >>>
        >>> # get the row by id which is from category info
        >>> row_by_id = mind_page.read_at_page_by_id(0, 0, 1)
        >>>
        >>> # get the row by name which is from category info
        >>> row_by_name = mind_page.read_at_page_by_name("8.jpg", 0, 1)
    """
    @check_parameter
    def __init__(self, file_name, num_consumer=4):
        self._segment = ShardSegment()
        self._segment.open(file_name, num_consumer)
        self._category_field = None
        self._candidate_fields = [field[:field.rfind('_')] for field in self._segment.get_category_fields()]

    @property
    def candidate_fields(self):
        """
        Return candidate category fields.

        Note:
            Please refer to the Examples of :class:`mindspore.mindrecord.MindPage` .

        Returns:
            list[str], by which data could be grouped.
        """
        return self._candidate_fields

    @property
    def category_field(self):
        """
        Getter function for category fields.

        Note:
            Please refer to the Examples of :class:`mindspore.mindrecord.MindPage` .

        Returns:
            list[str], by which data could be grouped.
        """
        return self._category_field

    @category_field.setter
    def category_field(self, category_field):
        """
        Setter function for category field.

        Note:
            Please refer to the Examples of :class:`mindspore.mindrecord.MindPage` .

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

        Note:
            Please refer to the Examples of :class:`mindspore.mindrecord.MindPage` .

        Returns:
            str, description of group information.

        Raises:
            MRMReadCategoryInfoError: If failed to read category information.
        """
        return self._segment.read_category_info()

    def read_at_page_by_id(self, category_id, page, num_row):
        """
        Query by category id in pagination.

        Note:
            Please refer to the Examples of :class:`mindspore.mindrecord.MindPage` .

        Args:
             category_id (int): Category id, referred to the return of `read_category_info` .
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

        Note:
            Please refer to the Examples of :class:`mindspore.mindrecord.MindPage` .

        Args:
            category_name (str): String of category field's value,
                referred to the return of `read_category_info` .
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
