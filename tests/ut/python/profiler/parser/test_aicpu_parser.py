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
# ============================================================================
"""Test the aicpu parser."""
import os
import tempfile
import shutil

from unittest import TestCase

from mindspore.profiler.parser.aicpu_data_parser import DataPreProcessParser


def get_result(file_path):
    """
    Get result from the aicpu file.

    Args:
        file_path (str): The aicpu file path.

    Returns:
        list[list], the parsed aicpu information.
    """
    result = []
    file = None
    try:
        file = open(file_path, 'r')
        result.append(file.read())
        return result
    finally:
        if file:
            file.close()


class TestAicpuParser(TestCase):
    """Test the class of Aicpu Parser."""

    def setUp(self) -> None:
        """Initialization before test case execution."""
        self.profiling_dir = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                           '../../../data/profiler_data/'
                                                           'JOB_AICPU/data'))
        self.expect_dir = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                        '../../../data/profiler_data/'
                                                        'JOB_AICPU/expect'))
        self.output_path = tempfile.mkdtemp(prefix='output_data_preprocess_aicpu_')
        self.output_file = os.path.join(self.output_path, 'output_data_preprocess_aicpu_0.txt')
        self.expect_file = os.path.join(self.expect_dir, 'output_data_preprocess_aicpu_0.txt')

    def test_aicpu_parser(self):
        """Test the class of Aicpu Parser."""
        data = DataPreProcessParser(self.profiling_dir, self.output_file)
        data.execute()
        expect_result = get_result(self.expect_file)
        result = get_result(self.output_file)
        shutil.rmtree(self.output_path)
        assert expect_result == result

    def test_aicpu_parser_file_not_exist(self):
        """Test the class of Aicpu Parser."""
        profiling_dir = os.path.realpath(os.path.join(self.profiling_dir, 'data'))
        data = DataPreProcessParser(profiling_dir, self.output_file)
        data.execute()
        shutil.rmtree(self.output_path)
