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
"""Test the minddata pipeline parser module."""
import csv
import os
import shutil
import tempfile

from mindspore.profiler.parser.minddata_pipeline_parser import \
    MinddataPipelineParser
from tests.ut.python.profiler import PROFILER_DIR, RAW_DATA, RAW_DATA_JOB2


def get_minddata_pipeline_result(file_path):
    """
    Get minddata pipeline result from the minddata pipeline file.

    Args:
        file_path (str): The minddata pipeline file path.

    Returns:
        list[list], the parsed minddata pipeline information.
    """
    result = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            result.append(row)
    return result


class TestMinddataPipelineParser:
    """Test the class of `MinddataPipelineParser`."""
    def setup_method(self):
        """Initialization before test case execution."""
        self._output_path_1 = tempfile.mkdtemp(
            prefix='test_minddata_pipeline_parser_'
        )
        self._parser_1 = MinddataPipelineParser(
            RAW_DATA, '0', self._output_path_1
        )

        self._output_path_2 = tempfile.mkdtemp(
            prefix='test_minddata_pipeline_parser_'
        )
        self._parser_2 = MinddataPipelineParser(
            RAW_DATA_JOB2, '0', self._output_path_2
        )

    def teardown_method(self) -> None:
        """Clear up after test case execution."""
        shutil.rmtree(self._output_path_1)
        shutil.rmtree(self._output_path_2)

    def test_save_path(self):
        """Test the querying save path function."""
        expect_result = os.path.join(
            self._output_path_1, 'minddata_pipeline_raw_0.csv'
        )
        assert expect_result == self._parser_1.save_path

    def test_parse(self):
        """Test the parse function."""
        expect_pipeline_file = os.path.join(
            PROFILER_DIR, 'minddata_pipeline_raw_0.csv'
        )
        expect_result = get_minddata_pipeline_result(expect_pipeline_file)

        self._parser_1.parse()
        pipeline_file = os.path.join(
            self._output_path_1, 'minddata_pipeline_raw_0.csv'
        )
        result = get_minddata_pipeline_result(pipeline_file)
        assert expect_result == result

        self._parser_2.parse()
        pipeline_file = os.path.join(
            self._output_path_2, 'minddata_pipeline_raw_0.csv'
        )
        result = get_minddata_pipeline_result(pipeline_file)
        assert expect_result == result
