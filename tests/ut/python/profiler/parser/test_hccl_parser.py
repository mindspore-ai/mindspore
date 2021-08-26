# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Test the hccl parser module."""
import csv
import os
import shutil
import tempfile

from mindspore.profiler.parser.hccl_parser import HcclParser
from tests.ut.python.profiler import PROFILER_DIR


def get_hccl_result(file_path):
    """
    Get hccl result from the hccl file.

    Args:
        file_path (str): The hccl file path.

    Returns:
        list[list], the parsed hccl information.
    """
    result = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            result.append(row)
    return result


class TestHcclParser:
    """Test the class of `HcclParser`."""
    def setup_method(self):
        """Initialization before test case execution."""
        self._output_path = tempfile.mkdtemp(
            prefix='test_hccl_parser_'
        )
        shutil.copyfile(os.path.join(PROFILER_DIR, 'step_trace_raw_6_detail_time.csv'),
                        os.path.join(self._output_path, 'step_trace_raw_6_detail_time.csv'))
        self._parser = HcclParser(os.path.join(PROFILER_DIR, 'hccl_info'), '6', '6', self._output_path)

    def teardown_method(self) -> None:
        """Clear up after test case execution."""
        shutil.rmtree(self._output_path)

    def test_parse(self):
        """Test the parse function."""
        expect_hccl_file = os.path.join(
            PROFILER_DIR, 'hccl_raw_6.csv'
        )
        expect_result = get_hccl_result(expect_hccl_file)

        self._parser.parse()
        hccl_file = os.path.join(
            self._output_path, 'hccl_raw_6.csv'
        )
        result = get_hccl_result(hccl_file)
        assert expect_result == result
