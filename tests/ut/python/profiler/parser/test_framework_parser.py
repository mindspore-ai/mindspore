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
"""Test the framework parser module."""
import csv
import os
import json
import tempfile
from mindspore.profiler.parser.framework_parser import FrameworkParser
from tests.ut.python.profiler import FRAMEWORK_RAW_DATA


def get_framework_result(file_path):
    """
    Get framework result from the framework file.

    Args:
        file_path (str): The framework file path.

    Returns:
        list[list], the parsed framework information.
    """
    result = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            result.append(row)
    return result


class TestFrameworkParser:
    """Test the class of `FrameworkParser`."""
    def setup_class(self):
        """Initialization before test case execution."""
        self._profiling_path = os.path.realpath(FRAMEWORK_RAW_DATA)

    def test_format_save_path(self):
        """Test the querying save path function."""
        output_path = os.path.realpath('./')
        rank_id = '2'
        parser = FrameworkParser(self._profiling_path, rank_id, output_path)
        expect_result = os.path.join(output_path, FrameworkParser.output_file_format.format(rank_id=rank_id))
        assert parser.save_path == expect_result

    def test_parse(self):
        """Test the parse function."""
        with tempfile.TemporaryDirectory() as output_path:
            parser = FrameworkParser(self._profiling_path, '1', output_path)
            parser.parse()
            assert os.path.exists(parser.save_path)

            expected_point = {2: "Default/loss_scaling_manager-DynamicLossScaleUpdateCell/Add-op123",
                              3: "Default/Assign-op245",
                              4: "Default/Assign-op245"}
            assert parser.point_info == expected_point

            task_id_full_op_name_file = os.path.join(self._profiling_path, 'expected', 'task_id_full_op_name.json')
            self._compare_json(task_id_full_op_name_file, parser.to_task_id_full_op_name_dict())

    @staticmethod
    def _compare_json(expected_file, result_dict):
        with open(expected_file, 'r') as file_handler:
            expected_data = json.load(file_handler)

        assert expected_data == result_dict
