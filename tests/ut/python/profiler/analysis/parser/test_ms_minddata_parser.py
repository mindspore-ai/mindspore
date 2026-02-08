# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
"""Test the MindDataParser class."""
import os
import unittest
from unittest.mock import patch

from mindspore.profiler.analysis.parser.ms_minddata_parser import MindDataParser
from mindspore.profiler.common.constant import ProfilerActivity
from mindspore.profiler.common.log import ProfilerLogger
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.exceptions.exceptions import (
    ProfilerPathErrorException,
    ProfilerRawFileException
)


class TestMindDataParser(unittest.TestCase):
    @patch.object(ProfilerLogger, 'init')
    @patch.object(ProfilerLogger, 'get_instance')
    def setUp(self, mock_get_instance, mock_init):
        self.kwargs = {
            "rank_id": "0",
            "activities": [ProfilerActivity.CPU.value, ProfilerActivity.NPU.value],
            "framework_path": "test_minddata_path",
            "ascend_ms_dir": "test_ascend_ms_dir"
        }
        mock_logger = unittest.mock.MagicMock()
        mock_get_instance.return_value = mock_logger
        mock_init.return_value = None
        self.parser = MindDataParser(**self.kwargs)

    def test_parse_data_none(self):
        data = None
        result = self.parser._parse(data)
        self.assertEqual(result, {})

    @patch.object(os.path, "exists")
    @patch.object(FileManager, "read_json_file")
    def test_parse_pipeline_info_dict_raise_exception_when_pipeline_file_empty(self, mock_read_json_file, mock_exists):
        mock_exists.return_value = True
        mock_read_json_file.return_value = None
        with self.assertRaises(ProfilerRawFileException) as exception:
            self.parser._parse_pipeline_info_dict()
        self.assertIn("pipeline file is empty", str(exception.exception))

    @patch.object(os.path, "exists")
    @patch.object(FileManager, "read_json_file")
    def test_parse_pipeline_info_dict_raise_exception_when_none_sample_interval(self, mock_read_json_file, mock_exists):
        mock_exists.return_value = True
        mock_read_json_file.return_value = {
            "sampling_interval": None,
            "op_info": [{"op_id": 1, "op_type": "TFReader"}]
        }
        with self.assertRaises(ProfilerRawFileException) as exception:
            self.parser._parse_pipeline_info_dict()
        self.assertIn("The format of minddata pipeline raw file is wrong", str(exception.exception))

    @patch.object(os.path, "exists")
    @patch.object(FileManager, "read_json_file")
    def test_parse_pipeline_info_dict_raise_exception_when_empty_op_info(self, mock_read_json_file, mock_exists):
        mock_exists.return_value = True
        mock_read_json_file.return_value = {
            "sampling_interval": 10,
            "op_info": []
        }
        with self.assertRaises(ProfilerRawFileException) as exception:
            self.parser._parse_pipeline_info_dict()
        self.assertIn("The format of minddata pipeline raw file is wrong", str(exception.exception))

    @patch.object(os.path, "exists")
    @patch.object(FileManager, "read_json_file")
    def test_parse_pipeline_info_dict_raise_exception_when_none_op_info(self, mock_read_json_file, mock_exists):
        mock_exists.return_value = True
        mock_read_json_file.return_value = {
            "sampling_interval": 10,
            "op_info": None
        }
        with self.assertRaises(ProfilerRawFileException) as exception:
            self.parser._parse_pipeline_info_dict()
        self.assertIn("The format of minddata pipeline raw file is wrong", str(exception.exception))

    @patch.object(os.path, "exists")
    @patch.object(FileManager, "read_json_file")
    def test_parse_pipeline_info_dict_raise_exception_when_duplicate_op_id(self, mock_read_json_file, mock_exists):
        mock_exists.return_value = True
        mock_read_json_file.return_value = {
            "sampling_interval": 10,
            "op_info": [
                {"op_id": 1, "op_type": "TFReader"},
                {"op_id": 1, "op_type": "Shuffle"},
                {"op_id": 2, "op_type": "Batch"}
            ]
        }
        with self.assertRaises(ProfilerRawFileException) as exception:
            self.parser._parse_pipeline_info_dict()
        self.assertIn("The content of minddata pipeline raw file is wrong", str(exception.exception))

    @patch.object(FileManager, "read_json_file")
    def test_parse_cpu_util_info(self, mock_read_json_file):
        mock_read_json_file.return_value = None
        cpu_util_info = self.parser._parse_cpu_util_info()
        self.assertEqual(cpu_util_info, {})

    @patch.object(MindDataParser, "_setup_device_trace")
    @patch.object(FileManager, "read_txt_file")
    def test_parse_device_trace(self, mock_read_txt_file, mock_setup_device_trace):
        mock_setup_device_trace.return_value = ["mock_device_trace_path", True]
        mock_read_txt_file.return_value = None
        device_trace = self.parser._parse_device_trace()
        self.assertEqual(device_trace, [])

    def test_setup_device_trace(self):
        with self.assertRaises(ProfilerPathErrorException) as exception:
            self.parser._setup_device_trace()
        self.assertIn("A MindData device trace profiling file cannot be found", str(exception.exception))


if __name__ == "__main__":
    unittest.main()
