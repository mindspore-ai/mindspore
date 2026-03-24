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
"""Test the AscendKernelDetailsViewer class."""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from mindspore.profiler.analysis.viewer.ascend_kernel_details_viewer import AscendKernelDetailsViewer
from mindspore.profiler.common.constant import (
    JitLevel,
    ProfilerLevel,
    OpSummaryHeaders,
    ProfilerActivity
)
from mindspore.profiler.common.file_manager import FileManager


# pylint: disable=protected-access
class TestAscendKernelDetailsViewer(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MagicMock()
        self.kwargs = {
            "ascend_profiler_output_path": "/fake/path",
            "ascend_ms_dir": "/fake/ms_dir",
            "is_set_schedule": True,
            "jit_level": JitLevel.GRAPH_LEVEL,
            "profiler_level": ProfilerLevel.Level1.value,
            "activities": [ProfilerActivity.CPU.value]
        }
        with patch('mindspore.profiler.analysis.viewer.ascend_kernel_details_viewer.ProfilerLogger') \
            as mock_profiler_logger:
            mock_profiler_logger.get_instance.return_value = self.mock_logger
            self.viewer = AscendKernelDetailsViewer(**self.kwargs)

    @patch.object(AscendKernelDetailsViewer, '_check_input_data')
    @patch.object(AscendKernelDetailsViewer, '_update_kernel_name_and_step_id')
    @patch.object(AscendKernelDetailsViewer, '_update_headers')
    @patch.object(AscendKernelDetailsViewer, '_write_data')
    def test_save_should_success_when_correct(self, mock_write_data, mock_update_headers,
                                              mock_update_kernel_name, mock_check_input_data):
        data = {
            "trace_view_container": MagicMock(),
            "op_summary": MagicMock(),
            "op_summary_headers": MagicMock()
        }
        self.viewer.save(data)

        self.viewer._logger.info.assert_any_call("AscendKernelDetailsViewer start")
        mock_check_input_data.assert_called_once_with(data)
        mock_update_kernel_name.assert_called_once()
        mock_update_headers.assert_called_once()
        mock_write_data.assert_called_once()
        self.viewer._logger.info.assert_any_call("Kernel details saved done")
        self.viewer._logger.info.assert_any_call("AscendKernelDetailsViewer end")

    def test_save_profiler_should_not_end_when_level_none(self):
        self.viewer._profiler_level = ProfilerLevel.LevelNone.value
        data = {}
        self.viewer.save(data)
        self.viewer._logger.info.assert_called_once_with("AscendKernelDetailsViewer start")

    def test_save_should_log_error_when_exception(self):
        data = {
            "trace_view_container": MagicMock(),
            "op_summary": MagicMock(),
            "op_summary_headers": MagicMock()
        }
        with patch.object(self.viewer, '_check_input_data', side_effect=Exception("Test exception")):
            self.viewer.save(data)
            self.viewer._logger.error.assert_called_once()

    def test_check_input_data_should_success_when_correct(self):
        data = {
            "trace_view_container": MagicMock(),
            "op_summary": MagicMock(),
            "op_summary_headers": MagicMock()
        }
        self.viewer._check_input_data(data)
        self.assertEqual(self.viewer.trace_container, data["trace_view_container"])
        self.assertEqual(self.viewer.op_summary, data["op_summary"])
        self.assertEqual(self.viewer.op_summary_headers, data["op_summary_headers"])

    def test_check_input_data_should_exception_when_op_summary_empty(self):
        data = {
            "trace_view_container": MagicMock(),
            "op_summary": None,
            "op_summary_headers": MagicMock()
        }
        with self.assertRaises(ValueError) as context:
            self.viewer._check_input_data(data)
        self.assertEqual(str(context.exception), "op summary is empty")

    def test_check_input_data_should_exception_when_trace_container_none(self):
        data = {
            "trace_view_container": None,
            "op_summary": MagicMock(),
            "op_summary_headers": MagicMock()
        }
        with self.assertRaises(ValueError) as context:
            self.viewer._check_input_data(data)
        self.assertEqual(str(context.exception), "trace view container is None")

    @patch.object(FileManager, 'create_csv_file')
    def test_write_data_should_success_when_correct(self, mock_create_csv_file):
        self.viewer.op_summary = [
            {OpSummaryHeaders.OP_NAME.value: "op1", OpSummaryHeaders.TASK_START_TIME.value: 100},
            {OpSummaryHeaders.OP_NAME.value: "op2", OpSummaryHeaders.TASK_START_TIME.value: 200}
        ]
        self.viewer.op_summary_headers = [OpSummaryHeaders.OP_NAME.value, OpSummaryHeaders.TASK_START_TIME.value]
        self.viewer.kernel_details_headers = ["Name", "Start Time(us)"]
        self.viewer._save_path = MagicMock
        self.viewer._write_data()

        mock_create_csv_file.assert_called_once_with(
            file_path=self.viewer._save_path,
            data=[['op1', 100], ['op2', 200]],
            headers=self.viewer.kernel_details_headers
        )

    def test_update_headers_should_success_when_correct(self):
        self.viewer.op_summary_headers = [
            OpSummaryHeaders.OP_NAME.value,
            OpSummaryHeaders.DEVICE_ID.value,
            OpSummaryHeaders.MIX_BLOCK_NUM.value,
            OpSummaryHeaders.STEP_ID.value
        ]
        self.viewer._update_headers()
        self.assertEqual(OpSummaryHeaders.DEVICE_ID.value not in self.viewer.op_summary_headers, True)
        if self.viewer._profiler_level == ProfilerLevel.Level0.value:
            self.assertEqual(OpSummaryHeaders.MIX_BLOCK_NUM.value not in self.viewer.op_summary_headers, True)

    def test_update_kernel_name_and_step_id_should_success_when_correct(self):
        self.viewer.trace_container = MagicMock()
        mock_summary = MagicMock()
        mock_summary.__getitem__.return_value = MagicMock()
        self.viewer.op_summary = mock_summary
        op1 = MagicMock()
        type(op1).name = "kernel1"
        type(op1).ts = 100
        op1.parent = MagicMock()
        type(op1.parent).name = "fwk_op1"
        type(op1.parent).ts = 100
        type(op1).step_id = 1

        op2 = MagicMock()
        type(op2).name = "kernel2"
        type(op2).ts = 200
        op2.parent = MagicMock()
        type(op2.parent).name = "fwk_op2"
        type(op2.parent).ts = 200
        type(op2).step_id = 2

        self.viewer.trace_container.hardware_op_event = {
            1: [op1],
            2: [op2]
        }
        self.viewer.trace_container.get_step_id_time_dict.return_value = {
            1: (0, 150),
            2: (150, 300)
        }
        dtype = [
            (OpSummaryHeaders.OP_NAME.value, 'U50'),
            (OpSummaryHeaders.TASK_START_TIME.value, 'i8')
        ]
        data = np.array(
            [("kernel1", 100), ("kernel2", 200)],
            dtype=dtype
        )
        self.viewer.op_summary = data
        self.viewer._update_kernel_name_and_step_id()
        self.viewer._logger.info.assert_any_call("Update kernel name start")
        self.viewer._logger.info.assert_any_call("Update kernel name done")


if __name__ == "__main__":
    unittest.main()
