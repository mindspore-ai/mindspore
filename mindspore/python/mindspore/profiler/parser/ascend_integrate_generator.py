# Copyright 2024 Huawei Technologies Co., Ltd
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
"""PROF csv data analyze module"""
import os

from mindspore.profiler.parser.ascend_analysis.file_manager import FileManager


class AscendIntegrateGenerator:
    """Generate ms profiler output csv"""

    def __init__(self, mindstudio_profiler_output: str, ascend_ms_output_path: str):
        self._mindstudio_profiler_output = mindstudio_profiler_output
        self._ascend_ms_output_path = ascend_ms_output_path

    def parse(self):
        """Generate ms profiler output csv"""
        self._generate_kernel_details()

    def _generate_kernel_details(self):
        """Generate kernel_details.csv"""
        header_map = {
            "Op Name": "Name", "OP Type": "Type", "Task Type": "Accelerator Core",
            "Task Start Time(us)": "Start Time(us)", "Task Duration(us)": "Duration(us)",
            "Task Wait Time(us)": "Wait Time(us)",
        }
        op_summary_file_list = FileManager.get_csv_file_list_by_start_name(self._mindstudio_profiler_output,
                                                                           "op_summary")
        kernel_details_file = os.path.join(self._ascend_ms_output_path, "kernel_details.csv")
        FileManager.combine_csv_file(op_summary_file_list, kernel_details_file, header_map)
