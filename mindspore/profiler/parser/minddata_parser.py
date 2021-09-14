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
"""Minddata aicpu parser."""
import os

from mindspore.profiler.common.util import get_file_join_name, fwrite_format
from mindspore import log as logger
from mindspore.profiler.common.validator.validate_path import \
    validate_and_normalize_path


class MinddataParser:
    """Minddata Aicpu Parser."""

    @staticmethod
    def parse_step_minddata_aicpu_data(one_step, result):
        """
        Parse step mind_data ai_cpu data.

        Args:
            one_step (str): The mind_data step info text, it is one of two structures.

                Type queue: node_name,queue_size,run_start,run_end
                Type run: node_name,run_start,run_end,queue_size

            result ([[node_name, node_start, node_end, queue_size]]): Step info list.
        """

        if not one_step:
            return
        node_info = one_step.split(", ")
        node_name, node_start, node_end, queue_size = "", 0, 0, 0
        if node_info:
            node_name = node_info[0].replace("Node:", "")

        if len(node_info) > 3:
            if "queue" in node_info[1]:
                queue_size = node_info[1].replace("queue size:", "")
                node_start = node_info[2].replace("Run start:", "")
                node_end = node_info[3].replace("Run end:", "")
            elif "Run" in node_info[1]:
                queue_size = node_info[3].replace("queue size:", "")
                node_start = node_info[1].replace("Run start:", "")
                node_end = node_info[2].replace("Run end:", "")
            queue_size = int(queue_size) if queue_size.isdigit() else queue_size
            node_start = int(node_start) if node_start.isdigit() else node_start
            node_end = int(node_end) if node_end.isdigit() else node_end

        one_step_list = [node_name, node_start, node_end, queue_size]
        result.append(one_step_list)

    @staticmethod
    def parse_minddata_aicpu_data(minddata_aicpu_source_path):
        """
        Parse minddata get_next info which contains queue size and execute time.

        Args:
            minddata_aicpu_source_path (str): the source file path.

        Returns:
            list[Union[str, float]], the converted data.
        """
        result = list()
        try:
            minddata_aicpu_source_path = validate_and_normalize_path(minddata_aicpu_source_path)
            with open(minddata_aicpu_source_path) as source_data_file:
                source_data = source_data_file.read()
                step_data = source_data.split("\x00")
                for one_step in step_data:
                    MinddataParser.parse_step_minddata_aicpu_data(one_step, result)
        except OSError:
            logger.error("Open get_next profiling file error.")

        return result

    @staticmethod
    def execute(source_path, output_path, device_id):
        """
        Execute the parser.

        Args:
            source_path (str): the source file path.
            output_path (str): the output file path.
            device_id (str): the device id.
        """
        col_names = ["node_name", "start_time", "end_time", "queue_size"]
        source_path = validate_and_normalize_path(source_path)
        minddata_aicpu_source_path = get_file_join_name(
            input_path=source_path, file_name='DATA_PREPROCESS.AICPUMI')
        if not minddata_aicpu_source_path:
            minddata_aicpu_source_path = get_file_join_name(
                input_path=source_path, file_name='DATA_PREPROCESS.dev.AICPUMI')
            if not minddata_aicpu_source_path:
                minddata_aicpu_source_path = get_file_join_name(
                    input_path=os.path.join(source_path, "data"), file_name='DATA_PREPROCESS.AICPUMI')
                if not minddata_aicpu_source_path:
                    minddata_aicpu_source_path = get_file_join_name(
                        input_path=os.path.join(source_path, "data"), file_name='DATA_PREPROCESS.dev.AICPUMI')
                    if not minddata_aicpu_source_path:
                        return
        minddata_aicpu_output_path = os.path.join(output_path, "minddata_aicpu_" + device_id + ".txt")
        minddata_aicpu_data = MinddataParser.parse_minddata_aicpu_data(minddata_aicpu_source_path)
        if minddata_aicpu_data:
            fwrite_format(minddata_aicpu_output_path, " ".join(col_names), is_start=True)
            fwrite_format(minddata_aicpu_output_path, minddata_aicpu_data, is_start=True)
