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
"""
The parser for AI CPU preprocess data.
"""
import os
import stat

from mindspore.profiler.common.util import fwrite_format, get_file_join_name
from mindspore import log as logger


class DataPreProcessParser:
    """
    The Parser for AI CPU preprocess data.

    Args:
         input_path(str): The profiling job path.
         output_filename(str): The output data path and name.

    """
    _source_file_target_old = 'DATA_PREPROCESS.dev.AICPU.'
    _source_file_target = 'DATA_PREPROCESS.AICPU.'
    _dst_file_title = 'title:DATA_PREPROCESS AICPU'
    _dst_file_column_title = ['serial_number', 'node_type_name', 'total_time(ms)',
                              'dispatch_time(ms)', 'run_start', 'run_end']
    _ms_unit = 1000

    def __init__(self, input_path, output_filename):
        self._input_path = input_path
        self._output_filename = output_filename
        self._source_file_name = self._get_source_file()
        self._ms_kernel_flag = 3
        self._other_kernel_flag = 6
        self._thread_flag = 7
        self._ms_kernel_run_end_index = 2
        self._other_kernel_run_end_index = 5
        self._result_list = []
        self._min_cycle_counter = float('inf')

    def _get_source_file(self):
        """Get log file name, which was created by ada service."""
        file_name = get_file_join_name(self._input_path, self._source_file_target)
        if not file_name:
            file_name = get_file_join_name(self._input_path, self._source_file_target_old)
            if not file_name:
                data_path = os.path.join(self._input_path, "data")
                file_name = get_file_join_name(data_path, self._source_file_target)
                if not file_name:
                    file_name = get_file_join_name(data_path, self._source_file_target_old)
        return file_name

    def _get_kernel_result(self, number, node_list, thread_list):
        """Get the profiling data form different aicpu kernel"""
        try:
            if len(node_list) == self._ms_kernel_flag and len(thread_list) == self._thread_flag:
                node_type_name = node_list[0].split(':')[-1]
                run_end_index = self._ms_kernel_run_end_index
            elif len(node_list) == self._other_kernel_flag and len(thread_list) == self._thread_flag:
                node_type_name = node_list[0].split(':')[-1].split('/')[-1].split('-')[0]
                run_end_index = self._other_kernel_run_end_index
            else:
                logger.warning("the data format can't support 'node_list':%s", str(node_list))
                return None

            run_start = node_list[1].split(':')[-1].split(' ')[0]
            run_end = node_list[run_end_index].split(':')[-1].split(' ')[0]
            total_time = float(thread_list[-1].split('=')[-1].split()[0]) / self._ms_unit
            dispatch_time = float(thread_list[-2].split('=')[-1].split()[0]) / self._ms_unit

            return [number, node_type_name, total_time, dispatch_time,
                    run_start, run_end]
        except IndexError as e:
            logger.error(e)
            return None

    def execute(self):
        """Execute the parser, get result data, and write it to the output file."""

        if not os.path.exists(self._source_file_name):
            logger.info("Did not find the aicpu profiling source file")
            return

        with open(self._source_file_name, 'rb') as ai_cpu_data:
            ai_cpu_str = str(ai_cpu_data.read().replace(b'\n\x00', b' ___ ')
                             .replace(b'\x00', b' ___ '))[2:-1]
            ai_cpu_lines = ai_cpu_str.split(" ___ ")
        os.chmod(self._source_file_name, stat.S_IREAD | stat.S_IWRITE)
        result_list = list()
        ai_cpu_total_time_summary = 0
        # Node serial number.
        serial_number = 1
        for i in range(len(ai_cpu_lines) - 1):
            node_line = ai_cpu_lines[i]
            thread_line = ai_cpu_lines[i + 1]
            if "Node" in node_line and "Thread" in thread_line:
                # Get the node data from node_line
                node_list = node_line.split(',')
                thread_list = thread_line.split(',')
                result = self._get_kernel_result(serial_number, node_list, thread_list)

                if result is None:
                    continue

                result_list.append(result)
                # Calculate the total time.
                total_time = result[2]
                ai_cpu_total_time_summary += total_time
                # Increase node serial number.
                serial_number += 1
            elif "Node" in node_line and "Thread" not in thread_line:
                node_type_name = node_line.split(',')[0].split(':')[-1]
                logger.warning("The node type:%s cannot find thread data", node_type_name)

        if result_list:
            ai_cpu_total_time = format(ai_cpu_total_time_summary, '.6f')
            result_list.append(["AI CPU Total Time(ms):", ai_cpu_total_time])
            fwrite_format(self._output_filename, " ".join(self._dst_file_column_title), is_start=True, is_print=True)
            fwrite_format(self._output_filename, result_list, is_print=True)

        # For timeline display.
        self._result_list = result_list

    def query_aicpu_data(self):
        """
        Get execution time of AI CPU operator.

        Returns:
            a dict, the metadata of AI CPU operator execution time.
        """
        stream_id = 0  # Default stream id for AI CPU.
        pid = 9000  # Default pid for AI CPU.
        factor = 1000  # Convert time unit from 1us to 1ms
        total_time = 0
        min_cycle_counter = float('inf')
        aicpu_info = []
        op_count_list = []
        for aicpu_item in self._result_list:
            if "AI CPU Total Time(ms):" in aicpu_item:
                total_time = aicpu_item[-1]
                continue

            op_name = aicpu_item[1]
            start_time = float(aicpu_item[4]) / factor
            min_cycle_counter = min(min_cycle_counter, start_time)
            end_time = float(aicpu_item[5]) / factor
            duration = end_time - start_time
            aicpu_info.append([op_name, stream_id, start_time, duration, pid])

            # Record the number of operator types.
            if op_name not in op_count_list:
                op_count_list.append(op_name)

        self._min_cycle_counter = min_cycle_counter
        aicpu_dict = {
            'info': aicpu_info,
            'total_time': float(total_time),
            'op_exe_times': len(aicpu_info),
            'num_of_ops': len(op_count_list),
            'num_of_streams': 1
        }

        return aicpu_dict

    @property
    def min_cycle_counter(self):
        """Get minimum cycle counter in AI CPU."""
        return self._min_cycle_counter
