# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from collections import namedtuple
import struct

from mindspore.profiler.common.util import fwrite_format, get_file_join_name
from mindspore import log as logger
from mindspore.profiler.common.struct_type import StructType


class DataPreProcessParser:
    """
    The Parser for AI CPU preprocess data.

    Args:
         input_path(str): The profiling job path.
         output_filename(str): The output data path and name.

    """
    AI_CPU_STRUCT = dict(
        magic_number=StructType.UINT16,
        data_tag=StructType.UINT16,
        stream_id=StructType.UINT16,
        task_id=StructType.UINT16,
        run_start=StructType.UINT64,
        run_start_counter=StructType.UINT64,

        compute_start=StructType.UINT64,
        memcpy_start=StructType.UINT64,
        memcpy_end=StructType.UINT64,
        run_end=StructType.UINT64,
        run_end_counter=StructType.UINT64,
        thread=StructType.UINT32,

        device=StructType.UINT32,
        submit_tick=StructType.UINT64,
        schedule_tick=StructType.UINT64,
        tick_before_run=StructType.UINT64,
        tick_after_fun=StructType.UINT64,
        kernel_type=StructType.UINT32,

        dispatch_time=StructType.UINT32,
        total_time=StructType.UINT32,
        FFTS_thread_id=StructType.UINT16,
        version=StructType.UINT8
    )

    AiCpuStruct = namedtuple(
        'AiCpuStruct', ['magic_number', 'data_tag', 'stream_id', 'task_id', 'run_start', 'run_start_counter',
                        'compute_start', 'memcpy_start', 'memcpy_end', 'run_end', 'run_end_counter', 'thread',
                        'device', 'submit_tick', 'schedule_tick', 'tick_before_run', 'tick_after_fun', 'kernel_type',
                        'dispatch_time', 'total_time', 'FFTS_thread_id', 'version']
    )

    _source_file_target_old = 'DATA_PREPROCESS.dev.AICPU.'
    _source_file_target = 'DATA_PREPROCESS.AICPU.'
    _dst_file_title = 'title:DATA_PREPROCESS AICPU'
    _dst_file_column_title = [
        'serial_number', 'node_type_name', 'total_time(ms)',
        'dispatch_time(ms)', 'execution_time(ms)', 'run_start',
        'run_end'
    ]
    _ms_unit = 1000
    _us_unit = 100  # Convert 10ns to 1us.
    _task_id_threshold = 65536

    def __init__(self, input_path, output_filename, op_task_dict):
        self._input_path = input_path
        self._output_filename = output_filename
        self._source_file_name = self._get_source_file()
        self._ms_kernel_flag = 3
        self._other_kernel_flag = 6
        self._ms_kernel_run_end_index = 2
        self._other_kernel_run_end_index = 5
        self._dispatch_time_index = 5
        self._total_time_index = 6
        self._result_list = []
        self._min_cycle_counter = float('inf')
        self._ai_cpu_len = 128
        self._op_task_dict = op_task_dict

    @property
    def min_cycle_counter(self):
        """Get minimum cycle counter in AI CPU."""
        return self._min_cycle_counter

    def execute(self):
        """Execute the parser, get result data, and write it to the output file."""

        if not os.path.exists(self._source_file_name):
            logger.info("Did not find the aicpu profiling source file")
            return

        with open(self._source_file_name, 'rb') as ai_cpu_data:
            content = ai_cpu_data.read()
            if content[0:2].hex().upper() == "5A5A":
                ai_cpu_total_time_summary, result_list = self.parser_binary_file(content)
            else:
                ai_cpu_total_time_summary, result_list = self.parser_txt_file(content)

        os.chmod(self._source_file_name, stat.S_IREAD)

        if result_list:
            ai_cpu_total_time = format(ai_cpu_total_time_summary, '.6f')
            result_list.append(["AI CPU Total Time(ms):", ai_cpu_total_time])
            fwrite_format(self._output_filename, " ".join(self._dst_file_column_title), is_start=True, is_print=True)
            fwrite_format(self._output_filename, result_list, is_print=True)

        # For timeline display.
        self._result_list = result_list

    def parser_binary_file(self, content):
        """Parse binary format file."""
        result_list = list()
        ai_cpu_total_time_summary = 0
        # Node serial number.
        serial_number = 1

        i = 0
        ai_cpu_format = StructType.format(DataPreProcessParser.AI_CPU_STRUCT.values())
        ai_cpu_size = StructType.sizeof(DataPreProcessParser.AI_CPU_STRUCT.values())
        while i < len(content):
            ai_cpu_data = struct.unpack(ai_cpu_format, content[i:i + ai_cpu_size])
            ai_cpu = DataPreProcessParser.AiCpuStruct(*ai_cpu_data)
            if ai_cpu.task_id < self._task_id_threshold:
                node_type_name = f'{ai_cpu.stream_id}_{ai_cpu.task_id}'
                if self._op_task_dict and node_type_name in self._op_task_dict:
                    node_type_name = self._op_task_dict[node_type_name].split('/')[-1]
                else:
                    logger.warning("[profiler] the op name of %s cannot be found.", node_type_name)
                exe_time = (float(ai_cpu.run_end) - float(ai_cpu.run_start)) / self._ms_unit
                total_time = ai_cpu.total_time / self._ms_unit
                result_list.append([serial_number, node_type_name, total_time, ai_cpu.dispatch_time / self._ms_unit,
                                    exe_time, ai_cpu.run_start_counter / self._us_unit,
                                    ai_cpu.run_end_counter / self._us_unit])

                ai_cpu_total_time_summary += total_time
                # Increase node serial number.
                serial_number += 1

            i = i + self._ai_cpu_len

        return ai_cpu_total_time_summary, result_list

    def parser_txt_file(self, content):
        """Parse txt format file."""
        ai_cpu_str = str(content.replace(b'\n\x00', b' ___ ').replace(b'\x00', b' ___ '))[2:-1]
        ai_cpu_lines = ai_cpu_str.split(" ___ ")
        result_list = list()
        ai_cpu_total_time_summary = 0
        # Node serial number.
        serial_number = 1
        for i in range(len(ai_cpu_lines) - 1):
            node_line = ai_cpu_lines[i]
            thread_line = ai_cpu_lines[i + 1]
            if "Node" in node_line and "Thread" in thread_line:
                # Get the node data from node_line
                result = self._get_kernel_result(
                    serial_number,
                    node_line.split(','),
                    thread_line.split(',')
                )

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
        return ai_cpu_total_time_summary, result_list

    def query_aicpu_data(self):
        """
        Get execution time of AI CPU operator.

        Returns:
            a dict, the metadata of AI CPU operator execution time.
        """
        stream_id = 0  # Default stream id for AI CPU.
        pid = 9000  # Default pid for AI CPU.
        total_time = 0
        min_cycle_counter = float('inf')
        aicpu_info = []
        op_count_list = []
        for aicpu_item in self._result_list:
            if "AI CPU Total Time(ms):" in aicpu_item:
                total_time = aicpu_item[-1]
                continue

            op_name = aicpu_item[1]
            start_time = float(aicpu_item[5]) / self._ms_unit
            min_cycle_counter = min(min_cycle_counter, start_time)
            duration = aicpu_item[4]
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
            if len(node_list) == self._ms_kernel_flag:
                node_type_name = node_list[0].split(':')[-1]
                run_end_index = self._ms_kernel_run_end_index
            elif len(node_list) == self._other_kernel_flag:
                node_type_name = node_list[0].split(':')[-1].split('/')[-1].split('-')[0]
                run_end_index = self._other_kernel_run_end_index
            else:
                logger.warning("the data format can't support 'node_list':%s", str(node_list))
                return None

            us_unit = 100  # Convert 10ns to 1us.
            run_start_counter = float(node_list[1].split(':')[-1].split(' ')[1]) / us_unit
            run_end_counter = float(node_list[run_end_index].split(':')[-1].split(' ')[1]) / us_unit
            run_start = node_list[1].split(':')[-1].split(' ')[0]
            run_end = node_list[run_end_index].split(':')[-1].split(' ')[0]
            exe_time = (float(run_end) - float(run_start)) / self._ms_unit
            total_time = float(thread_list[self._total_time_index].split('=')[-1].split()[0]) / self._ms_unit
            dispatch_time = float(thread_list[self._dispatch_time_index].split('=')[-1].split()[0]) / self._ms_unit

            return [number, node_type_name, total_time, dispatch_time, exe_time,
                    run_start_counter, run_end_counter]
        except IndexError as e:
            logger.error(e)
            return None
