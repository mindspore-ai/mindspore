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
"""Op compute time files parser."""
import os
import stat
from mindspore.profiler.common.util import fwrite_format
from mindspore.profiler.common.exceptions.exceptions import ProfilerFileNotFoundException, \
    ProfilerIOException
from mindspore import log as logger
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path
from mindspore.profiler.parser.container import HWTSContainer

TIMELINE_FILE_COLUMN_TITLE = 'op_name, stream_id, start_time(ms), duration(ms)'


class OPComputeTimeParser:
    """
    Join hwts info and framework info, get op time info, and output to the result file.

    Args:
         hwts_output_file (str): The file path of hwts_output_file. Such as: './output_format_data_hwts_0.txt".
         output_filename (str): The output data file path and name. Such as: './output_op_compute_time_0.txt'.
         op_task_info (dict): The task and op relation info. The format: {task_id, [opname, stream_id, block dim]}.
    """

    _dst_file_title = 'title:op compute time'
    _dst_file_column_title = 'op_name       compute_time(ms) stream_id execution_times'
    _dst_file_column_title += '\n------------  ---------------  ---------'

    def __init__(self, hwts_output_file, output_filename, op_task_info,
                 output_path, device_id):
        hwts_output_file = validate_and_normalize_path(hwts_output_file)
        self._hwts_output_file = hwts_output_file
        self._output_filename = output_filename
        self._op_task_info = op_task_info
        self._output_path = output_path
        self._device_id = device_id
        self._min_cycle_counter = float("inf")

    @property
    def min_cycle_counter(self):
        """Get minimum cycle counter."""
        return self._min_cycle_counter

    @staticmethod
    def _convert_op_time_unit(op_data_list, op_name_time_dict, op_name_stream_dict,
                              op_name_count_dict, op_name_task_dict, op_name_start_time):
        """
        Calculate the execution time of operator and convert it into millisecond.

        Args:
            op_data_list (list): The list of operator metadata.
            op_name_time_dict (dict): The mapping relation of operator name and its execution time.
            op_name_stream_dict (dict): The mapping relation of operator name and its stream id.
            op_name_count_dict (dict): The mapping relation of operator name and its count.
            op_name_task_dict (dict): The mapping relation of operator name and its task id.
            op_name_start_time (dict): The mapping relation of operator name and its start time.
        """
        factor = 1e5
        for item in op_data_list:
            op_name = item.op_name
            # Unit conversion: converting the cycle counter into ms.
            op_start_time_str = str(item.cycle_counter / factor)
            op_duration = item.duration / factor
            op_duration_str = str(item.duration / factor)
            if op_name in op_name_time_dict.keys():
                op_name_time_dict[op_name] += op_duration
                op_name_count_dict[op_name] += 1
                op_name_start_time[op_name].append(
                    (op_start_time_str, op_duration_str)
                )

            else:
                op_name_time_dict[op_name] = op_duration
                op_name_stream_dict[op_name] = item.stream_id
                op_name_task_dict[op_name] = item.task_id
                op_name_count_dict[op_name] = 1
                op_name_start_time[op_name] = []
                op_name_start_time[op_name].append(
                    (op_start_time_str, op_duration_str)
                )

    def execute(self):
        """Execute the parser, compute all op, get op time, and write it to the output file."""
        # Calculate the execution time of operators,
        # and update the minimum cycle counter.
        tmp_result_data = self._calculate_op_execution_time()

        # Convert time units from nanoseconds to milliseconds.
        # The unit of the cycle counter is 10 nanoseconds.
        op_name_time_dict = {}
        op_name_stream_dict = {}
        op_name_count_dict = {}
        op_name_task_dict = {}
        op_name_start_time = {}
        self._convert_op_time_unit(
            tmp_result_data, op_name_time_dict, op_name_stream_dict,
            op_name_count_dict, op_name_task_dict, op_name_start_time
        )

        result_data = ""
        total_time = 0
        for op_name, time in op_name_time_dict.items():
            if op_name in op_name_stream_dict.keys():
                stream_id = op_name_stream_dict.get(op_name)
                if op_name_count_dict.get(op_name) == 0:
                    raise ValueError("The number of operations can not be 0.")
                avg_time = time / op_name_count_dict.get(op_name)
                total_time += avg_time
                result_data += ("%s %s %s %s\n" % (op_name, str(avg_time), stream_id, op_name_count_dict.get(op_name)))
        result_data += ("total op  %s 0" % (str(total_time)))

        timeline_data = []
        for op_name, _ in op_name_time_dict.items():
            if op_name in op_name_stream_dict.keys():
                stream_id = op_name_stream_dict[op_name]
                start_time_list = op_name_start_time.get(op_name)
                for (start_time, duration) in start_time_list:
                    timeline_data.append([op_name, stream_id, start_time, duration])

        # Write the metadata of operators into the file,
        # including operator name, average time, and stream id.
        self._write_op_time_into_file(result_data)
        # Write the timeline data into file,
        # including operator name, stream id, start time, and duration.
        self._write_timeline_data_into_file(timeline_data)

    def _get_op_task_id_map(self):
        """
        Read hwts data file, get the task time info.

        Returns:
           list: all hwts task time info.
        """

        op_map_result = []
        hwts_list = []

        if not os.path.exists(self._hwts_output_file):
            logger.critical('The hwts output file does not exist.')
            raise ProfilerFileNotFoundException('hwts output file')

        with open(self._hwts_output_file, 'r') as data_file:
            lines = data_file.readlines()
            for line in lines:
                if line.startswith("Start of task") or line.startswith("End of task"):
                    line_split = line.split()
                    container = HWTSContainer(line_split)
                    hwts_list.append(container)

        # hwts op map by taskId
        for hwts in hwts_list:
            if hwts.task_id in self._op_task_info.keys():
                hwts.op_name = self._op_task_info[hwts.task_id]
                op_map_result.append(hwts)

        return op_map_result

    def _write_op_time_into_file(self, result_data):
        """
        Write the metadata of operators into the file, including
            op name, average time, and stream id.

        Args:
            result_data (str): The metadata to be written into the file.
                    'op_name_1', 'avg_time_1', 'stream_id_1',
                    'op_name_2', 'avg_time_2', 'stream_id_2',
                    ...
        """

        fwrite_format(self._output_filename, data_source=self._dst_file_title, is_start=True)
        fwrite_format(self._output_filename, data_source=self._dst_file_column_title)
        fwrite_format(self._output_filename, data_source=result_data)

    def _write_timeline_data_into_file(self, timeline_data):
        """
        Write the timeline information into the file, including
            operator name, stream id, start time and duration.

        Args:
            timeline_data (list): The metadata to be written into the file.
                [
                    ['op_name_1', 'stream_id_1', 'start_time_1', 'duration_1'],
                    ['op_name_2', 'stream_id_2', 'start_time_2', 'duration_2'],
                    [...]
                ]
        """
        # sorted by start times
        timeline_data.sort(key=lambda x: float(x[2]))
        filename = 'output_timeline_data_{}.txt'.format(self._device_id)
        file_path = os.path.join(self._output_path, filename)
        file_path = validate_and_normalize_path(file_path)

        # write to file
        try:
            with open(file_path, 'w') as f_obj:
                f_obj.write(TIMELINE_FILE_COLUMN_TITLE + '\n')
                for timeline in timeline_data:
                    timeline = [str(item) for item in timeline]
                    f_obj.write(','.join(timeline) + '\n')
            os.chmod(file_path, stat.S_IREAD | stat.S_IWRITE)
        except (IOError, OSError) as err:
            logger.critical('Error occurred when writing intermediate timeline file: %s', err)
            raise ProfilerIOException from err

    def _calculate_op_execution_time(self):
        """
        Calculate the execution time of each operator.

        Returns:
            list, including the intermediate data of op execution time.
        """
        tmp_result_data = []
        op_map_list = self._get_op_task_id_map()

        cur_index = 0
        length = len(op_map_list)
        min_cycle_counter = float("inf")
        while cur_index < length:
            if cur_index + 1 == length:
                break

            op_start = op_map_list[cur_index]
            op_end = op_map_list[cur_index + 1]
            if op_start.status == "Start" and op_end.status == "End" \
                    and op_start.op_name == op_end.op_name:
                op_start.duration = op_end.cycle_counter - op_start.cycle_counter
                tmp_result_data.append(op_start)
                cur_index += 2
                if not op_start.op_name.startswith("assign"):
                    min_cycle_counter = min(min_cycle_counter, op_start.cycle_counter)
            else:
                cur_index += 1

        # Update the value of minimum cycle counter.
        self._min_cycle_counter = min_cycle_counter / 1e5  # Convert the time unit from 10ns to 1ms

        return tmp_result_data
