# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Op intermediate files parser."""
import csv
import os
import stat
from mindspore.profiler.common.exceptions.exceptions import ProfilerFileNotFoundException, \
    ProfilerIOException
from mindspore import log as logger
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path


class OPIntermediateParser:
    """
    Op intermediate files parser.

    Args:
        profiling_dir (str): The directory where the parsed profiling files are
            located.
        rank_id (str): The rank ID.
    """

    _output_timeline_data_file_path = 'output_timeline_data_{}.txt'
    _file_name_op_intermediate_type = 'pynative_op_intermediate_{}_type.csv'
    _file_name_op_intermediate_detail = 'pynative_op_intermediate_{}_detail.csv'

    _op_intermediate_type_header = ['op_type', 'execution_time', 'execution_frequency', 'percent']
    _op_intermediate_op_header = ['full_op_name', 'execution_time']

    _ms_decimal_digits = 6
    _percent_decimal_digits = 2

    def __init__(self, profiling_dir, rank_id):
        self._profiling_dir = profiling_dir
        self._rank_id = rank_id

    def get_timeline_data(self, all_reduce_names=None):
        """
        Load timeline data from file.

        Args:
            all_reduce_names (list): The communication operator list.
        """
        all_reduce_names = all_reduce_names or []
        file_path = os.path.join(
            self._profiling_dir,
            self._output_timeline_data_file_path.format(self._rank_id)
        )
        file_path = validate_and_normalize_path(file_path)
        if not os.path.exists(file_path):
            logger.critical("Failed to find parsed timeline file.")
            raise ProfilerFileNotFoundException('parsed timeline file')

        timeline_list = []
        try:
            with open(file_path, 'r') as f_obj:
                for line in f_obj:
                    # line: op_name, stream_id, start_time(ms), duration(ms)
                    line_list = line.strip('\n').split(',')
                    # filter out communication operators
                    if line_list[0] == 'op_name' or line_list[0] in all_reduce_names:
                        continue
                    timeline_list.append(line_list)
        except (IOError, OSError) as err:
            logger.critical('Error occurred when read timeline intermediate file: %s', err)
            raise ProfilerIOException() from err
        finally:
            pass

        return timeline_list

    def parser_pynative_op_intermediate_detail(self):
        """Parse pynative op intermediate detail."""
        timeline_list = self.get_timeline_data(None)
        # key:op name, value:[op count, total op execution time]
        op_intermediate_detail = {}
        for timeline in timeline_list:
            op_name = timeline[0].split('/')[-1]

            detail = op_intermediate_detail.get(op_name)
            if not detail:
                detail = [0, 0]
                op_intermediate_detail[op_name] = detail
            detail[0] = detail[0] + 1
            detail[1] = detail[1] + float(timeline[3])

        op_op_file_path = os.path.join(self._profiling_dir,
                                       self._file_name_op_intermediate_detail.format(self._rank_id))
        with os.fdopen(os.open(op_op_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as op_file:
            csv_writer = csv.writer(op_file)
            csv_writer.writerow(self._op_intermediate_op_header)

            for op_name, op_name_time_info in op_intermediate_detail.items():
                op_info = [
                    op_name, round(op_name_time_info[1] / op_name_time_info[0], self._ms_decimal_digits)
                ]
                csv_writer.writerow(op_info)
        os.chmod(op_op_file_path, stat.S_IREAD | stat.S_IWRITE)

    def parser_pynative_op_type(self):
        """Parse pynative op intermediate type."""
        timeline_list = self.get_timeline_data(None)
        # key:op type, value:[op count, total op execution time, op execution time percent]
        op_type_list = {}
        for timeline in timeline_list:
            type_name = timeline[0].split('/')[-1].split('-')[0]
            op_type = op_type_list.get(type_name)
            if not op_type:
                op_type = [0, 0, 0]
                op_type_list[type_name] = op_type
            op_type[0] = op_type[0] + 1
            op_type[1] = op_type[1] + float(timeline[3])

        sum_avg_time = 0
        for _, op_type in op_type_list.items():
            op_type[1] = op_type[1] / op_type[0]
            sum_avg_time = sum_avg_time + op_type[1]

        if sum_avg_time <= 0:
            logger.error("Operator time must be greater than 0.")
            return
        for _, op_type in op_type_list.items():
            op_type[2] = op_type[1] / sum_avg_time

        op_type_file_path = os.path.join(self._profiling_dir,
                                         self._file_name_op_intermediate_type.format(self._rank_id))
        with os.fdopen(os.open(op_type_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as type_file:
            csv_writer = csv.writer(type_file)
            csv_writer.writerow(self._op_intermediate_type_header)

            for op_type, op_type_time_info in op_type_list.items():
                type_info = [
                    op_type, op_type_time_info[1], op_type_time_info[0],
                    round((op_type_time_info[1] / sum_avg_time) * 100, self._percent_decimal_digits)
                ]
                csv_writer.writerow(type_info)
        os.chmod(op_type_file_path, stat.S_IREAD | stat.S_IWRITE)
