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
"""The OpAnalyser."""
import json
import os
from collections import defaultdict
from mindspore.profiler.common.path_manager import PathManager


class OpAnalyser:
    """
    The parser for parsing framework files.

    Note:
        This parser only supports CPU and GPU devices.

    Args:
        output_path (str): The profiling path which should contain GPU profiling data.
        dev_id (str): The device ID.
    """

    def __init__(self, output_path, dev_id, op_names=None):
        """The parser for parsing framework files."""
        self._dev_id = dev_id
        self._output_path = PathManager.get_real_path(output_path)
        self.op_names = op_names
        self.op_name = ''
        self.framework_list = []
        self.op_detail = {}
        self.operation_info = {}
        self.activity_info_dir = []
        self.framework_info_dir = []
        self.cpu_detail_info_dir = []
        self.gpu_op_type_info_dir = []
        self.op_execute_times = {}
        self.op_step_shape_info = defaultdict(list)

    def parse(self):
        """Parse op performance data."""
        self.get_device_target_filename()
        self.get_framework_summary()
        self.get_cpu_op_detail_info()
        self.get_activity_op_info()
        if isinstance(self.op_names, str):
            self.combine_performance_data(self.op_names)
        elif isinstance(self.op_names, list):
            for op_name in self.op_names:
                self.combine_performance_data(op_name)
        self.operation_info["device_id"] = self._dev_id
        return json.dumps(self.operation_info)

    def get_framework_summary(self):
        """Get framework data."""
        for filename in self.framework_info_dir:
            op_side = filename.split('_')[0]
            framework_file_path = os.path.join(self._output_path, filename)
            PathManager.check_input_file_path(framework_file_path)
            PathManager.check_directory_path_readable(framework_file_path)
            with open(framework_file_path, 'r', encoding='utf-8') as f_obj:
                framework_info = f_obj.readlines()
            for line_info in framework_info:
                line_info = line_info.strip(' ').strip('\n').split(';')
                # line_info[0]: op_type, line_info[1]: op_name, line_info[2]: graph_id, line_info[3]: input_shape;
                input_shape = line_info[3:]
                item = [line_info[0], line_info[1], input_shape, op_side]
                if not self.op_step_shape_info.get(line_info[1]):
                    self.op_step_shape_info[line_info[1]].append(op_side)
                self.op_step_shape_info[line_info[1]].append(input_shape)
                if item not in self.framework_list:
                    self.framework_list.append(item)

    def get_cpu_op_detail_info(self):
        """Get cpu operators detail data."""
        for filename in self.cpu_detail_info_dir:
            op_side = filename.split('_')[0]
            op_detail_file_path = os.path.join(self._output_path, filename)
            PathManager.check_input_file_path(op_detail_file_path)
            PathManager.check_directory_path_readable(op_detail_file_path)
            with open(op_detail_file_path, 'r', encoding='utf-8') as f_obj:
                op_detail_info = f_obj.readlines()
            for line_info in op_detail_info[1:]:
                line_info = line_info.strip(' ').strip('\n').split(',')
                if not self.op_detail.get(line_info[2]):
                    # line_info[4]: op_occurrences, line_info[5]: op_detail_time(us), line_info[6]: op_avg_time(us);
                    self.op_detail[line_info[2]] = [float(line_info[4]), float(line_info[5]),
                                                    float(line_info[6]), op_side]

    def get_execute_times(self):
        """Get gpu operators execute times."""
        if self.gpu_op_type_info_dir:
            gpu_op_type_file_path = os.path.join(self._output_path, self.gpu_op_type_info_dir[0])
            PathManager.check_input_file_path(gpu_op_type_file_path)
            PathManager.check_directory_path_readable(gpu_op_type_file_path)
            with open(gpu_op_type_file_path, 'r', encoding='utf-8') as fp:
                op_type_info = fp.readlines()
                for line_info in op_type_info[1:]:
                    line_info = line_info.strip(' ').strip('\n').split(',')
                    self.op_execute_times[line_info[0]] = line_info[1]

    def get_activity_op_info(self):
        """Get op detail data."""
        all_file = os.listdir(self._output_path)
        for file_name in all_file:
            if file_name.startswith('gpu_op_type') and file_name.endswith(f'{self._dev_id}.csv'):
                self.gpu_op_type_info_dir.append(file_name)
        if not self.gpu_op_type_info_dir and self.activity_info_dir:
            raise RuntimeError('The output file <%s> is not found.' % self.gpu_op_type_info_dir)
        self.get_execute_times()
        for filename in self.activity_info_dir:
            op_side = filename.split('_')[0]
            activity_file_path = os.path.join(self._output_path, filename)
            PathManager.check_input_file_path(activity_file_path)
            PathManager.check_directory_path_readable(activity_file_path)
            with open(activity_file_path, 'r', encoding='utf-8') as file:
                activity_info = file.readlines()
            for line_info in activity_info[1:]:
                line_info = line_info.strip(' ').strip('\n').replace(', ', ';').split(',')
                op_name = line_info[2].split('/')[-1]
                # op_name: xxx-opx
                op_type = op_name.split('-')[0]
                op_value = self.op_execute_times.get(op_type)
                if op_value is not None and op_value != '':
                    try:
                        op_occurrences = int(op_value)
                    except (ValueError, TypeError):
                        op_occurrences = 1
                else:
                    op_occurrences = 1

                op_total_time = float(line_info[-4])
                if not self.op_detail.get(op_name):
                    # line_info[4]: op_occurrences, line_info[5]: op_detail_time(us), line_info[6]: op_avg_time(us);
                    if op_occurrences > 0:
                        avg_time = round(op_total_time / op_occurrences, 4)
                    else:
                        avg_time = 0
                    self.op_detail[op_name] = [
                        op_occurrences, op_total_time, avg_time, op_side
                    ]
                else:
                    self.op_detail.get(op_name)[1] += op_total_time
                    self.op_detail.get(op_name)[2] = self.op_detail.get(op_name)[1] / self.op_detail.get(op_name)[0]
                    self.op_detail[op_name] = [
                        self.op_detail.get(op_name)[0],
                        round(self.op_detail.get(op_name)[1], 4),
                        round(self.op_detail.get(op_name)[2], 4), op_side
                    ]

    def combine_performance_data(self, op_name):
        """Combine operator detail info with framework info."""
        unique_op_info = []
        op_shape_dict = {}
        operation_info = {}
        factor = 1000  # convert time unit from ms to us.
        for line_info in self.framework_list:
            op_detail = self.op_detail.get(line_info[1])
            if not op_detail:
                continue
            if op_name in line_info and line_info[3] == op_detail[3]:
                op_side = line_info[3]
                op_shape = '[{}]{}'.format(op_side, ','.join(line_info[2]))
                op_occurrences = int(op_detail[0])
                op_total_time = float(op_detail[1])
                op_avg_time = float(op_detail[2])
                if op_shape in op_shape_dict:
                    # Classify according to the operator information of the same shape.
                    op_shape_dict.get(op_shape)[0] += op_occurrences
                    op_shape_dict.get(op_shape)[1] += op_total_time
                    if op_shape_dict.get(op_shape)[0] > 0:
                        op_shape_dict.get(op_shape)[2] = op_shape_dict.get(op_shape)[1] / op_shape_dict.get(op_shape)[0]
                    else:
                        op_shape_dict.get(op_shape)[2] = 0
                    op_shape_dict[op_shape] = [
                        op_shape_dict.get(op_shape)[0], round(op_shape_dict.get(op_shape)[1], 4),
                        round(op_shape_dict.get(op_shape)[2], 4), op_side
                    ]
                else:
                    op_shape_dict[op_shape] = [op_occurrences, op_total_time, op_avg_time, op_side]

        for input_shape in op_shape_dict:
            # 0: op_occurrences, 1: op_total_time, 2: op_avg_time, 3: op_side
            operation_info['op_side'] = op_shape_dict.get(input_shape)[3]
            operation_info['input_shape'] = input_shape.strip('[').split(']')[-1]
            operation_info['op_occurrences'] = op_shape_dict.get(input_shape)[0]
            if operation_info.get('op_side') == 'cpu':
                operation_info['op_total_time(us)'] = round(op_shape_dict.get(input_shape)[1] * factor, 4)
                operation_info['op_avg_time(us)'] = round(op_shape_dict.get(input_shape)[2] * factor, 4)
            else:
                operation_info['op_total_time(us)'] = op_shape_dict.get(input_shape)[1]
                operation_info['op_avg_time(us)'] = op_shape_dict.get(input_shape)[2]
            unique_op_info.append(operation_info)
            operation_info = {}

        if unique_op_info:
            self.operation_info[op_name] = unique_op_info
        else:
            raise RuntimeError(f'The information of <{op_name}> is not found. Please verify that the operator name is'
                               ' correct or the operator is used in the network.')

    def get_device_target_filename(self):
        """Get device target filename."""
        gpu_framework_file = f'gpu_framework_{self._dev_id}.txt'
        cpu_framework_file = f'cpu_framework_{self._dev_id}.txt'
        gpu_activity_file = f'gpu_activity_data_{self._dev_id}.csv'
        cpu_op_detail_file = f'cpu_op_detail_info_{self._dev_id}.csv'
        all_file = os.listdir(self._output_path)
        if not all_file:
            raise RuntimeError('No profiler file is found in the path <%s>. '
                               'Check whether the profiler path is correct.' % self._output_path)
        if gpu_activity_file in all_file and gpu_framework_file not in all_file:
            raise RuntimeError('The output file <%s> is not found.' % gpu_framework_file)
        if cpu_op_detail_file in all_file and cpu_framework_file not in all_file:
            raise RuntimeError('The output file <%s> is not found.' % cpu_framework_file)
        if gpu_framework_file in all_file and gpu_activity_file not in all_file:
            raise RuntimeError('The output file <%s> is not found.' % gpu_activity_file)
        if cpu_framework_file in all_file and cpu_op_detail_file not in all_file:
            raise RuntimeError('The output file <%s> is not found.' % cpu_op_detail_file)
        if gpu_activity_file not in all_file and cpu_op_detail_file not in all_file:
            raise RuntimeError(f'The profiling data of this card which device_id is equal to {self._dev_id} does not'
                               ' exist. Check whether device_id is correct.')
        for file_name in all_file:
            if file_name.endswith(f'activity_data_{self._dev_id}.csv'):
                self.activity_info_dir.append(file_name)
            if file_name.endswith(f'framework_{self._dev_id}.txt'):
                self.framework_info_dir.append(file_name)
            if file_name.startswith('cpu_op_detail') and file_name.endswith(f'{self._dev_id}.csv'):
                self.cpu_detail_info_dir.append(file_name)
