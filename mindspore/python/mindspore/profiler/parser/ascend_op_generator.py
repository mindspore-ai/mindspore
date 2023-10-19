# Copyright 2023 Huawei Technologies Co., Ltd
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
"""op analyse model"""
import csv
import json
import logging
import os
import stat

import numpy as np
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException


class AscendOPGenerator:
    """Generate ascend op data from DataFrame."""

    def __init__(self, op_summary, op_statistic, dynamic_status=False):
        self.op_summary = op_summary
        self.op_statistic = op_statistic
        self.dynamic_status = dynamic_status
        self.op_detail = None
        self.op_type = None
        self.aicpu_detail = None
        self.framework_raw = None
        self.output_timeline_data = None

        self.op_detail_dt = np.dtype(
            [('full_op_name', object), ('task_duration', float), ('execution_frequency', int), ('task_type', object)])

        self.op_type_dt = np.dtype(
            [('op_type', object), ('total_time', float), ('execution_frequency', int), ('percent', float)])

        self.aicpu_detail_dt = np.dtype(
            [('serial_number', int), ('op_type', object), ('total_time', float), ('dispatch_time', float),
             ('execution_time', float), ('run_start', float), ('run_end', float)])

        self.framwork_raw_dt = np.dtype(
            [('task_id', int), ('stream_id', int), ('block_dim', int), ('full_op_name', object), ('op_name', object),
             ('op_type', object), ('subgraph', object), ('op_info', object), ('model_id', int),
             ('kernel_type', object)])

    def parse(self):
        """
        Analyse op summary op statistic generate op data.
        """

        # aicore intermediation detail
        self.op_detail = self._parse_op_detail(self.op_summary)

        # aicore intermediation type
        self.op_type = self._parse_op_type(self.op_statistic)

        # aicpu_intermediation
        self.aicpu_detail = self._parse_aicpu_detail(self.op_summary)

        # framwork_raw
        self.framework_raw = self._parse_framework_raw(self.op_summary)

        self.output_timeline_data = self.op_summary[self.op_summary['Task Type'] == 'AI_CORE'][
            ['Op Name', 'Stream ID', 'Task Start Time', 'Task Duration']]

    def write(self, aicore_intermediate_detail_path, aicore_intermediate_type_path, aicpu_intermediate_detail_path,
              framework_raw_path, output_timeline_data_path):
        """
        Write the op_intermediate_detail.csv op_intermediate_type.csv aicpu_intermediate.csv and framework_raw.csv.

        Args:
            aicore_intermediate_detail_path(str): op_intermediate_detail.csv path.
            aicore_intermediate_type_path(str): op_intermediate_type.csv path.
            aicpu_intermediate_detail_path(str): aicpu_intermediate.csv path.
            framework_raw_path: framework_raw.csv path
            output_timeline_data_path : output_timeline_data.txt path
        """
        # aicore intermediation detail
        if self.op_detail.shape[0] != 0:
            try:
                with os.fdopen(os.open(aicore_intermediate_detail_path,
                                       os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IWUSR | stat.S_IRUSR),
                               'w') as aicore_detail:
                    writer = csv.writer(aicore_detail)
                    writer.writerow(self.op_detail.dtype.names)
                    writer.writerows(self.op_detail.tolist())
            except (IOError, OSError) as err:
                logging.critical('Errot occurred when write aicore detail file: %s', err)
                raise ProfilerIOException() from err
            if os.path.exists(aicore_intermediate_detail_path):
                os.chmod(aicore_intermediate_detail_path, stat.S_IREAD | stat.S_IWRITE)

        # aicore intermediation type
        if self.op_type.shape[0] != 0:
            try:
                with os.fdopen(os.open(aicore_intermediate_type_path,
                                       os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IWUSR | stat.S_IRUSR),
                               'w') as aicore_type:
                    writer = csv.writer(aicore_type)
                    writer.writerow(self.op_type.dtype.names)
                    writer.writerows(self.op_type.tolist())
            except (IOError, OSError) as err:
                logging.critical('Errot occurred when write aicore type file: %s', err)
                raise ProfilerIOException() from err
            if os.path.exists(aicore_intermediate_type_path):
                os.chmod(aicore_intermediate_type_path, stat.S_IREAD | stat.S_IWRITE)

        # aicpu_intermediation
        if self.aicpu_detail.shape[0] != 0:
            try:
                with os.fdopen(os.open(aicpu_intermediate_detail_path,
                                       os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IWUSR | stat.S_IRUSR),
                               'w') as aicpu_type:
                    writer = csv.writer(aicpu_type)
                    writer.writerow(self.aicpu_detail.dtype.names)
                    writer.writerows(self.aicpu_detail.tolist())
            except (IOError, OSError) as err:
                logging.critical('Errot occurred when write aicpu detail file: %s', err)
                raise ProfilerIOException() from err
            if os.path.exists(aicpu_intermediate_detail_path):
                os.chmod(aicpu_intermediate_detail_path, stat.S_IREAD | stat.S_IWRITE)

        # framwork_raw
        if self.framework_raw.shape[0] != 0:
            try:
                with os.fdopen(os.open(framework_raw_path,
                                       os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IWUSR | stat.S_IRUSR),
                               'w') as framework:
                    writer = csv.writer(framework)
                    writer.writerow(self.framework_raw.dtype.names)
                    writer.writerows(self.framework_raw.tolist())
            except (IOError, OSError) as err:
                logging.critical('Errot occurred when write framework file: %s', err)
                raise ProfilerIOException() from err
            if os.path.exists(framework_raw_path):
                os.chmod(framework_raw_path, stat.S_IREAD | stat.S_IWRITE)

        # output_timeline_data
        if self.output_timeline_data.shape[0] != 0 and output_timeline_data_path:
            try:
                with os.fdopen(os.open(output_timeline_data_path,
                                       os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IWUSR | stat.S_IRUSR),
                               'w') as output_timeline_data:
                    writer = csv.writer(output_timeline_data)
                    writer.writerow(['op_name', 'stream_id', 'start_time(us)', 'duration(ms)'])
                    writer.writerows(self.output_timeline_data.tolist())
            except (IOError, OSError) as err:
                logging.critical('Error occurred when write output timeline data file: %s', err)
                raise ProfilerIOException() from err
            if os.path.exists(aicpu_intermediate_detail_path):
                os.chmod(aicpu_intermediate_detail_path, stat.S_IREAD | stat.S_IWRITE)

    def _parse_op_detail(self, op_summary):
        """
        Analyse op summary generate op detail data.

        Args:
            op_summary(DataFrame): op summary data.
        """
        groups, index, inverse, counts = np.unique(op_summary['Op Name'], return_index=True, return_inverse=True,
                                                   return_counts=True)

        op_detail = np.empty((len(groups),), dtype=self.op_detail_dt)
        op_detail['full_op_name'] = groups
        op_detail['task_type'] = op_summary[index]['Task Type']
        nonzero_duration = np.bincount(inverse) != 0
        op_detail['task_duration'] = np.where(nonzero_duration,
                                              np.bincount(inverse, weights=op_summary['Task Duration']) / np.bincount(
                                                  inverse), 0)
        op_detail['execution_frequency'] = counts

        return op_detail

    def _parse_op_type(self, op_statistic):
        """
        Analyse op statistic generate op type data.

        Args:
            op_statistic(DataFrame): op statistic data.
        """

        groups, _, inverse, _ = np.unique(op_statistic['Op Type'], return_index=True, return_inverse=True,
                                          return_counts=True)

        op_type = np.empty((len(groups),), dtype=self.op_type_dt)
        op_type['op_type'] = groups
        op_type['total_time'] = np.bincount(inverse, weights=op_statistic['Total Time'])
        op_type['execution_frequency'] = np.bincount(inverse, weights=op_statistic['Count'])
        op_type['percent'] = op_type['total_time'] / np.sum(op_statistic['Total Time']) if np.sum(
            op_statistic['Total Time']) != 0 else 0

        return op_type

    def _parse_aicpu_detail(self, op_summary):
        """
        Analyse op summary generate aicpu detail data.

        Args:
            op_summary(DataFrame): op summary data.
        """

        op_summary = op_summary[op_summary['Task Type'] == 'AI_CPU']

        aicpu_detail = np.empty((len(op_summary),), dtype=self.aicpu_detail_dt)

        aicpu_detail['serial_number'] = [i for i in range(1, op_summary.shape[0] + 1)]
        aicpu_detail['op_type'] = op_summary['Op Type']
        aicpu_detail['total_time'] = op_summary['Task Duration'] + op_summary['Task Wait Time']
        aicpu_detail['dispatch_time'] = op_summary['Task Wait Time']
        aicpu_detail['execution_time'] = op_summary['Task Duration']
        aicpu_detail['run_start'] = op_summary['Task Start Time']
        aicpu_detail['run_end'] = aicpu_detail['run_start'] + aicpu_detail['total_time']

        return aicpu_detail

    def _parse_framework_raw(self, op_summary):
        """
        Analyse op summary generate op framework data.

        Args:
            op_summary(DataFrame): op summary data.
        """

        def op_info_analyse(row):
            """generate op info data"""
            input_shapes = row['Input Shapes'].replace('"', '').split(';')
            input_data_types = row['Input Data Types'].replace('_', '').split(';')
            input_formats = row['Input Formats'].replace('_', '').split(';')
            output_shapes = row['Output Shapes'].replace('"', '').split(';')
            output_data_types = row['Output Data Types'].replace('_', '').split(';')
            output_formats = row['Output Formats'].replace('_', '').split(';')
            op_info = {}
            if isinstance(input_shapes, list) and len(input_shapes) >= 1 and input_shapes[0] != '':
                input_size = len(input_shapes)
                for i in range(input_size):
                    op_info[f'Input_{i}'] = {
                        'format': input_formats[i],
                        'data_type': input_data_types[i],
                        'shape': input_shapes[i]
                    }
            if isinstance(output_shapes, list) and len(output_shapes) >= 1 and output_shapes[0] != '':
                output_size = len(output_shapes)
                for i in range(output_size):
                    op_info[f'Output_{i}'] = {
                        'format': output_formats[i],
                        'data_type': output_data_types[i],
                        'shape': output_shapes[i]
                    }
            return json.dumps(op_info)

        if self.dynamic_status:
            index = list(range(op_summary.shape[0]))
        else:
            _, index, _, _ = np.unique(op_summary['Op Name'], return_index=True, return_inverse=True,
                                       return_counts=True)
        framwork_raw = np.empty((len(index),), dtype=self.framwork_raw_dt)

        framwork_raw['task_id'] = op_summary[index]['Task ID']
        framwork_raw['stream_id'] = op_summary[index]['Stream ID']
        framwork_raw['full_op_name'] = op_summary[index]['Op Name']
        framwork_raw['op_name'] = [x[-1] for x in np.char.split(op_summary[index]['Op Name'].astype(str), sep='/')]
        framwork_raw['op_type'] = op_summary[index]['Op Type']
        framwork_raw['subgraph'] = [x[0] for x in np.char.split(op_summary[index]['Op Name'].astype(str), sep='/')]
        framwork_raw['op_info'] = [op_info_analyse(x) for x in op_summary[index]]
        framwork_raw['model_id'] = op_summary[index]['Model ID']
        framwork_raw['kernel_type'] = op_summary[index]['Task Type']

        return framwork_raw
