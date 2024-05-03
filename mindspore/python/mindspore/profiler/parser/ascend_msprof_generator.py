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
"""msprof data generate model"""
import csv
import glob
import numpy as np
from mindspore.profiler.common.util import get_newest_file


class AscendMsprofDataGenerator:
    """Generate ascend data from files."""

    _mindspore_model_id = 4294967295

    def __init__(self, mindstudio_profiler_output):
        self.mindstudio_profiler_output = mindstudio_profiler_output
        self.op_summary = None
        self.op_statistic = None
        self.steptrace = []
        self.steptrace_model = []

        self.op_summary_type = [
            ('Model ID', float),
            ('Task ID', int),
            ('Stream ID', int),
            ('Op Name', object),
            ('Op Type', object),
            ('Task Type', object),
            ('Task Start Time', object),
            ('Task Duration', float),
            ('Task Wait Time', float),
            ('Input Shapes', object),
            ('Input Data Types', object),
            ('Input Formats', object),
            ('Output Shapes', object),
            ('Output Data Types', object),
            ('Output Formats', object),
            ('Task Start Time(us)', object)
        ]

        self.op_statistic_type = [
            ('Op Type', object),
            ('Count', int),
            ('Total Time', float),
        ]

        self.steptrace_type = [
            ('Iteration ID', int),
            ('FP Start', float),
            ('BP End', float),
            ('Iteration End', float),
            ('Iteration Time', float),
            ('FP to BP Time', float),
            ('Iteration Refresh', float),
            ('Data Aug Bound', float),
            ('Model ID', float),
        ]

    def parse(self):
        """read msprof data generate DataFrame data"""
        self._read_op_summary()

        self._read_op_statistic()

        self._read_steptrace()

        self.steptrace_model = self.steptrace[self.steptrace['Model ID'] == self._mindspore_model_id]

        self.steptrace = self.steptrace[self.steptrace['Model ID'] != self._mindspore_model_id]

        result = (self.op_summary, self.op_statistic, self.steptrace, self.steptrace_model)

        return result

    def _read_op_summary(self):
        """read op summary to memory"""
        op_summary = []
        op_summary_name = fr'{self.mindstudio_profiler_output}/op_summary_*.csv'
        op_summary_files = glob.glob(op_summary_name)
        if not op_summary_files:
            return
        op_summary_file = get_newest_file(op_summary_files)[0]
        with open(op_summary_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                vector_fops = row.get('vector_fops', None)
                cube_fops = row.get('cube_fops', None)
                aiv_vector_fops = row.get('aiv_vector_fops', None)
                aic_cube_fops = row.get('aic_cube_fops', None)

                new_row = [row.get('Model ID'), row.get('Task ID'), row.get('Stream ID'), row.get('Op Name'),
                           row.get('OP Type'), row.get('Task Type'), row.get('Task Start Time(us)'),
                           row.get('Task Duration(us)'), row.get('Task Wait Time(us)'), row.get('Input Shapes'),
                           row.get('Input Data Types'), row.get('Input Formats'), row.get('Output Shapes'),
                           row.get('Output Data Types'), row.get('Output Formats'), '0.000']
                if vector_fops is not None and cube_fops is not None:
                    new_row.append(vector_fops)
                    new_row.append(cube_fops)
                elif aiv_vector_fops is not None and aic_cube_fops is not None:
                    new_row.append(aiv_vector_fops)
                    new_row.append(aic_cube_fops)
                op_summary.append(tuple(['0' if d == 'N/A' else d for d in new_row]))

        if op_summary and len(op_summary[0]) > len(self.op_summary_type):
            self.op_summary_type.extend([
                ('vector_fops', float),
                ('cube_fops', float)
            ])

        op_summary_dt = np.dtype(self.op_summary_type)

        self.op_summary = np.array(op_summary, dtype=op_summary_dt)
        high_acc_time = self.op_summary['Task Start Time'].copy()
        self.op_summary['Task Start Time(us)'] = high_acc_time
        self.op_summary['Task Start Time'] = self.op_summary['Task Start Time'].astype(float) * 1e-3
        self.op_summary['Task Duration'] = self.op_summary['Task Duration'] * 1e-3
        self.op_summary['Task Wait Time'] = self.op_summary['Task Wait Time'] * 1e-3

    def _read_op_statistic(self):
        """read op statistic to memory"""
        op_statistic = []
        op_statistic_name = fr'{self.mindstudio_profiler_output}/op_statistic_*.csv'
        op_statistic_files = glob.glob(op_statistic_name)
        if not op_statistic_files:
            return
        op_statistic_file = get_newest_file(op_statistic_files)[0]
        with open(op_statistic_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                new_row = (
                    row.get('OP Type'),
                    row.get('Count'),
                    row.get('Total Time(us)'),
                )
                new_row = tuple(['0' if d == 'N/A' else d for d in new_row])
                op_statistic.append(new_row)
        if not op_statistic:
            return
        op_statistic_dt = np.dtype(self.op_statistic_type)
        self.op_statistic = np.array(op_statistic, dtype=op_statistic_dt)
        self.op_statistic['Total Time'] *= 1e-3

    def _read_steptrace(self):
        """read steptrace to memory"""
        step_trace = []
        step_trace_name = fr'{self.mindstudio_profiler_output}/step_trace_*.csv'
        step_trace_file_list = get_newest_file(glob.glob(step_trace_name))
        for step_trace_file in step_trace_file_list:
            with open(step_trace_file, newline='') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
                for row in reader:
                    new_row = [
                        row.get('Iteration ID'),
                        row.get('FP Start(us)'),
                        row.get('BP End(us)'),
                        row.get('Iteration End(us)'),
                        row.get('Iteration Time(us)'),
                        row.get('FP to BP Time(us)'),
                        row.get('Iteration Refresh(us)'),
                        row.get('Data Aug Bound(us)'),
                        row.get('Model ID'),
                    ]
                    step_trace.append(tuple(['0' if i == 'N/A' else i for i in new_row]))
            break

        steptrace_dt = np.dtype(self.steptrace_type)

        self.steptrace = np.array(step_trace, dtype=steptrace_dt)
        self.steptrace['FP Start'] = self.steptrace['FP Start'] * 1e-3
        self.steptrace['BP End'] = self.steptrace['BP End'] * 1e-3
        self.steptrace['Iteration End'] = self.steptrace['Iteration End'] * 1e-3
        self.steptrace['Iteration Time'] = self.steptrace['Iteration Time'] * 1e-3
        self.steptrace['FP to BP Time'] = self.steptrace['FP to BP Time'] * 1e-3
        self.steptrace['Iteration Refresh'] = self.steptrace['Iteration Refresh'] * 1e-3
        self.steptrace['Data Aug Bound'] = self.steptrace['Data Aug Bound'] * 1e-3
