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
import fnmatch
import os

import numpy as np


class AscendMsprofDataGenerator:
    """Generate ascend data from files."""

    def __init__(self, source_path):
        self.source_path = source_path
        self.op_summary = None
        self.op_statistic = None
        self.steptrace = []

        self.invalid_index = 1000
        self.op_summary_basis_name = {
            'Model ID': {'index': self.invalid_index, 'dtype': ('Model ID', int)},
            'Task ID': {'index': self.invalid_index, 'dtype': ('Task ID', int)},
            'Stream ID': {'index': self.invalid_index, 'dtype': ('Stream ID', int)},
            'Op Name': {'index': self.invalid_index, 'dtype': ('Op Name', object)},
            'OP Type': {'index': self.invalid_index, 'dtype': ('Op Type', object)},
            'Task Type': {'index': self.invalid_index, 'dtype': ('Task Type', object)},
            'Task Start Time(us)': {'index': self.invalid_index, 'dtype': ('Task Start Time', float)},
            'Task Duration(us)': {'index': self.invalid_index, 'dtype': ('Task Duration', float)},
            'Task Wait Time(us)': {'index': self.invalid_index, 'dtype': ('Task Wait Time', float)},
            'Input Shapes': {'index': self.invalid_index, 'dtype': ('Input Shapes', object)},
            'Input Data Types': {'index': self.invalid_index, 'dtype': ('Input Data Types', object)},
            'Input Formats': {'index': self.invalid_index, 'dtype': ('Input Formats', object)},
            'Output Shapes': {'index': self.invalid_index, 'dtype': ('Output Shapes', object)},
            'Output Data Types': {'index': self.invalid_index, 'dtype': ('Output Data Types', object)},
            'Output Formats': {'index': self.invalid_index, 'dtype': ('Output Formats', object)},
        }
        self.op_summary_extend_name = {
            'vector_fops': {'index': self.invalid_index, 'dtype': ('vector_fops', float)},
            'cube_fops': {'index': self.invalid_index, 'dtype': ('cube_fops', float)},
        }
        self.op_summary_name = None

        self.op_statistic_name = {
            'OP Type': {'index': self.invalid_index, 'dtype': ('Op Type', object)},
            'Count': {'index': self.invalid_index, 'dtype': ('Count', int)},
            'Total Time(us)': {'index': self.invalid_index, 'dtype': ('Total Time', float)},
        }

        self.steptrace_name = {
            'Iteration ID': {'index': self.invalid_index, 'dtype': ('Iteration ID', int)},
            'FP Start(us)': {'index': self.invalid_index, 'dtype': ('FP Start', float)},
            'BP End(us)': {'index': self.invalid_index, 'dtype': ('BP End', float)},
            'Iteration End(us)': {'index': self.invalid_index, 'dtype': ('Iteration End', float)},
            'Iteration Time(us)': {'index': self.invalid_index, 'dtype': ('Iteration Time', float)},
            'FP to BP Time(us)': {'index': self.invalid_index, 'dtype': ('FP to BP Time', float)},
            'Iteration Refresh(us)': {'index': self.invalid_index, 'dtype': ('Iteration Refresh', float)},
            'Data Aug Bound(us)': {'index': self.invalid_index, 'dtype': ('Data Aug Bound', float)},
            'Model ID': {'index': self.invalid_index, 'dtype': ('Model ID', int)},
        }

    @staticmethod
    def find_files(directory, pattern):
        """Find files with feature 'pattern' from the directory"""

        for root, _, files in os.walk(directory):
            files.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)))
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    yield filename

    def link_index_with_name(self, header, name_dict):
        """link index with row name"""
        for index, value in enumerate(header):
            if value in name_dict:
                name_dict[value]['index'] = index
        for value in name_dict.values():
            if value['index'] == self.invalid_index:
                return False
        return True

    def parse(self):
        """read msprof data generate DataFrame data"""
        self._read_op_summary()

        self._read_op_statistic()

        self._read_steptrace()

        return self.op_summary, self.op_statistic, self.steptrace

    def _read_op_summary(self):
        """read op summary to memory"""
        op_summary = []
        for file in self.find_files(self.source_path, "op_summary*.csv"):
            with open(file, newline='') as csvfile:
                iteration = int(file.split('_')[-1].split('.')[0])
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                header = next(reader)
                self.link_index_with_name(header, self.op_summary_basis_name)
                extend_flag = self.link_index_with_name(header, self.op_summary_extend_name)
                if extend_flag:
                    self.op_summary_name = {**self.op_summary_basis_name, **self.op_summary_extend_name}
                else:
                    self.op_summary_name = self.op_summary_basis_name
                self.op_summary_name['Iteration ID'] = {'index': -1, 'dtype': ('Iteration ID', object)}
                for row in reader:
                    row = [row[index.get('index')] for index in self.op_summary_name.values()]
                    row[self.op_summary_name['Iteration ID']['index']] = iteration
                    row = ['0' if i == 'N/A' else i for i in row]
                    op_summary.append(tuple(row))

        op_summary_dt = np.dtype([value['dtype'] for value in self.op_summary_name.values()])

        self.op_summary = np.array(op_summary, dtype=op_summary_dt)
        self.op_summary['Task Start Time'] = self.op_summary['Task Start Time'] * 1e-3
        self.op_summary['Task Duration'] = self.op_summary['Task Duration'] * 1e-3
        self.op_summary['Task Wait Time'] = self.op_summary['Task Wait Time'] * 1e-3

    def _read_op_statistic(self):
        """read op statistic to memory"""
        op_statistic = []
        for file in self.find_files(self.source_path, "op_statistic*.csv"):
            with open(file, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                header = next(reader)
                self.link_index_with_name(header, self.op_statistic_name)
                for row in reader:
                    row = [row[index.get('index')] for index in self.op_statistic_name.values()]
                    row = ['0' if i == 'N/A' else i for i in row]
                    op_statistic.append(tuple(row))

        op_statistic_dt = np.dtype([value['dtype'] for value in self.op_statistic_name.values()])
        self.op_statistic = np.array(op_statistic, dtype=op_statistic_dt)
        self.op_statistic['Total Time'] = self.op_statistic['Total Time'] * 1e-3

    def _read_steptrace(self):
        """read steptrace to memory"""
        steptrace = []
        header = []
        for file in self.find_files(self.source_path, "step_trace*.csv"):
            with open(file, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                header = next(reader)
                self.link_index_with_name(header, self.steptrace_name)
                for row in reader:
                    rows = [row[index.get('index')] for index in self.steptrace_name.values()]
                    if row[9:]:
                        rows.extend(row[9:len(header)])
                    if len(rows) < len(header):
                        rows.extend([0 for _ in range(len(header) - len(rows))])
                    rows = ['0' if i == 'N/A' else i for i in rows]
                    steptrace.append(tuple(rows))
            break

        hccl_data = self.op_summary[(self.op_summary['Task Type'] == 'HCCL') & (self.op_summary['Iteration ID'] == 1)][
            ['Task ID', 'Stream ID', 'Op Name']]

        index = len(self.steptrace_name)
        for name in hccl_data:
            if index >= len(header):
                break
            name = f"stream_{name['Stream ID']}_{name['Task ID']}_{name['Op Name']}"
            self.steptrace_name[name] = {'index': index, 'dtype': (name, float)}
            index += 1
            self.steptrace_name[f'{name} duration'] = {'index': index, 'dtype': (f'{name} duration', float)}
            index += 1

        for i in range(len(self.steptrace_name), len(header), 2):
            name = f'hccl_{i}'
            self.steptrace_name[name] = {'index': i, 'dtype': (name, float)}
            self.steptrace_name[f'{name} duration'] = {'index': i+1, 'dtype': (f'{name} duration', float)}

        steptrace_dt = np.dtype([value['dtype'] for value in self.steptrace_name.values()])

        self.steptrace = np.array(steptrace, dtype=steptrace_dt)
        self.steptrace['FP Start'] = self.steptrace['FP Start'] * 1e-3
        self.steptrace['BP End'] = self.steptrace['BP End'] * 1e-3
        self.steptrace['Iteration End'] = self.steptrace['Iteration End'] * 1e-3
        self.steptrace['Iteration Time'] = self.steptrace['Iteration Time'] * 1e-3
        self.steptrace['FP to BP Time'] = self.steptrace['FP to BP Time'] * 1e-3
        self.steptrace['Iteration Refresh'] = self.steptrace['Iteration Refresh'] * 1e-3
        self.steptrace['Data Aug Bound'] = self.steptrace['Data Aug Bound'] * 1e-3

        for name in self.steptrace.dtype.names[9:]:
            self.steptrace[name] = self.steptrace[name] * 1e-3
