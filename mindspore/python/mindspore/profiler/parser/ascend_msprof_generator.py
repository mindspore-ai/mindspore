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
        self.steptrace = None
        self.op_summary_dt = np.dtype(
            [('Model Name', object), ('Model ID', int), ('Task ID', int), ('Stream ID', int), ('Infer ID', int),
             ('Op Name', object), ('Op Type', object), ('Task Type', object), ('Task Start Time', float),
             ('Task Duration', float), ('Task Wait Time', float), ('Block Dim', float), ('Input Shapes', object),
             ('Input Data Types', object), ('Input Formats', object), ('Output Shapes', object),
             ('Output Data Types', object), ('Output Formats', object), ('aicore_time', float), ('total_cycles', float),
             ('mac_fp16_ratio', float), ('mac_int8_ratio', float), ('vec_fp32_ratio', float), ('vec_fp16_ratio', float),
             ('vec_int32_ratio', float), ('vec_misc_ratio', float), ('cube_fops', float), ('vector_fops', float),
             ('Iteration ID', int)])
        self.op_statistic_dt = np.dtype(
            [('Model Name', object), ('Op Type', object), ('Core Type', object), ('Count', float),
             ('Total Time', float), ('Min Time', float), ('Avg Time', float), ('Max Time', float), ('Ratio', float)])
        self.steptrace_dt = [('Iteration ID', int), ('FP Start', float), ('BP End', float), ('Iteration End', float),
                             ('Iteration Time', float), ('FP to BP Time', float), ('Iteration Refresh', float),
                             ('Data Aug Bound', float), ('Model ID', int)]

    @staticmethod
    def find_files(directory, pattern):
        """Find files with feature 'pattern' from the directory"""

        for root, _, files in os.walk(directory):
            files.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)))
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    yield filename

    def read_files(self, directory, pattern):
        """Read files with feature 'pattern' from the directory"""

        dataset = []
        for file in self.find_files(directory, pattern):
            with open(file, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                next(reader)
                for row in reader:
                    row = ['0' if i == 'N/A' else i for i in row]
                    dataset.append(tuple(row))
        return dataset

    def parse(self):
        """read msprof data generate DataFrame data"""

        op_summary = []
        for file in self.find_files(self.source_path, "op_summary*.csv"):
            with open(file, newline='') as csvfile:
                step = int(file.split('_')[-1].split('.')[0])
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                next(reader)
                for row in reader:
                    row.append(step)
                    row = ['0' if i == 'N/A' else i for i in row]
                    op_summary.append(tuple(row))

        self.op_summary = np.array(op_summary, dtype=self.op_summary_dt)
        self.op_summary['Task Start Time'] = self.op_summary['Task Start Time'] * 1e-6
        self.op_summary['Task Duration'] = self.op_summary['Task Duration'] * 1e-3
        self.op_summary['Task Wait Time'] = self.op_summary['Task Wait Time'] * 1e-3
        self.op_summary['aicore_time'] = self.op_summary['aicore_time'] * 1e-3

        op_statistic = self.read_files(self.source_path, 'op_statistic*.csv')
        self.op_statistic = np.array(op_statistic, dtype=self.op_statistic_dt)
        self.op_statistic['Total Time'] = self.op_statistic['Total Time'] * 1e-3
        self.op_statistic['Min Time'] = self.op_statistic['Min Time'] * 1e-3
        self.op_statistic['Avg Time'] = self.op_statistic['Avg Time'] * 1e-3
        self.op_statistic['Max Time'] = self.op_statistic['Max Time'] * 1e-3

        steptrace = []
        for file in self.find_files(self.source_path, "step_trace*.csv"):
            with open(file, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                next(reader)
                for row in reader:
                    row = ['0' if i == 'N/A' else i for i in row]
                    steptrace.append(tuple(row))
            break

        hccl_data = self.op_summary[(self.op_summary['Task Type'] == 'HCCL') & (self.op_summary['Iteration ID'] == 1)][
            'Op Name']

        for name in hccl_data:
            self.steptrace_dt.append((name, float))
            self.steptrace_dt.append((f'{name} duration', float))
        self.steptrace_dt = np.dtype(self.steptrace_dt)

        self.steptrace = np.array(steptrace, dtype=self.steptrace_dt)
        self.steptrace['FP Start'] = self.steptrace['FP Start'] * 1e-3
        self.steptrace['BP End'] = self.steptrace['BP End'] * 1e-3
        self.steptrace['Iteration End'] = self.steptrace['Iteration End'] * 1e-3
        self.steptrace['Iteration Time'] = self.steptrace['Iteration Time'] * 1e-3
        self.steptrace['FP to BP Time'] = self.steptrace['FP to BP Time'] * 1e-3
        self.steptrace['Iteration Refresh'] = self.steptrace['Iteration Refresh'] * 1e-3
        self.steptrace['Data Aug Bound'] = self.steptrace['Data Aug Bound'] * 1e-3

        for name in self.steptrace.dtype.names[9:]:
            self.steptrace[name] = self.steptrace[name] * 1e-3

        return self.op_summary, self.op_statistic, self.steptrace
