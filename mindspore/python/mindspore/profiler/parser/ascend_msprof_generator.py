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

        self.op_summary_type = [
            ('Model ID', int),
            ('Task ID', int),
            ('Stream ID', int),
            ('Op Name', object),
            ('Op Type', object),
            ('Task Type', object),
            ('Task Start Time', float),
            ('Task Duration', float),
            ('Task Wait Time', float),
            ('Input Shapes', object),
            ('Input Data Types', object),
            ('Input Formats', object),
            ('Output Shapes', object),
            ('Output Data Types', object),
            ('Output Formats', object),
            ('vector_fops', float),
            ('cube_fops', float),
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
            ('Model ID', int),
        ]

    @staticmethod
    def find_files(directory, pattern):
        """Find files with feature 'pattern' from the directory"""

        for root, _, files in os.walk(directory):
            files.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)))
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    yield filename

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
                reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
                for row in reader:
                    new_row = (
                        row.get('Model ID'),
                        row.get('Task ID'),
                        row.get('Stream ID'),
                        row.get('Op Name'),
                        row.get('OP Type'),
                        row.get('Task Type'),
                        row.get('Task Start Time(us)'),
                        row.get('Task Duration(us)'),
                        row.get('Task Wait Time(us)'),
                        row.get('Input Shapes'),
                        row.get('Input Data Types'),
                        row.get('Input Formats'),
                        row.get('Output Shapes'),
                        row.get('Output Data Types'),
                        row.get('Output Formats'),
                        row.get('vector_fops', '0'),
                        row.get('cube_fops', '0')
                    )
                    new_row = tuple(['0' if d == 'N/A' else d for d in new_row])
                    op_summary.append(new_row)

        op_summary_dt = np.dtype(self.op_summary_type)

        self.op_summary = np.array(op_summary, dtype=op_summary_dt)
        self.op_summary['Task Start Time'] *= 1e-3
        self.op_summary['Task Duration'] *= 1e-3
        self.op_summary['Task Wait Time'] *= 1e-3

    def _read_op_statistic(self):
        """read op statistic to memory"""
        op_statistic = []
        for file in self.find_files(self.source_path, "op_statistic*.csv"):
            with open(file, newline='') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
                for row in reader:
                    new_row = (
                        row.get('OP Type'),
                        row.get('Count'),
                        row.get('Total Time(us)'),
                    )
                    new_row = tuple(['0' if d == 'N/A' else d for d in new_row])
                    op_statistic.append(new_row)

        op_statistic_dt = np.dtype(self.op_statistic_type)
        self.op_statistic = np.array(op_statistic, dtype=op_statistic_dt)
        self.op_statistic['Total Time'] *= 1e-3

    def _read_steptrace(self):
        """read steptrace to memory"""
        steptrace = []
        for file in self.find_files(self.source_path, "step_trace*.csv"):
            with open(file, newline='') as csvfile:
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
                    new_row = ['0' if i == 'N/A' else i for i in new_row]
                    steptrace.append(tuple(new_row))

        steptrace_dt = np.dtype(self.steptrace_type)

        self.steptrace = np.array(steptrace, dtype=steptrace_dt)
        self.steptrace['FP Start'] = self.steptrace['FP Start'] * 1e-3
        self.steptrace['BP End'] = self.steptrace['BP End'] * 1e-3
        self.steptrace['Iteration End'] = self.steptrace['Iteration End'] * 1e-3
        self.steptrace['Iteration Time'] = self.steptrace['Iteration Time'] * 1e-3
        self.steptrace['FP to BP Time'] = self.steptrace['FP to BP Time'] * 1e-3
        self.steptrace['Iteration Refresh'] = self.steptrace['Iteration Refresh'] * 1e-3
        self.steptrace['Data Aug Bound'] = self.steptrace['Data Aug Bound'] * 1e-3
