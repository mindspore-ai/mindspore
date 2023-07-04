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
""""step analyse model"""
import csv
import logging
import os
import stat

import numpy as np
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException


class AscendStepTraceGenerator:
    """Generate ascend steptrace data from DataFrame."""

    def __init__(self, steptrace):
        self.steptrace = steptrace
        self.steptrace_detail = None

        self.steptrace_detail_dt = [('step_num', object), ('start_point', int), ('end_point', int), ('total', int),
                                    ('fp_point', int),
                                    ('bp_point', int), ('iteration_interval', int), ('fp_and_bp', int), ('tail', int)]

    def parse(self):
        """Analyse the original steptrace data generate steptrace data."""

        for name in self.steptrace.dtype.names[9::2]:
            self.steptrace_detail_dt.append((name, float))
            self.steptrace_detail_dt.append((f'{name}_start_point', float))
            self.steptrace_detail_dt.append((f'{name}_end_point', float))
        self.steptrace_detail_dt = np.dtype(self.steptrace_detail_dt)
        self.steptrace_detail = np.empty((len(self.steptrace),), dtype=self.steptrace_detail_dt)

        unit = 1e+5
        self.steptrace_detail['step_num'] = self.steptrace['Iteration ID']
        self.steptrace_detail['start_point'] = (self.steptrace['FP Start'] - self.steptrace['Data Aug Bound']) * unit
        self.steptrace_detail['end_point'] = self.steptrace['Iteration End'] * unit
        self.steptrace_detail['total'] = (self.steptrace['Iteration Time'] + self.steptrace['Data Aug Bound']) * unit
        self.steptrace_detail['fp_point'] = self.steptrace['FP Start'] * unit
        self.steptrace_detail['bp_point'] = self.steptrace['BP End'] * unit
        self.steptrace_detail['iteration_interval'] = self.steptrace['Data Aug Bound'] * unit
        self.steptrace_detail['fp_and_bp'] = self.steptrace['FP to BP Time'] * unit
        self.steptrace_detail['tail'] = self.steptrace['Iteration Refresh'] * unit

        for name in self.steptrace.dtype.names[9::2]:
            self.steptrace_detail[name] = self.steptrace[f'{name} duration'] * unit
            self.steptrace_detail[f'{name}_start_point'] = self.steptrace[name] * unit
            self.steptrace_detail[f'{name}_end_point'] = (self.steptrace[name] + self.steptrace[
                f'{name} duration']) * unit

        avg_values = np.mean(self.steptrace_detail.tolist(), axis=0)
        avg_row = np.array([tuple(avg_values)], dtype=self.steptrace_detail_dt)
        avg_row['step_num'] = '-'
        self.steptrace_detail = np.append(self.steptrace_detail, avg_row)

    def write(self, step_trace_intermediate_file_path):
        """
        Write the step_trace_raw.csv

        Args:
            step_trace_intermediate_file_path(str): step_trace_raw.csv path.

        """
        try:
            with os.fdopen(os.open(step_trace_intermediate_file_path,
                                   os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o660), 'w') as st:
                writer = csv.writer(st)
                writer.writerow(self.steptrace_detail.dtype.names)
                writer.writerows(self.steptrace_detail.tolist())
        except (IOError, OSError) as err:
            logging.critical('Errot occurred when write step trace file: %s', err)
            raise ProfilerIOException()
        if os.path.exists(step_trace_intermediate_file_path):
            os.chmod(step_trace_intermediate_file_path, stat.ST_MODE | stat.S_IWRITE)
