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
"""flops analyse model"""
import csv
import json
import logging
import os
import stat

import numpy as np
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException


class AscendFlopsGenerator:
    """Generate ascend flops data from DataFrame."""

    def __init__(self, op_summary):
        self.op_summary = op_summary
        self.flops_dt = np.dtype(
            [('op_full_name', object), ('MFLOPs(10^6 cube)', float), ('GFLOPS(10^9 cube)', float),
             ('MFLOPs(10^6 vector)', float), ('GFLOPS(10^9 vector)', float)])
        self.flops = None
        self.flops_summary = None

    def parse(self):
        """Analyse the op_summary data generate flops data."""

        self.flops = np.empty((len(self.op_summary)), dtype=self.flops_dt)

        nonzero_duration = self.op_summary['Task Duration'] != 0
        self.flops['op_full_name'] = self.op_summary['Op Name']
        self.flops['MFLOPs(10^6 cube)'] = self.op_summary['cube_fops'] * 1e-6
        self.flops['GFLOPS(10^9 cube)'] = np.where(nonzero_duration, self.op_summary['cube_fops'] / self.op_summary[
            'Task Duration'] * 1e-6, 0)
        self.flops['MFLOPs(10^6 vector)'] = self.op_summary['vector_fops'] * 1e-6
        self.flops['GFLOPS(10^9 vector)'] = self.op_summary['vector_fops'] / self.op_summary['Task Duration'] * 1e-6
        self.flops['GFLOPS(10^9 vector)'] = np.where(nonzero_duration, self.op_summary['vector_fops'] / self.op_summary[
            'Task Duration'] * 1e-6, 0)

        cube_flops = 0
        vec_flops = 0
        if np.sum(self.op_summary['Task Duration']) != 0:
            cube_flops = round(np.sum(self.flops['GFLOPS(10^9 cube)']) / np.sum(self.op_summary['Task Duration']), 4)
            vec_flops = round(np.sum(self.flops['GFLOPS(10^9 vector)']) / np.sum(self.op_summary['Task Duration']), 4)

        self.flops_summary = {
            'cube_FLOPs': round(float(np.sum(self.flops['MFLOPs(10^6 cube)'])), 4),
            'cube_FLOPS': cube_flops,
            'vec_FLOPs': round(float(np.sum(self.flops['MFLOPs(10^6 vector)'])), 4),
            'vec_FLOPS': vec_flops
        }

    def write(self, flops_path, flops_summary_path):
        """
        Write the flops.csv and flops_summary.json

        Args:
            flops_path(str): flops.csv path.
            flops_summary_path(str): flops_summary.json path.
        """
        try:
            with os.fdopen(os.open(flops_path,
                                   os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IWUSR | stat.S_IRUSR), 'w') as fp:
                writer = csv.writer(fp)
                writer.writerow(self.flops.dtype.names)
                writer.writerows(self.flops.tolist())
        except (IOError, OSError) as err:
            logging.critical('Errot occurred when write flops file: %s', err)
            raise ProfilerIOException() from err
        if os.path.exists(flops_path):
            os.chmod(flops_path, stat.S_IREAD | stat.S_IWRITE)

        try:
            with os.fdopen(os.open(flops_summary_path,
                                   os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IWUSR | stat.S_IRUSR),
                           'w') as json_file:
                json.dump(self.flops_summary, json_file)
        except (IOError, OSError) as err:
            logging.critical('Errot occurred when write step trace point info file: %s', err)
            raise ProfilerIOException() from err
        if os.path.exists(flops_summary_path):
            os.chmod(flops_summary_path, stat.S_IREAD | stat.S_IWRITE)
