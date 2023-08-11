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
"""fp bp and is_training_mode_flag analyse model"""
import json
import logging
import os
import stat
from collections import defaultdict

import numpy as np
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException


class AscendFPBPGenerator:
    """Generate ascend fp bp data from DataFrame."""

    def __init__(self, op_summary, steptrace):
        self.op_summary = op_summary
        self.steptrace = steptrace
        self.points = None

    def parse(self):
        """Analyse the op_summary and steptrace data generate fpbp data."""
        is_training_mode_flag = False
        points = None

        steptrace = self.steptrace[self.steptrace['Iteration ID'] == 1]
        steptrace = steptrace[(steptrace['FP Start'] != 0) & (steptrace['BP End'] != 0)]
        if steptrace.size:
            is_training_mode_flag = True
            op_summary = self.op_summary[np.argsort(self.op_summary['Task Start Time'])]
            fp_index = np.searchsorted(op_summary['Task Start Time'], steptrace['FP Start'], side='right') - 1
            bp_index = np.searchsorted(op_summary['Task Start Time'] + op_summary['Task Duration'], steptrace['BP End'],
                                       side='left') - 1
            points = defaultdict(dict)
            for i in range(steptrace.size):
                model_id = f"model_{steptrace[i]['Iteration ID']}"
                points[model_id]['fp_start'] = op_summary[fp_index[i]]['Op Name']
                points[model_id]['bp_end'] = op_summary[bp_index[i]]['Op Name']
            self.points = defaultdict()
            self.points['fp_start'] = op_summary[fp_index[0]]['Op Name']
            self.points['bp_end'] = op_summary[bp_index[0]]['Op Name']

        return points, is_training_mode_flag

    def write(self, step_trace_point_info_path):
        """
        Write the flops.csv and flops_summary.json

        Args:
            step_trace_point_info_path(str): step_trace_point_info.json path.
        """
        if not self.points:
            return
        try:
            with os.fdopen(os.open(step_trace_point_info_path,
                                   os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IWUSR | stat.S_IRUSR),
                           'w') as json_file:
                json.dump(self.points, json_file)
        except (IOError, OSError) as err:
            logging.critical('Errot occurred when write step trace point info file: %s', err)
            raise ProfilerIOException() from err
        if os.path.exists(step_trace_point_info_path):
            os.chmod(step_trace_point_info_path, stat.S_IREAD | stat.S_IWRITE)
