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
"""trace step time analyse model"""
import csv
import fnmatch
import json
import logging
import os
import stat

import numpy as np
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException


def find_files(directory, pattern):
    """Find files from the directory"""
    file_list = []
    for root, _, file in os.walk(directory):
        file.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)))
        for base in file:
            if fnmatch.fnmatch(base, pattern):
                filename = os.path.join(root, base)
                file_list.append(filename)
    return file_list


class AscendClusterGenerator:
    """Generate step trace time data from msprof*.json"""

    def __init__(self, source_path):
        self.root_path = source_path
        self.msprof_data = np.array([])
        self.step_trace_time = {'Step': None, 'Computing': 0, 'comunNotOverlp': 0, 'Overlapped': 0, 'Communication': 0,
                                'Free': 0, 'Stage': 0, 'Bubble': 0, 'comunNotOverlpRec': 0}
        self.msprof_data_df = np.dtype([('name', object), ('ts', float), ('dur', float)])
        self.trace_step_time_df = np.dtype(
            [('Step', int), ('Computing', float), ('comunNotOverlp', float), ('Communication', float), ('Free', float),
             ('Stage', float), ('Bubble', float), ('comunNotOverlpRec', float)])
        self.title = ['Step', 'Computing', 'Communication(Not Overlapped)', 'Overlapped', 'Communication', 'Free',
                      'Stage', 'Bubble', 'Communication(Not Overlapped and Exclude Receive)']

    def parse(self):
        """
        Analyse msprof json generate cluster data.
        """
        self.read_msprof()

        self.step_trace_time['Computing'] = np.sum(self.msprof_data[self.msprof_data['name'] == 'Computing']['dur'])
        self.step_trace_time['comunNotOverlp'] = np.sum(
            self.msprof_data[self.msprof_data['name'] == 'Communication(Not Overlapped)']['dur'])
        self.step_trace_time['Communication'] = np.sum(
            self.msprof_data[self.msprof_data['name'] == 'Communication']['dur'])
        self.step_trace_time['Free'] = np.sum(self.msprof_data[self.msprof_data['name'] == 'Free']['dur'])
        self.step_trace_time['Bubble'] = np.sum(
            self.msprof_data[np.char.find(self.msprof_data['name'].astype('str'), '/Receive-op')]['dur'])

        self.step_trace_time['Overlapped'] = self.step_trace_time['Communication'] - self.step_trace_time[
            'comunNotOverlp']
        self.step_trace_time['Stage'] = np.max(self.msprof_data['ts'] + self.msprof_data['dur']) - np.min(
            self.msprof_data['ts']) - self.step_trace_time['Bubble']
        self.step_trace_time['comunNotOverlpRec'] = self.step_trace_time['comunNotOverlp'] - self.step_trace_time[
            'Bubble']

    def read_msprof(self):
        """
        read msprof json information into memory.
        """
        msprof_data = []
        for file in find_files(self.root_path, "msprof_*.json"):
            with open(file) as jsonfile:
                for row in json.load(jsonfile):
                    if row.get('name') in ['Computing', 'Communication', 'Communication(Not Overlapped)',
                                           'Free'] or row.get('name').find('/Receive-op'):
                        name = row.get('name', '')
                        ts = row.get('ts', 0)
                        dur = row.get('dur', 0)
                        msprof_data.append(tuple([name, ts, dur]))
        self.msprof_data = np.array(msprof_data, dtype=self.msprof_data_df)

    def write(self, step_trace_time_path):
        """
        Write the step trace time csv.

        Args:
            step_trace_time_path(str): step_trace_time.csv path.
        """
        try:
            with os.fdopen(os.open(step_trace_time_path,
                                   os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IWUSR | stat.S_IRUSR),
                           'w') as step_trace_time:
                writer = csv.writer(step_trace_time)
                writer.writerow(self.title)
                writer.writerow([v for _, v in self.step_trace_time.items()])
        except (IOError, OSError) as err:
            logging.critical('Error occurred when write step trace time file: %s', err)
            raise ProfilerIOException() from err
        if os.path.exists(step_trace_time_path):
            os.chmod(step_trace_time_path, stat.S_IREAD | stat.S_IWRITE)
