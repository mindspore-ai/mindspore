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
"""hccl analyse model"""
import csv
import fnmatch
import json
import logging
import os
import stat

import numpy as np
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException


def find_files(directory, pattern):
    """Find files with feature 'pattern' from the directory"""
    file_list = []
    for root, _, files in os.walk(directory):
        files.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)))
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                file_list.append(filename)
    return file_list


class AscendHCCLGenerator:
    """Generate ascend hccl data from files."""

    def __init__(self, source_path):
        self.root_path = source_path
        self.hccl_raw = []
        self.hccl_data_df = np.dtype(
            [('name', object), ('pid', int), ('tid', int), ('ts', float), ('te', float), ('dur', float), ('ph', object),
             ('task_type', object), ('link_info', object), ('transport_type', object), ('size', int), ('tag', object)])

    def parse(self):
        """Analyse the original hccl data generator hccl data."""
        file_list = find_files(self.root_path, "hccl_*.json")

        for hccl_file in file_list:
            iteration_id = int(hccl_file.split('_')[-1].split(('.'))[0])
            with open(hccl_file) as f:
                hccl_abstract_data, hccl_detail_data = self._original_data_analyse(json.load(f))
                raw = self._iteration_analyse(hccl_abstract_data, hccl_detail_data, iteration_id)
                self.hccl_raw.append(raw)
        self.hccl_raw = sorted(self.hccl_raw, key=lambda x: x[0])
        # use avg = count_average(calculate_average(self.hccl_raw))
        self.hccl_raw[-1][0] = '-'
        for _, value in self.hccl_raw[-1][4].items():
            value[0] = '-'

    def write(self, hccl_raw_path):
        """
        Write the flops.csv and flops_summary.json

        Args:
            hccl_raw_path(str): hccl_raw.csv path.
        """
        try:
            with os.fdopen(os.open(hccl_raw_path,
                                   os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o660), 'w') as aicore_detail:
                writer = csv.writer(aicore_detail)
                writer.writerow(
                    ['step_num', 'communication_cost', 'wait_cost', 'link_info', 'communication_operator_cost'])
                writer.writerows(self.hccl_raw)
        except (IOError, OSError) as err:
            logging.critical('Errot occurred when write aicore detail file: %s', err)
            raise ProfilerIOException()
        if os.path.exists(hccl_raw_path):
            os.chmod(hccl_raw_path, stat.ST_MODE | stat.S_IWRITE)

    def _original_data_analyse(self, original_data):
        """analyse original data"""
        target_data = []
        for row in original_data:
            if row.get('ph') == 'X':
                name = row.get('name')
                pid = row.get('pid')
                tid = row.get('tid')
                ts = row.get('ts')
                dur = row.get('dur')
                te = ts + dur
                ph = row.get('ph')
                task_type = row.get('args').get('task type')
                src_rank = row.get('args').get('src rank')
                dst_rank = row.get('args').get('dst rank')
                if src_rank == int('0xffffffff', 16):
                    src_rank = dst_rank
                if dst_rank == int('0xffffffff', 16):
                    dst_rank = src_rank
                link_info = str(src_rank) + '-' + str(dst_rank)
                transport_type = row.get('args').get('transport type')
                size = row.get('args').get('size(Byte)')
                target_data.append(
                    tuple([name, pid, tid, ts, te, dur, ph, task_type, link_info, transport_type, size, -1]))
        hccl_data = np.array(target_data, dtype=self.hccl_data_df)

        hccl_abstract_data = hccl_data[hccl_data['tid'] >= 100]
        hccl_detail_data = hccl_data[hccl_data['tid'] < 100]

        hccl_abstract_data = hccl_abstract_data[np.argsort(hccl_abstract_data['ts'])]
        hccl_detail_data = hccl_detail_data[np.argsort(hccl_detail_data['ts'])]
        hccl_detail_data['ts'] = hccl_detail_data['ts']
        tag = np.searchsorted(hccl_abstract_data['ts'], hccl_detail_data['ts'], side='right') - 1
        hccl_detail_data['tag'] = [x[-1] for x in np.char.split(hccl_abstract_data[tag]['name'].astype(str), sep='/')]
        return hccl_abstract_data, hccl_detail_data

    def _iteration_analyse(self, hccl_abstract_data, hccl_detail_data, iteration):
        """analyse data by iteration """
        communication_cost, wait_cost = self._cost_analyse(hccl_abstract_data)
        link_info = self._link_info_analyse(hccl_detail_data)
        communication_operator_cost = self._communication_operator_cost_analyse(hccl_detail_data, iteration)
        return [iteration, communication_cost, wait_cost, link_info, communication_operator_cost]

    @staticmethod
    def _cost_analyse(iteration):
        """analyse communication cost and wait cost"""
        communication_cost = np.sum(iteration[iteration['name'] != 'Notify_Wait']['dur'])
        wait_cost = np.sum(iteration[iteration['name'] == 'Notify_Wait']['dur'])
        return communication_cost, wait_cost

    @staticmethod
    def _link_info_analyse(hccl_detail_data):
        """analyse link info data"""
        groupby_iteration = hccl_detail_data[hccl_detail_data['task_type'] != 'Notify Record']
        link_info_groups = np.unique(groupby_iteration['link_info'])
        link_info_information = dict()
        for link_info_index in link_info_groups:
            groupby_link_info = groupby_iteration[groupby_iteration['link_info'] == link_info_index]
            transport_groups = np.unique(groupby_iteration['transport_type'])
            transport_information = dict()
            for transport_index in transport_groups:
                groupby_transport = groupby_link_info[groupby_link_info['transport_type'] == transport_index]
                if transport_index == 'SDMA':
                    groupby_sdma = \
                        groupby_transport[np.isin(groupby_transport['task_type'], ['Memcpy', 'Reduce Inline'])][
                            ['dur', 'size']]
                    sdma_communication_time = np.sum(groupby_sdma['dur']) * 1e-3
                    sdma_communication_size = np.sum(groupby_sdma['size']) * 1e-3
                    sdma_bandwidth = sdma_communication_size / sdma_communication_time * 1e-3 \
                        if sdma_communication_time != 0 else 0
                    transport_information['SDMA'] = [sdma_communication_time, sdma_communication_size, sdma_bandwidth]
                elif transport_index == 'RDMA':
                    transport_information['RDMA'] = []
            link_info_information[link_info_index] = transport_information
        return link_info_information

    @staticmethod
    def _communication_operator_cost_analyse(hccl_detail_data, iteration_index):
        """analyse communication operator cost"""
        groupby_iteration = hccl_detail_data[hccl_detail_data['task_type'] != 'Notify Record']
        tag_groups = np.unique(groupby_iteration['tag'])
        tag_information = dict()
        for tag_index in tag_groups:
            groupby_tag = groupby_iteration[groupby_iteration['tag'] == tag_index]
            link_groups = np.unique(groupby_iteration['link_info'])
            link_info_information = dict()
            for link_info_index in link_groups:
                groupby_link_info = groupby_tag[groupby_tag['link_info'] == link_info_index]
                transport_groups = np.unique(groupby_link_info['transport_type'])
                transport_information = dict()
                for transport_index in transport_groups:
                    groupby_transport = groupby_link_info[groupby_link_info['transport_type'] == transport_index]
                    if transport_index == 'SDMA':
                        groupby_sdma = \
                            groupby_transport[np.isin(groupby_transport['task_type'], ['Memcpy', 'Reduce Inline'])][
                                ['dur', 'size']]
                        sdma_communication_time = np.sum(groupby_sdma['dur']) * 1e-3
                        sdma_communication_size = np.sum(groupby_sdma['size']) * 1e-3
                        sdma_bandwidth = sdma_communication_size / sdma_communication_time * 1e-3 \
                            if sdma_communication_time != 0 else 0
                        transport_information['SDMA'] = [sdma_communication_time, sdma_communication_size,
                                                         sdma_bandwidth]
                    elif transport_index == 'RDMA':
                        transport_information['RDMA'] = []
                    link_info_information[link_info_index] = transport_information
                communication_cost = np.sum(groupby_tag[groupby_tag['name'] != 'Notify_Wait']['dur'])
                wait_cost = np.sum(groupby_tag[groupby_tag['name'] == 'Notify_Wait']['dur'])
                tag_information[tag_index] = [str(iteration_index), communication_cost, wait_cost,
                                              link_info_information]
        return tag_information
