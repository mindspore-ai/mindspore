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
import copy
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


def calculate_average(data):
    """Calculate hccl data average"""
    result = data
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            result[key] = calculate_average(value)
    elif isinstance(data, list):
        if all(isinstance(item, (dict, list)) for item in data):
            if isinstance(data[0], dict):
                result_dict = {}
                keys = set()
                for item in data:
                    keys.update(item.keys())
                for key in keys:
                    values = []
                    for item in data:
                        values.append(item.get(key))
                    result_dict[key] = calculate_average(values)
                result = result_dict
            elif isinstance(data[0], list):
                transposed_list = np.array(data).T.tolist()
                result = [calculate_average(sub_list) for sub_list in transposed_list]
    return result


def count_average(data):
    """Count average"""
    result = data
    if isinstance(data, list):
        if all(isinstance(x, (int, float)) for x in data):
            if data:
                result = sum(data) / len(data)
            else:
                result = 0
        else:
            result = [count_average(x) for x in data]
    elif isinstance(data, dict):
        result = {key: count_average(value) for key, value in data.items()}
    return result


class AscendHCCLGenerator:
    """Generate ascend hccl data from files."""

    def __init__(self, source_path):
        self.root_path = source_path
        self.hccl_raw = []
        self.hccl_data_df = np.dtype(
            [('name', object), ('pid', int), ('tid', int), ('ts', float), ('te', float), ('dur', float), ('ph', object),
             ('task_type', object), ('link_info', object), ('transport_type', object), ('size', int), ('tag', object)])

    @staticmethod
    def _cost_analyse(iteration):
        """analyse communication cost and wait cost"""
        communication_cost = np.sum(iteration[iteration['name'] != 'Notify_Wait']['dur'])
        wait_cost = np.sum(iteration[iteration['name'] == 'Notify_Wait']['dur'])
        return communication_cost, wait_cost

    @staticmethod
    def _rdma_analyse(groupby_transport):
        """rdma analyse"""
        thread_groups = np.unique(groupby_transport['tid'])
        thread_information = []
        for thread_index in thread_groups:
            groupby_thread = groupby_transport[groupby_transport['tid'] == thread_index]
            rdma_communication_time = 0
            rdma_communication_size = 0
            rdma_communication_wait_time = 0
            start_index = 0
            end_index = groupby_thread.size - 2
            while start_index < end_index:
                first_task_type = groupby_thread[start_index]['task_type']
                if first_task_type == 'RDMASend':
                    second_index = start_index + 1
                    third_index = start_index + 2
                    second_task_type = groupby_thread[second_index]['task_type']
                    third_task_type = groupby_thread[third_index]['task_type']
                    if second_task_type == 'RDMASend' and third_task_type == 'Notify Wait':
                        rdma_send_cost = groupby_thread[start_index]['dur']
                        notify_record_cost = groupby_thread[second_index]['dur']
                        notify_wait_cost = groupby_thread[third_index]['dur']
                        rdma_communication_time += rdma_send_cost + notify_record_cost + notify_wait_cost
                        rdma_communication_wait_time += notify_wait_cost
                        rdma_communication_size += groupby_thread[start_index]['size'] + groupby_thread[second_index][
                            'size']
                        start_index += 2
                start_index += 1
            rdma_communication_wait_time = rdma_communication_wait_time / 1e3
            rdma_communication_size = rdma_communication_size / 1e3
            rdma_communication_time = rdma_communication_time / 1e3
            rdma_bandwidth = rdma_communication_size / (rdma_communication_time / 1e3) \
                if rdma_communication_size else 0
            thread_information.append(
                [rdma_communication_time, rdma_communication_size, rdma_bandwidth, rdma_communication_wait_time])
        if len(thread_information) > 1:
            thread_information = np.sum(thread_information, axis=0).tolist()

        return thread_information

    def parse(self):
        """Analyse the original hccl data generator hccl data."""
        file_list = find_files(self.root_path, "hccl_*.json")

        for hccl_file in file_list:
            _, relative_path = os.path.split(hccl_file)
            iteration_id = int(relative_path.split('_')[3])
            with open(hccl_file) as f:
                _, hccl_detail_data = self._original_data_analyse(json.load(f))
                raw = self._iteration_analyse(hccl_detail_data, iteration_id)
                self.hccl_raw.append(raw)
        self.hccl_raw = sorted(self.hccl_raw, key=lambda x: x[0])
        self.hccl_raw.append(copy.deepcopy(self.hccl_raw[-1]))
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
                                   os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IWUSR | stat.S_IRUSR), 'w',
                           newline='') as hccl_row:
                writer = csv.writer(hccl_row)
                writer.writerow(
                    ['step_num', 'communication_cost', 'wait_cost', 'link_info', 'communication_operator_cost'])
                for row in self.hccl_raw:
                    row[3] = json.dumps(row[3])
                    row[4] = json.dumps(row[4])
                writer.writerows(self.hccl_raw)
        except (IOError, OSError) as err:
            logging.critical('Errot occurred when write aicore detail file: %s', err)
            raise ProfilerIOException() from err
        if os.path.exists(hccl_raw_path):
            os.chmod(hccl_raw_path, stat.S_IREAD | stat.S_IWRITE)

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
                task_type = row.get('args', {}).get('task type', '')
                src_rank = row.get('args', {}).get('src rank', 0)
                dst_rank = row.get('args', {}).get('dst rank', 0)
                if src_rank == int('0xffffffff', 16):
                    src_rank = dst_rank
                if dst_rank == int('0xffffffff', 16):
                    dst_rank = src_rank
                transport_type = row.get('args', {}).get('transport type', '')
                if transport_type == 'LOCAL':
                    src_rank, dst_rank = dst_rank, src_rank
                link_info = str(src_rank) + '-' + str(dst_rank)
                size = row.get('args', {}).get('size(Byte)', 0)
                size = size if isinstance(size, int) else int(size, 16)
                target_data.append(
                    tuple([name, pid, tid, ts, te, dur, ph, task_type, link_info, transport_type, size, -1]))
        hccl_data = np.array(target_data, dtype=self.hccl_data_df)

        hccl_abstract_data = hccl_data[hccl_data['task_type'] == '']
        hccl_detail_data = hccl_data[hccl_data['task_type'] != '']

        hccl_abstract_data = hccl_abstract_data[np.argsort(hccl_abstract_data['ts'])]
        hccl_detail_data = hccl_detail_data[np.argsort(hccl_detail_data['ts'])]
        hccl_detail_data['ts'] = hccl_detail_data['ts']
        tag = np.searchsorted(hccl_abstract_data['ts'], hccl_detail_data['ts'], side='right') - 1
        hccl_detail_data['tag'] = [x[-1] for x in np.char.split(hccl_abstract_data[tag]['name'].astype(str), sep='/')]
        return hccl_abstract_data, hccl_detail_data

    def _iteration_analyse(self, hccl_detail_data, iteration):
        """analyse data by iteration """
        communication_cost, wait_cost = self._cost_analyse(hccl_detail_data)
        link_info = self._link_info_analyse(hccl_detail_data)
        communication_operator_cost = self._communication_operator_cost_analyse(hccl_detail_data, iteration)
        return [iteration, communication_cost, wait_cost, link_info, communication_operator_cost]

    def _link_info_analyse(self, hccl_detail_data):
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
                if transport_index == 'SDMA' and groupby_transport.size > 0:
                    groupby_sdma = \
                        groupby_transport[np.isin(groupby_transport['task_type'], ['Memcpy', 'Reduce Inline'])][
                            ['dur', 'size']]
                    sdma_communication_time = np.sum(groupby_sdma['dur']) * 1e-3
                    sdma_communication_size = np.sum(groupby_sdma['size']) * 1e-3
                    sdma_bandwidth = sdma_communication_size / sdma_communication_time * 1e-3 \
                        if sdma_communication_time != 0 else 0
                    transport_information['SDMA'] = [sdma_communication_time, sdma_communication_size, sdma_bandwidth]
                elif transport_index == 'RDMA' and groupby_transport.size > 0:
                    transport_information['RDMA'] = self._rdma_analyse(groupby_transport)
            link_info_information[link_info_index] = transport_information
        return link_info_information

    def _communication_operator_cost_analyse(self, hccl_detail_data, iteration_index):
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
                        transport_information['SDMA'] = [
                            sdma_communication_time, sdma_communication_size,
                            sdma_bandwidth
                        ]
                    elif transport_index == 'RDMA':
                        transport_information['RDMA'] = self._rdma_analyse(groupby_transport)
                    link_info_information[link_info_index] = transport_information
                communication_cost = np.sum(groupby_tag[groupby_tag['name'] != 'Notify_Wait']['dur'])
                wait_cost = np.sum(groupby_tag[groupby_tag['name'] == 'Notify_Wait']['dur'])
                tag_information[tag_index] = [
                    str(iteration_index), communication_cost, wait_cost,
                    link_info_information
                ]
        return tag_information
