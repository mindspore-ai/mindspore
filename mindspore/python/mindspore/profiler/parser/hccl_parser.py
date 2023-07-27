# Copyright 2021 Huawei Technologies Co., Ltd
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
"""The parser for parsing hccl files."""
import csv
import json
import os
import stat
from enum import Enum
import numpy as np

from mindspore.profiler.common.exceptions.exceptions import \
    ProfilerPathErrorException, ProfilerFileNotFoundException, \
    ProfilerDirNotFoundException, ProfilerRawFileException
from mindspore import log as logger
from mindspore.profiler.common.validator.validate_path import \
    validate_and_normalize_path


class CommunicationInfo(Enum):
    """
    Communication related enumeration types.

    Enum:
        RDMA: Communication link between servers in cluster training.
        SDMA: Communication link inside server in cluster training.
        LOCAL: The operation of this card has no transmission process.
        RDMASEND: Communication operator of RDMA link.
        REDUCE_INLINE: Communication operator of SDMA link.
        MEMCPY: Communication operator of SDMA link.
        NOTIFY_RECORD: Communication operator of SDMA link.
        NOTIFY_WAIT: operator of LOCAL.
    """
    RDMA = 'RDMA'
    SDMA = 'SDMA'
    LOCAL = 'LOCAL'
    RDMASEND = 'RDMASend'
    REDUCE_INLINE = 'Reduce Inline'
    MEMCPY = 'Memcpy'
    NOTIFY_RECORD = 'Notify Record'
    NOTIFY_WAIT = 'Notify Wait'


class HcclParser:
    """
    The parser for parsing hccl file.

    Args:
        source_dir (str): The hccl source dir.
        device_id (str): The device ID.
        rank_id (str): The rank ID.
        output_path (str): The directory of the parsed file. Default: `./`.

    Raises:
        ProfilerPathErrorException: If the hccl file path or the output path is invalid.
        ProfilerFileNotFoundException: If the hccl file or the output dir does not exist.
    """
    _parsed_hccl_file_name = 'hccl_raw_{}.csv'
    _col_names = ['step_num', 'communication_cost', 'wait_cost', 'link_info', 'communication_operator_cost']

    def __init__(self, source_dir, device_id, rank_id, output_path):
        self._dev_id = device_id
        self._rank_id = rank_id
        self._source_dir = source_dir
        self._save_path = self._get_save_path(output_path)
        self._step_trace_info = self._get_step_trace_info(output_path)
        self._communication_operator_name_mapping_info = self._get_communication_operator_name_mapping_info()

    @staticmethod
    def _divide_communication_info_by_thread(trace_events: list):
        """Divide information by thread."""
        threads_dict = dict()
        for item in trace_events:
            thread_id = item.get("tid")
            if thread_id not in threads_dict.keys():
                threads_dict[thread_id] = [item]
            else:
                threads_dict[thread_id].append(item)
        return threads_dict

    @staticmethod
    def _calculate_adma_link_info(trace_event: list):
        """
        Calculate RDMA link info.

        When the link is RDMA,it is necessary to match three consecutive operators RDMASend, RDMASend \
        and Notify Wait,and take the sum of the time of the three operators as one communication time.
        """
        rdma_communication_time = 0
        rdma_communication_size = 0
        rdma_communication_wait_time = 0
        start_index = 0
        end_index = len(trace_event) - 1
        while start_index < end_index:
            first_task_type = trace_event[start_index].get("args").get("task type")
            if first_task_type == CommunicationInfo.RDMASEND.value and start_index < end_index - 1:
                second_task_type = trace_event[start_index + 1].get("args").get("task type")
                third_task_type = trace_event[start_index + 2].get("args").get("task type")
                if second_task_type == CommunicationInfo.RDMASEND.value and \
                        third_task_type == CommunicationInfo.NOTIFY_WAIT.value:
                    rdma_send_cost = trace_event[start_index].get("dur", 0)
                    notify_record_cost = trace_event[start_index + 1].get("dur", 0)
                    notify_wait_cost = trace_event[start_index + 2].get("dur", 0)
                    rdma_communication_time += rdma_send_cost + notify_record_cost + notify_wait_cost
                    rdma_communication_wait_time += notify_wait_cost
                    rdma_size = trace_event[start_index].get("args").get("size")
                    if rdma_size:
                        rdma_size = rdma_size if isinstance(rdma_size, int) else int(rdma_size, 16)
                    else:
                        rdma_size = 0
                    notify_record_size = trace_event[start_index + 1].get("args").get("size")
                    if notify_record_size:
                        notify_record_size = notify_record_size if isinstance(notify_record_size, int) \
                            else int(notify_record_size, 16)
                    else:
                        notify_record_size = 0
                    rdma_communication_size += rdma_size + notify_record_size
                    start_index += 2
            start_index += 1

        # The unit of rdma_communication_wait_time is ms.
        # The unit of rdma_bandwidth is KB/s.
        # The unit of rdma_communication_size is k_byte and The unit of rdma_communication_time is ms.
        rdma_communication_wait_time = rdma_communication_wait_time / 1e3
        rdma_communication_size = rdma_communication_size / 1e3
        rdma_communication_time = rdma_communication_time / 1e3
        rdma_bandwidth = rdma_communication_size / (rdma_communication_time / 1e3) \
            if rdma_communication_size else 0

        return [rdma_communication_time, rdma_communication_size, rdma_bandwidth, rdma_communication_wait_time]

    @staticmethod
    def _calculate_notify_wait_time(trace_event: list):
        """Calculate notify wait time."""
        total_notify_wait_time = 0
        for item in trace_event:
            task_type = item.get("args").get("task type")
            if task_type == CommunicationInfo.NOTIFY_WAIT.value:
                total_notify_wait_time += item.get("dur", 0)
        # The unit of total_notify_wait_time is ms.
        total_notify_wait_time = total_notify_wait_time / 1e3
        return total_notify_wait_time

    @staticmethod
    def _parser_link_dict(result_dict, src_dst_key, src_dst_value):
        """Parser link info to dict."""
        if src_dst_key not in result_dict.keys():
            result_dict[src_dst_key] = dict()
        for link_key, link_value in src_dst_value.items():
            if link_key not in result_dict[src_dst_key].keys():
                result_dict[src_dst_key][link_key] = list()
            result_dict[src_dst_key][link_key].append(link_value)

    @staticmethod
    def _calculate_link_value(link_info: list, calculate_type):
        """Calculate link average or total value."""
        result_dict = dict()
        for item in link_info:
            for src_dst_key, src_dst_value in item.items():
                HcclParser._parser_link_dict(result_dict, src_dst_key, src_dst_value)
        for src_dst_key, src_dst_value in result_dict.items():
            for link_key, _ in src_dst_value.items():
                if calculate_type == 'average':
                    result_dict[src_dst_key][link_key] = np.mean(result_dict[src_dst_key][link_key], axis=0).tolist()
                if calculate_type == 'total':
                    result_dict[src_dst_key][link_key] = np.sum(result_dict[src_dst_key][link_key], axis=0).tolist()

        return result_dict

    def parse(self):
        """Parse communication info."""
        self._parse_and_save(self._source_dir)

    def _parse_communication_cost(self, operators_cost_info, info, operators_dict):
        """Parse communication cost."""
        for k, v in operators_cost_info.items():
            for item in v:
                # index0:step_num
                if info[0] == item[0]:
                    operators_dict[k] = item

    def _parse_and_save(self, dir_path):
        """Parse and save communication info."""
        communication_info_cache = list()
        operators_cost_info = self._get_communication_operators_cost_info(dir_path)
        for _, v in operators_cost_info.items():
            for item in v:
                communication_info_cache.append(item)
        communication_info_cache = self._merge_communication_info_by_step_num(communication_info_cache)
        for info in communication_info_cache:
            operators_dict = dict()
            self._parse_communication_cost(operators_cost_info, info, operators_dict)
            info.append(operators_dict)
        # Calculate device communication average.
        device_communication_average_value = self._calculate_communication_average_value(communication_info_cache)
        # Calculate operator communication average.
        operators_average_value = dict()
        for k, v in operators_cost_info.items():
            average_value = self._calculate_communication_average_value(v)
            # The symbol '-' is used to indicate that the line is average information.
            average_value.insert(0, '-')
            operators_average_value[k] = average_value
        device_communication_average_value.append(operators_average_value)
        # The symbol '-' is used to indicate that the line is average information.
        device_communication_average_value.insert(0, '-')
        with open(self._save_path, 'w', newline='') as save_file:
            csv_writer = csv.writer(save_file)
            csv_writer.writerow(self._col_names)
            for item in communication_info_cache:
                # item[3]:link_info which is a dictionary that needs to be encoded before it is written to a CSV file.
                # item[4]:it is a dictionary that needs to be encoded before it is written to a CSV file.
                item[3] = json.dumps(item[3])
                item[4] = json.dumps(item[4])
                csv_writer.writerow(item)
            # device_communication_average_value[3]: average value for link info
            # device_communication_average_value[4]: average value for operator info
            device_communication_average_value[3] = json.dumps(device_communication_average_value[3])
            device_communication_average_value[4] = json.dumps(device_communication_average_value[4])

            csv_writer.writerow(device_communication_average_value)
        os.chmod(self._save_path, stat.S_IREAD | stat.S_IWRITE)

    def _get_save_path(self, output_path):
        """
        Get the save path.

        Args:
            output_path (str): The output dir.

        Returns:
            str, the save path.
        """
        output_path = self._validate_dir_path(output_path)
        return os.path.join(
            output_path, self._parsed_hccl_file_name.format(self._rank_id)
        )

    def _get_step_trace_info(self, source_dir):
        """Get the start and end timestamps in a step and communication operators names."""
        file_path = os.path.join(
            source_dir,
            f'step_trace_raw_{self._rank_id}_detail_time.csv'
        )
        try:
            file_path = validate_and_normalize_path(file_path)
        except RuntimeError as err:
            logger.warning('file path is invalid.')
            raise ProfilerPathErrorException('file path is invalid.') from err
        if not os.path.isfile(file_path):
            logger.warning('The step trace file <%s> not found.', file_path)
            raise ProfilerFileNotFoundException(file_path)

        with open(file_path, 'r') as src_file:
            csv_reader = csv.reader(src_file)
            # The first row of step trace file is like: step_num, start_point,...,communication_operator_name.
            # The position number of the first communication operator name is 9.
            communication_operators_names = next(csv_reader)[9:]

            # index_0:step_num, index_1:start_point, index_2:end_point
            # The unit of time stamp is 10ns. To convert it to μs, you need to divide it by 100.
            step_timestamps_info = [
                [info[0], float(info[1]) / 100, float(info[2]) / 100]
                for info in csv_reader if info[0].isdigit()
            ]

        return [communication_operators_names, step_timestamps_info]

    def _get_communication_operator_name_mapping_info(self):
        """Get the name of communication operators mapping between hccl and step trace."""
        dir_path = self._validate_dir_path(self._source_dir)
        # The name of the operator in hccl is like: operatorName_{Ordered_number}_xx_xx.
        operators_names_in_hccl = [entry.name for entry in os.scandir(dir_path) if entry.is_dir()]
        operators_names_in_hccl_set = set({i.split('_')[0] for i in operators_names_in_hccl})
        op_names_in_hccl_dic = dict()
        for item in operators_names_in_hccl_set:
            op_names_in_hccl_dic[item] = sorted([i for i in operators_names_in_hccl if i.split('_')[0] == item],
                                                key=lambda x: int(x.split('_')[1]))

        # The op_info in step trace is like: [op_name,op_name_start_point,op_name_end_point]
        # The name of the operator in step trace can be obtained every three.
        # The name of the operator in step trace is like: stream_xx_xx_operatorName-opxx.
        operators_names_in_step_trace = [self._step_trace_info[0][i]
                                         for i in range(0, len(self._step_trace_info[0]), 3)]
        op_names_in_step_trace_set = set({op_name.split('/')[-1].split('-')[0].split('_')[-1]
                                          for op_name in operators_names_in_step_trace})
        op_names_in_step_trace_dic = dict()
        for item in op_names_in_step_trace_set:
            op_names_in_step_trace_dic[item] = [
                op_name for op_name in operators_names_in_step_trace
                if op_name.split('/')[-1].split('-')[0].split('_')[-1] == item
            ]

        communication_operator_mapping_info = dict()
        for hccl_key, hccl_value in op_names_in_hccl_dic.items():
            for step_trace_key, step_trace_value in op_names_in_step_trace_dic.items():
                # the step_trace_key format is: operatorName
                if hccl_key.lower() == step_trace_key.lower().split('/')[-1]:
                    communication_operator_mapping_info[hccl_key] = list(zip(hccl_value, step_trace_value))

        logger.info("Communication operator name mapping info is %s", communication_operator_mapping_info)

        return communication_operator_mapping_info

    def _calculate_the_step_by_timestamp(self, timestamp):
        """Calculate the step according to the timestamp."""
        # index0:communication_operator_name, index1:step_timestamps_info
        step_timestamps_info = self._step_trace_info[1]
        step_timestamps_len = len(step_timestamps_info)
        # index_0:step_num, index_1:start_point, index_2:end_point
        if timestamp < step_timestamps_info[0][1]:
            step_num = "1"
        elif step_timestamps_info[step_timestamps_len - 1][2] < timestamp:
            step_num = step_timestamps_info[step_timestamps_len - 1][0]
        else:
            for item in step_timestamps_info:
                if item[1] <= timestamp < item[2]:
                    step_num = item[0]
        return step_num

    def _get_communication_operators_cost_info(self, dir_path):
        """Obtain time-consuming information of all communication operators."""
        operators_cost_info = dict()
        dir_path = self._validate_dir_path(dir_path)
        operators_dir = [entry.name for entry in os.scandir(dir_path) if entry.is_dir()]
        operator_dir_path = [os.path.join(dir_path, operator_dir) for operator_dir in operators_dir]
        for operator_dir in operator_dir_path:
            operator_cost = self._calculate_communication_operator_cost(operator_dir)
            operator_name = os.path.basename(operator_dir)
            op_mapping_info = self._communication_operator_name_mapping_info.get(operator_name.split('_')[0], [])
            # index1: operator name in step trace.
            op_mapping_name = [item[1] for item in op_mapping_info if item[0] == operator_name]
            if not op_mapping_name:
                logger.warning("The mapping relationship between op name in hccl and op name in step trace "
                               "cannot be found. Use op name in hccl to show the name of the communication operator.")
            else:
                operator_name = op_mapping_name[0]
            operators_cost_info[operator_name] = operator_cost
        return operators_cost_info

    def _calculate_communication_operator_cost(self, dir_path):
        """Calculate communication operator cost. Such as allReduce_1,allReduce_2."""
        dir_path = self._validate_dir_path(dir_path)
        files = [entry.name for entry in os.scandir(dir_path) if entry.is_file()]
        files_path = [os.path.join(dir_path, file) for file in files]
        operator_cost = list(map(self._calculate_communication_operator_iter_cost, files_path))
        # Add the same step_num merge.
        steps_operator_cost = self._merge_communication_info_by_step_num(operator_cost)
        return steps_operator_cost

    def _merge_communication_info_by_step_num(self, communication_info: list):
        """According to step num to merge communication info."""
        steps_communication_info = list()
        info_set = set()
        for item in communication_info:
            # index0:step_num,index1:communication_cost,index2:communication_wait_cost,index3:link_info
            if item[0].isdigit():
                info_set.add(int(item[0]))
        info_set = sorted(info_set)
        for item in info_set:
            item = str(item)
            step_communication_info = [info for info in communication_info if info[0] == item]
            step_communication_cost = sum([i[1] for i in step_communication_info])
            step_communication_wait_cost = sum([i[2] for i in step_communication_info])
            step_communication_link = self._calculate_link_value([i[3] for i in step_communication_info], "total")
            steps_communication_info.append([item, step_communication_cost,
                                             step_communication_wait_cost, step_communication_link])
        return steps_communication_info

    def _calculate_communication_operator_iter_cost(self, file_path):
        """Calculate the time-consuming of communication operator in one execution round."""

        def _inner_calculate_communication_operator_iter_cost(events):
            total_notify_wait = HcclParser._calculate_notify_wait_time(events)
            # Divide information by src dst rank_id.
            src_dst_dict = self._divide_communication_info_by_src_dst_rank(events)
            src_dst_link_info = self._calculate_src_dst_link_info(src_dst_dict)
            communication_cost, communication_wait = self._calculate_device_communication_cost(src_dst_link_info)
            total_notify_wait -= communication_wait
            return [communication_cost, total_notify_wait, src_dst_link_info]

        file_path = self._validate_file_path(file_path)
        with open(file_path, 'r') as src_file:
            try:
                operator_info = json.load(src_file)
            except (json.JSONDecodeError, TypeError) as err:
                logger.warning(err)
                raise ProfilerRawFileException('Fail to parse operator file.') from err
        trace_events = operator_info.get("traceEvents")
        operator_timestamp = trace_events[0].get("ts", 0)
        step_id = self._calculate_the_step_by_timestamp(operator_timestamp)
        # Statistics of communication operators in all streams.
        total_communication_operator_iter_cost = \
            _inner_calculate_communication_operator_iter_cost(trace_events)
        # Statistics of communication operators in mainstream.
        threads_dict = self._divide_communication_info_by_thread(trace_events)
        # The largest value is mainstream.
        major_thread = sorted(threads_dict, reverse=True)[0]
        major_thread_trace_events = threads_dict.get(major_thread)
        mainstream_communication_operator_iter_cost = \
            _inner_calculate_communication_operator_iter_cost(major_thread_trace_events)
        # index0:communication_cost,index1:communication_wait_cost,index2:link_info
        return [step_id, mainstream_communication_operator_iter_cost[0],
                mainstream_communication_operator_iter_cost[1],
                total_communication_operator_iter_cost[2]]

    def _divide_communication_info_by_src_dst_rank(self, trace_event: list):
        """Divide information by src rank id and dst rank id"""
        src_dst_dict = dict()
        for item in trace_event:
            src_rank = item.get("args").get("src rank")
            dst_rank = item.get("args").get("dst rank")
            if src_rank is None or dst_rank is None:
                continue

            # When the SDMA operation is in the card,
            # the source card or destination card is 0xffffffff, and it needs to be converted to localrank.
            if int(src_rank) == int('0xffffffff', 16):
                src_rank = dst_rank

            if int(dst_rank) == int('0xffffffff', 16):
                dst_rank = src_rank

            if item.get("args").get("transport type") == CommunicationInfo.LOCAL.value:
                item["args"]["src rank"] = dst_rank
                item["args"]["dst rank"] = src_rank
                src_dst_key = str(dst_rank) + '-' + str(src_rank)
            else:
                src_dst_key = str(src_rank) + '-' + str(dst_rank)

            if src_dst_key not in src_dst_dict.keys():
                src_dst_dict[src_dst_key] = [item]
            else:
                src_dst_dict[src_dst_key].append(item)
        return src_dst_dict

    def _divide_communication_info_by_link_type(self, trace_event: list):
        """Divide information by link type."""
        link_type_dict = dict()
        for item in trace_event:
            link_type_key = item.get("args").get("transport type")
            if link_type_key is None:
                continue
            if link_type_key in (CommunicationInfo.RDMA.value, CommunicationInfo.SDMA.value):
                task_type = item.get("args").get("task type")
                # Filter out the Notify Record operator in SDMA, because it does not transmit the actual amount of data.
                if task_type == CommunicationInfo.NOTIFY_RECORD.value:
                    continue
                if link_type_dict.get(link_type_key):
                    link_type_dict[link_type_key].append(item)
                else:
                    link_type_dict[link_type_key] = [item]
            if link_type_key == CommunicationInfo.LOCAL.value:
                if link_type_dict.get(CommunicationInfo.RDMA.value):
                    link_type_dict[CommunicationInfo.RDMA.value].append(item)
        return link_type_dict

    def _calculate_device_communication_cost(self, src_dst_link_info: dict):
        """Calculate notify wait time."""
        total_communication_time = 0
        total_wait_time = 0
        for src_dst_value in src_dst_link_info.values():
            for link_type_value in src_dst_value.values():
                # time_cost:0,size_cost:1,brand_width:2,wait_time:3
                total_communication_time += link_type_value[0]
                if len(link_type_value) > 3:
                    total_wait_time += link_type_value[3]
        return total_communication_time, total_wait_time

    def _parse_link_cost(self, result_dict, key, link_type_dict):
        """Parse link cost."""
        for link_type_key, link_type_value in link_type_dict.items():
            if link_type_key == CommunicationInfo.RDMA.value:
                # Divide information by thread.
                rdma_infos = []
                threads_dict = self._divide_communication_info_by_thread(link_type_value)
                for thread_value in threads_dict.values():
                    rdma_info = self._calculate_adma_link_info(thread_value)
                    rdma_infos.append(rdma_info)
                rdma_total_cost = np.sum(rdma_infos, axis=0).tolist()
                result_dict[key][link_type_key] = rdma_total_cost
            if link_type_key == CommunicationInfo.SDMA.value:
                sdma_total_cost = self._calculate_sdma_link_info(link_type_value)
                result_dict[key][link_type_key] = sdma_total_cost

    def _calculate_src_dst_link_info(self, src_dst_dict: dict):
        """Calculate src dst link info."""
        result_dict = dict()
        for k, v in src_dst_dict.items():
            # Divide information by link type.
            link_type_dict = self._divide_communication_info_by_link_type(v)
            if not link_type_dict:
                continue
            result_dict[k] = dict()
            self._parse_link_cost(result_dict, k, link_type_dict)
        return result_dict

    def _calculate_sdma_link_info(self, trace_event: list):
        """
        Calculate SDMA link info.

        When the link is SDMA, the communication time of the primary link is the sum of the execution time\
        of Reduce inline and Memcpy operators.
        """
        sdma_communication_time = 0
        sdma_communication_size = 0

        for item in trace_event:
            task_type = item.get("args").get("task type")
            if task_type in (CommunicationInfo.REDUCE_INLINE.value, CommunicationInfo.MEMCPY.value):
                sdma_communication_time += item.get("dur", 0)
                sdma_size = item.get("args").get("size")
                if sdma_size:
                    sdma_size = sdma_size if isinstance(sdma_size, int) else int(sdma_size, 16)
                else:
                    sdma_size = 0

                sdma_communication_size += sdma_size

        # The unit of sdma_bandwidth is KB/s.
        # The unit of sdma_communication_size is k_byte and The unit of sdma_communication_time is ms.
        sdma_communication_time = sdma_communication_time / 1e3
        sdma_communication_size = sdma_communication_size / 1e3
        sdma_bandwidth = sdma_communication_size / (sdma_communication_time / 1e3) \
            if sdma_communication_size else 0
        return [sdma_communication_time, sdma_communication_size, sdma_bandwidth]

    def _calculate_communication_average_value(self, communication_info: list):
        """Calculate communication average value."""
        communication_info_size = len(communication_info)
        if communication_info_size == 0:
            return []
        # index1: communication_cost,index2:wait_cost,index3:link_info
        communication_cost_average = sum([i[1] for i in communication_info]) / communication_info_size
        wait_cost_average = sum([i[2] for i in communication_info]) / communication_info_size
        link_info = [i[3] for i in communication_info]
        calculate_type = 'average'
        link_average_info = HcclParser._calculate_link_value(link_info, calculate_type)
        return [communication_cost_average, wait_cost_average, link_average_info]

    def _validate_file_path(self, file_path):
        """Validate file path."""
        try:
            file_path = validate_and_normalize_path(file_path)
        except RuntimeError as err:
            logger.warning('file path is invalid.')
            raise ProfilerPathErrorException('file path is invalid.') from err
        if not os.path.isfile(file_path):
            logger.warning('The file <%s> not found.', file_path)
            raise ProfilerFileNotFoundException(file_path)
        return file_path

    def _validate_dir_path(self, dir_path):
        """Validate dir path."""
        try:
            dir_path = validate_and_normalize_path(dir_path)
        except RuntimeError as err:
            logger.warning('dir path is invalid.')
            raise ProfilerPathErrorException('dir path is invalid.') from err
        if not os.path.isdir(dir_path):
            logger.warning('The  dir <%s> not found.', dir_path)
            raise ProfilerDirNotFoundException(dir_path)
        return dir_path
