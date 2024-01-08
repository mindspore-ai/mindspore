# Copyright (c) 2023, Huawei Technologies.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""communicate data analyze api file"""
import json
import logging
import os
import stat
from collections import defaultdict

from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException


class AscendCommunicationGenerator:
    """
    load and split communication info by step
    """
    COMMUNICATION_TIME_INFO = "Communication Time Info"
    START_TIMESTAMP = "Start Timestamp(us)"
    COMMUNICATION_BANDWIDTH_INFO = "Communication Bandwidth Info"
    HCOM_SEND = "Send"
    HCOM_RECEIVE = "Receive"
    TOTAL = "Total"
    SYNCHRONIZATION_TIME_RATIO = "Synchronization Time Ratio"
    SYNCHRONIZATION_TIME_MS = "Synchronization Time(ms)"
    WAIT_TIME_RATIO = "Wait Time Ratio"
    TRANSIT_TIME_MS = "Transit Time(ms)"
    TRANSIT_SIZE_MB = "Transit Size(MB)"
    SIZE_DISTRIBUTION = "Size Distribution"
    WAIT_TIME_MS = "Wait Time(ms)"
    BANDWIDTH_GB_S = "Bandwidth(GB/s)"
    COMMUNICATION = "communication.json"
    COMMUNICATION_MATRIX = "communication_matrix.json"
    P2P = "p2p"
    COLLECTIVE = "collective"
    TRANSPORT_TYPE = "Transport Type"

    def __init__(self, source_path):
        super().__init__()
        self.root_path = source_path
        self.step_list = [{"step_id": None, "start_ts": 0, "end_ts": float('inf'), "comm_ops": {}}]
        self.output_communication = {}
        self.output_matrix_data = {}

    @staticmethod
    def combine_size_distribution(op_dict: dict, total_dict: dict):
        for size, size_info in op_dict.items():
            total_dict[size][0] += size_info[0]
            total_dict[size][1] += size_info[1]

    @staticmethod
    def compute_ratio(dividend: float, divisor: float):
        if abs(divisor) < 1e-15:
            return 0
        return round(dividend / divisor, 4)

    def parse(self) -> None:
        self.generate_communication()
        self.generate_matrix()

    def generate_communication(self):
        """
        generate communication.json
        """
        communication_file = os.path.join(self.root_path, self.COMMUNICATION)
        with open(communication_file) as file:
            communication_data = json.load(file)
        if not communication_data:
            return
        self.split_comm_op_by_step(communication_data)

        for step_info in self.step_list:
            step = "step" + step_info.get("step_id") if step_info.get("step_id") else "step"
            self.output_communication[step] = self.get_communication_ops_dict(step_info.get("comm_ops"))

    def generate_matrix(self):
        communication_file = os.path.join(self.root_path, self.COMMUNICATION_MATRIX)
        with open(communication_file) as file:
            matrix_data = json.load(file)
        if not matrix_data:
            return
        matrix_data_by_step = self.split_matrix_by_step(matrix_data)

        for step, comm_matrix_data in matrix_data_by_step.items():
            self.output_matrix_data[step] = self.get_matrix_ops_dict(comm_matrix_data)

    def split_comm_op_by_step(self, communication_data: dict):
        if len(self.step_list) == 1:
            self.step_list[0]["comm_ops"] = communication_data
        for communication_op, communication_op_info in communication_data.items():
            start_time = communication_op_info.get(self.COMMUNICATION_TIME_INFO, {}).get(self.START_TIMESTAMP)
            for step_info in self.step_list:
                if step_info.get("start_ts", -1) <= start_time <= step_info.get("end_ts", -1):
                    step_info.get("comm_ops", {})[communication_op] = communication_op_info
                    break

    def split_communication_p2p_ops(self, op_data: dict):
        """
        split communicate
        """
        comm_op_dict = {self.P2P: {}, self.COLLECTIVE: {}}
        for communication_op, communication_info in op_data.items():
            # TODO: startswith
            if communication_op.find(self.HCOM_SEND) != -1 or communication_op.find(self.HCOM_RECEIVE) != -1:
                print(communication_op)
                comm_op_dict[self.P2P][communication_op] = communication_info
            elif communication_op.startswith(self.TOTAL):
                continue
            else:
                comm_op_dict[self.COLLECTIVE][communication_op] = communication_info
        return comm_op_dict

    def compute_total_link_info(self, op_matrix_data: dict):
        """
        compute total link info
        """
        total_op_info = defaultdict(lambda: {
            self.TRANSPORT_TYPE: '',
            self.TRANSIT_TIME_MS: 0,
            self.TRANSIT_SIZE_MB: 0
        })
        for _, link_dict in op_matrix_data.items():
            for link, link_info in link_dict.items():
                total_op_info[link][self.TRANSIT_TIME_MS] += link_info.get(self.TRANSIT_TIME_MS, 0)
                total_op_info[link][self.TRANSIT_SIZE_MB] += link_info.get(self.TRANSIT_SIZE_MB, 0)
                total_op_info[link][self.TRANSPORT_TYPE] = link_info.get(self.TRANSPORT_TYPE, 0)

        for link, _ in total_op_info.items():
            total_op_info[link][self.BANDWIDTH_GB_S] = \
                self.compute_ratio(total_op_info[link].get(self.TRANSIT_SIZE_MB, 0),
                                   total_op_info[link].get(self.TRANSIT_TIME_MS))
        op_matrix_data['Total Op Info'] = total_op_info

    def split_matrix_by_step(self, matrix_data: dict) -> dict:
        """
        split matrix by step
        """
        matrix_data_by_step = {}
        if self.is_step_list_empty():
            matrix_data_by_step["step"] = matrix_data
            return matrix_data_by_step

        for comm_op in matrix_data:
            for step_info in self.step_list:
                if comm_op in step_info.get("comm_ops", {}):
                    step = "step" + step_info.get("step_id") if step_info.get("step_id") else "step"
                    matrix_data_by_step.setdefault(step, {})[comm_op] = matrix_data.get(comm_op)
                    break
        return matrix_data_by_step

    def get_communication_ops_dict(self, op_data: dict) -> dict:
        comm_op_dict = self.split_communication_p2p_ops(op_data)
        self.compute_total_info(comm_op_dict[self.P2P])
        self.compute_total_info(comm_op_dict[self.COLLECTIVE])
        return comm_op_dict

    def get_matrix_ops_dict(self, op_data: dict) -> dict:
        comm_op_dict = self.split_communication_p2p_ops(op_data)
        self.compute_total_link_info(comm_op_dict[self.P2P])
        self.compute_total_link_info(comm_op_dict[self.COLLECTIVE])
        return comm_op_dict

    def is_step_list_empty(self):
        for step_info in self.step_list:
            if step_info.get("comm_ops"):
                return False
        return True

    def compute_total_info(self, comm_ops: dict):
        """
        compute total info
        """
        if not comm_ops:
            return
        total_time_info_dict = defaultdict(float)
        total_bandwidth_info_dict = {}
        for _, communication_op_info in comm_ops.items():
            for com_info, com_info_dict in communication_op_info.items():
                if com_info == self.COMMUNICATION_TIME_INFO:
                    self.combine_time_info(com_info_dict, total_time_info_dict)
                if com_info == self.COMMUNICATION_BANDWIDTH_INFO:
                    self.combine_bandwidth_info(com_info_dict, total_bandwidth_info_dict)
        self.compute_time_ratio(total_time_info_dict)
        self.compute_bandwidth_ratio(total_bandwidth_info_dict)
        comm_ops['Total Op Info'] = {
            self.COMMUNICATION_TIME_INFO: total_time_info_dict,
            self.COMMUNICATION_BANDWIDTH_INFO: total_bandwidth_info_dict
        }

    def combine_time_info(self, com_info_dict: dict, total_time_info_dict: dict):
        ratio_list = [self.WAIT_TIME_RATIO, self.SYNCHRONIZATION_TIME_RATIO]
        for time_info in com_info_dict:
            if time_info not in ratio_list and time_info != self.START_TIMESTAMP:
                total_time_info_dict[time_info] += com_info_dict.get(time_info)

    def combine_bandwidth_info(self, com_info_dict: dict, total_bandwidth_info_dict: dict):
        """
        combine bandwidth info
        """
        add_list = [self.TRANSIT_TIME_MS, self.TRANSIT_SIZE_MB]
        dict_list = [self.SIZE_DISTRIBUTION]
        for transport_type, part_transport_dict in com_info_dict.items():
            if transport_type not in total_bandwidth_info_dict:
                total_bandwidth_info_dict[transport_type] = {
                    self.TRANSIT_TIME_MS: 0,
                    self.TRANSIT_SIZE_MB: 0,
                    self.SIZE_DISTRIBUTION: defaultdict(lambda: [0, 0])
                }
            for bandwidth_msg, value in part_transport_dict.items():
                if bandwidth_msg in add_list:
                    total_bandwidth_info_dict[transport_type][bandwidth_msg] += value
                if bandwidth_msg in dict_list:
                    self.combine_size_distribution(value, total_bandwidth_info_dict[transport_type][bandwidth_msg])

    def compute_time_ratio(self, total_time_info_dict: dict):
        total_time_info_dict[self.WAIT_TIME_RATIO] = \
            self.compute_ratio(total_time_info_dict.get(self.WAIT_TIME_MS, 0),
                               total_time_info_dict.get(self.WAIT_TIME_MS, 0) +
                               total_time_info_dict.get(self.TRANSIT_TIME_MS, 0))
        total_time_info_dict[self.SYNCHRONIZATION_TIME_RATIO] = \
            self.compute_ratio(total_time_info_dict.get(self.SYNCHRONIZATION_TIME_MS, 0),
                               total_time_info_dict.get(self.TRANSIT_TIME_MS, 0) +
                               total_time_info_dict.get(self.SYNCHRONIZATION_TIME_MS, 0))

    def compute_bandwidth_ratio(self, total_bandwidth_info_dict: dict):
        for _, bandwidth_dict in total_bandwidth_info_dict.items():
            self.compute_ratio(bandwidth_dict.get(self.TRANSIT_SIZE_MB, 0), bandwidth_dict.get(self.TRANSIT_TIME_MS, 0))

    def write(self, communication_file_path, communication_matrix_file_path):
        """
        write communication file and communication matrix file
        """
        try:
            with os.fdopen(os.open(communication_file_path,
                                   os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as json_file:
                json.dump(self.output_communication, json_file)
        except (IOError, OSError) as err:
            logging.critical('Error occurred when write communication file: %s', err)
            raise ProfilerIOException() from err
        if os.path.exists(communication_file_path):
            os.chmod(communication_file_path, stat.S_IREAD | stat.S_IWRITE)

        try:
            with os.fdopen(os.open(communication_matrix_file_path,
                                   os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as json_file:
                json.dump(self.output_matrix_data, json_file)
        except (IOError, OSError) as err:
            logging.critical('Error occurred when write communication matrix file: %s', err)
            raise ProfilerIOException() from err
        if os.path.exists(communication_matrix_file_path):
            os.chmod(communication_matrix_file_path, stat.S_IREAD | stat.S_IWRITE)
