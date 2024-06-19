# Copyright 2024 Huawei Technologies Co., Ltd
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
"""ClusterMonitor Callback class."""
from __future__ import absolute_import

import os
import stat
import glob
import time
from threading import RLock

from mindspore.train.callback._callback import Callback
from mindspore.communication.management import get_rank, get_local_rank
from mindspore import log as logger
from mindspore.parallel._auto_parallel_context import _get_auto_parallel_context
from mindspore.parallel._utils import _get_device_num
from mindspore.train._utils import get_parameter_redundancy

_perf_mutex = RLock()


def _get_dp_tp_from_redundancy(redundancy_tuple):
    """From redundancy get dp and tp"""
    dp = []
    tp = []
    for dp_value in redundancy_tuple:
        dp.append(list(dp_value))
    for i in range(len(redundancy_tuple[0])):
        tp.append([v[i] for v in redundancy_tuple])
    return dp, tp


def _get_dp_tp_from_layout(parameter_layout_dict, initial_rank=0):
    """From layout dict get dp and tp"""
    tp = []
    dp = []
    parameter_redundancy_dict = get_parameter_redundancy(parameter_layout_dict, initial_rank)
    value_len = 0
    for _, value in parameter_redundancy_dict.items():
        if len(value) > value_len:
            value_len = len(value)
            dp, tp = _get_dp_tp_from_redundancy(value)
    return dp, tp


def _check_perf_config(perf_config):
    """Check if the format of perf_config is correct."""
    enabled = perf_config.get("enable", None)
    if enabled is None or not isinstance(enabled, bool):
        raise TypeError(f"For cluster monitor, enabled should be bool, but got {type(enabled)}.")
    enable_step_time = perf_config.get("steptime", None)
    if enable_step_time is None or not isinstance(enable_step_time, bool):
        raise TypeError(f"For cluster monitor, enable_step_time should be bool, but got {type(enable_step_time)}.")
    enabled_dtp_group = perf_config.get("dtpGroup", None)
    if enabled_dtp_group is None or not isinstance(enabled_dtp_group, bool):
        raise TypeError(f"For cluster monitor, enabled_dtp_group should be bool, but got {type(enabled_dtp_group)}.")


def _parse_perf_config():
    """parse perf config"""
    perf_config = os.getenv("PERF_DUMP_CONFIG")
    perf_config_dict = {}
    if perf_config is None:
        return perf_config_dict
    pairs = perf_config.split(',')
    for pair in pairs:
        key, value = pair.split(':')
        if value.lower() == 'true':
            perf_config_dict[key] = True
        elif value.lower() == 'false':
            perf_config_dict[key] = False
        elif value.isdigit():
            perf_config_dict[key] = int(value)
        else:
            perf_config_dict[key] = value
    _check_perf_config(perf_config_dict)
    return perf_config_dict


def _remove_pre_log():
    """Remove the previously saved log files."""
    directory = os.getenv("PERF_DUMP_PATH")
    pattern = os.path.join(directory, "perf_ms_*.log")
    files_to_delete = glob.glob(pattern)
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
        except OSError as e:
            logger.warning(f"When CCAE is opening, {file_path} need to be removed, but failed to remove.")
            raise e


class ClusterMonitor(Callback):
    """
    Monitor the cluster in train process.
    """

    def __init__(self):
        super(ClusterMonitor, self).__init__()
        _remove_pre_log()
        self.perf_config = _parse_perf_config()
        self.enabled = self.perf_config.get("enable")
        self.enable_step_time = self.perf_config.get("steptime")
        self.enabled_dtp_group = self.perf_config.get("dtpGroup")
        self.data_time_start = None
        self.data_time_end = None
        self.frame_work = "MindSpore"
        self.ms_sched_host = os.getenv("MS_SCHED_HOST", "127.0.0.1")
        self.ms_sched_port = os.getenv("MS_SCHED_PORT", "8888")
        self.uuid_value = self.ms_sched_host + "_" + self.ms_sched_port
        self.global_rank = get_rank()
        self.process_id = os.getpid()
        self.device_id = get_local_rank()
        self.log_name = "perf_ms" + "_" + str(self.process_id) + "_" + str(self.device_id) + ".log"
        self.log_path = os.getenv("PERF_DUMP_PATH")
        if not self.log_path.endswith(os.path.sep):
            self.log_path += os.path.sep
        self.full_path = self.log_path + self.log_name

        self.write_dp_tp_flag = True
        self.initial_rank = 0

    def begin(self, run_context):
        pp_num = _get_auto_parallel_context("pipeline_stages")
        device_num = _get_device_num()

        original_list = list(range(device_num))
        chunk_size = device_num // pp_num
        split_pp_lists = []
        for i in range(0, device_num, chunk_size):
            end_index = i + chunk_size if i + chunk_size <= device_num else device_num
            split_pp_lists.append(original_list[i:end_index])

        self.initial_rank = (self.global_rank // chunk_size) * chunk_size
        with _perf_mutex:
            dir_path = os.path.dirname(self.full_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            if os.path.exists(self.full_path):
                os.chmod(self.full_path, stat.S_IWUSR)
                os.remove(self.full_path)
            with open(self.full_path, 'w') as file:
                log_message = f'UUID:{self.uuid_value}\nFRAMEWORK:{self.frame_work}\nGLOBAL RANKID:{self.global_rank}\n'
                file.write(log_message)
                for _, split_pp_list in enumerate(split_pp_lists):
                    file.write(f'PP:{split_pp_list}\n')
            os.chmod(self.full_path, stat.S_IRUSR)

    def step_begin(self, run_context):
        """
        Record time at the beginning of step.

        Args:
            run_context (RunContext): Context of the process running. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        self.data_time_start = time.time()

    def step_end(self, run_context):
        """
        Record time at the end of step.

        Args:
            run_context (RunContext): Context of the process running. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        self.data_time_end = time.time()
        if self.enabled and self.enabled_dtp_group and self.write_dp_tp_flag:
            cb_params = run_context.original_args()
            param_layout_dict = cb_params.train_network.parameter_layout_dict
            dp, tp = _get_dp_tp_from_layout(param_layout_dict, self.initial_rank)
            with _perf_mutex:
                os.chmod(self.full_path, stat.S_IWUSR)
                with open(self.full_path, 'a') as file:
                    for dp_value in dp:
                        file.write(f'dp:{dp_value}\n')
                    for tp_value in tp:
                        file.write(f'tp:{tp_value}\n')
                os.chmod(self.full_path, stat.S_IRUSR)
            self.write_dp_tp_flag = False
        if self.enabled and self.enable_step_time:
            with _perf_mutex:
                os.chmod(self.full_path, stat.S_IWUSR)
                with open(self.full_path, 'a') as file:
                    file.write(f"STEPTIME:{int(self.data_time_start * 1000)},{int(self.data_time_end * 1000)}\n")
                os.chmod(self.full_path, stat.S_IRUSR)
