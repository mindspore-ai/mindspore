# Copyright 2025 Huawei Technologies Co., Ltd
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
"""profiler check base function."""
from __future__ import absolute_import

import csv
import json
import logging
import os
import re
import subprocess
import threading
import time

import pandas as pd

import mindspore as ms
from mindspore import log as logger
from mindspore._c_expression import MSContext
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics, ExportType
from mindspore.profiler import Profiler
from mindspore.train.callback import Callback

de = MSContext.get_instance().get_ascend_soc_version()
JIT_LEVEL = "O0"


class StopAtEpochNew(Callback):
    """for onconditions_ascend StopAtEpoch"""

    def __init__(self, start_epoch, stop_epoch, **kwargs):
        """init"""
        self.start_profile = kwargs.get('start_profile', False)
        self.activities = kwargs.get('activities', [ProfilerActivity.CPU, ProfilerActivity.NPU])
        self.with_stack = kwargs.get('with_stack', False)
        self.data_process = kwargs.get('data_process', False)
        self.parallel_strategy = kwargs.get('parallel_strategy', False)
        self.hbm_ddr = kwargs.get('hbm_ddr', False)
        self.pcie = kwargs.get('pcie', False)
        self.dir_name = kwargs.get('dir_name', "./data")
        self.worker_name = kwargs.get('worker_name', "pro_callback")
        self.analyse_flag = kwargs.get('analyse_flag', True)
        self.async_mode = kwargs.get('async_mode', False)
        self.wait = kwargs.get('wait', 0)
        self.warmup = kwargs.get('warmup', 0)
        self.active = kwargs.get('active', 1)
        self.repeat = kwargs.get('repeat', 1)
        self.skip_first = kwargs.get('skip_first', 0)
        self.profiler_level = kwargs.get('profiler_level', ProfilerLevel.Level0)
        self.profile_memory = kwargs.get('profile_memory', False)
        self.mstx = kwargs.get('mstx', False)
        self.data_simplification = kwargs.get('data_simplification', True)
        self.aic_metrics = kwargs.get('aic_metrics', AicoreMetrics.AiCoreNone)
        self.l2_cache = kwargs.get('l2_cache', False)
        self.export_type = kwargs.get('export_type', [ExportType.Db, ExportType.Text])
        super().__init__()
        self.start_epoch = start_epoch
        self.stop_epoch = stop_epoch
        self.profiler = ms.profiler.profile(start_profile=self.start_profile,
                                            activities=self.activities, with_stack=self.with_stack,
                                            profile_memory=self.profile_memory, data_process=self.data_process,
                                            parallel_strategy=self.parallel_strategy,
                                            hbm_ddr=self.hbm_ddr, pcie=self.pcie,
                                            on_trace_ready=ms.profiler.tensorboard_trace_handler(
                                                dir_name=self.dir_name, worker_name=self.worker_name,
                                                analyse_flag=self.analyse_flag,
                                                async_mode=self.async_mode),
                                            schedule=ms.profiler.schedule(
                                                wait=self.wait, warmup=self.warmup, active=self.active,
                                                repeat=self.repeat, skip_first=self.skip_first),
                                            experimental_config=ms.profiler._ExperimentalConfig(
                                                profiler_level=self.profiler_level, mstx=True,
                                                data_simplification=self.data_simplification,
                                                aic_metrics=self.aic_metrics, l2_cache=self.l2_cache,
                                                export_type=self.export_type))

    def on_train_epoch_begin(self, run_context):
        """epoch_begin"""
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        if epoch_num == self.start_epoch:
            self.profiler.start()

    def on_train_epoch_end(self, run_context):
        """epoch_stop"""
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        if epoch_num == self.stop_epoch:
            self.profiler.step()


class StopAtStepNew(Callback):
    """for onconditions ascend StopAtStep"""

    def __init__(self, **kwargs):
        """init"""
        self.start_profile = kwargs.get('start_profile', False)
        self.activities = kwargs.get('activities', [ProfilerActivity.CPU, ProfilerActivity.NPU])
        self.with_stack = kwargs.get('with_stack', False)
        self.data_process = kwargs.get('data_process', False)
        self.parallel_strategy = kwargs.get('parallel_strategy', False)
        self.hbm_ddr = kwargs.get('hbm_ddr', False)
        self.pcie = kwargs.get('pcie', False)
        self.dir_name = kwargs.get('dir_name', "./data")
        self.worker_name = kwargs.get('worker_name', "pro_callback")
        self.analyse_flag = kwargs.get('analyse_flag', True)
        self.async_mode = kwargs.get('async_mode', False)
        self.wait = kwargs.get('wait', 0)
        self.warmup = kwargs.get('warmup', 0)
        self.active = kwargs.get('active', 1)
        self.repeat = kwargs.get('repeat', 1)
        self.skip_first = kwargs.get('skip_first', 0)
        self.profiler_level = kwargs.get('profiler_level', ProfilerLevel.Level0)
        self.mstx = kwargs.get('mstx', False)
        self.profile_memory = kwargs.get('profile_memory', False)
        self.data_simplification = kwargs.get('data_simplification', True)
        self.aic_metrics = kwargs.get('aic_metrics', AicoreMetrics.AiCoreNone)
        self.l2_cache = kwargs.get('l2_cache', False)
        self.export_type = kwargs.get('export_type', [ExportType.Db, ExportType.Text])
        self.host_sys = kwargs.get('host_sys', [])
        self.record_shapes = kwargs.get('record_shapes', False)
        self.mstx_domain_include = kwargs.get('mstx_domain_include', [])
        self.mstx_domain_exclude = kwargs.get('mstx_domain_exclude', [])
        super().__init__()
        self.start_step = self.skip_first
        self.stop_step = self.skip_first + self.repeat * (self.wait + self.warmup + self.active)
        self.profiler = ms.profiler.profile(start_profile=self.start_profile, activities=self.activities,
                                            with_stack=self.with_stack,
                                            profile_memory=self.profile_memory, data_process=self.data_process,
                                            parallel_strategy=self.parallel_strategy,
                                            hbm_ddr=self.hbm_ddr, pcie=self.pcie,
                                            record_shapes=self.record_shapes,
                                            on_trace_ready=ms.profiler.tensorboard_trace_handler(
                                                dir_name=self.dir_name, worker_name=self.worker_name,
                                                analyse_flag=self.analyse_flag,
                                                async_mode=self.async_mode),
                                            schedule=ms.profiler.schedule(
                                                wait=self.wait, warmup=self.warmup, active=self.active,
                                                repeat=self.repeat, skip_first=self.skip_first),
                                            experimental_config=ms.profiler._ExperimentalConfig(
                                                profiler_level=self.profiler_level, mstx=True,
                                                data_simplification=self.data_simplification,
                                                aic_metrics=self.aic_metrics, l2_cache=self.l2_cache,
                                                export_type=self.export_type, host_sys=self.host_sys,
                                                mstx_domain_include=self.mstx_domain_include,
                                                mstx_domain_exclude=self.mstx_domain_exclude))
        data = {"world_size": 2, "sequence_parallel": False, "hooks": "dajl"}
        self.profiler.add_metadata("gsfa", "13")
        self.profiler.add_metadata("q3123", "12%")
        self.profiler.add_metadata_json("distribute_args", json.dumps(data))

    def on_train_step_begin(self, run_context):
        """step_begin"""
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step:
            self.profiler.start()

    def on_train_step_end(self, run_context):
        """epoch_stop"""
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if self.start_step <= step_num <= self.stop_step:
            self.profiler.step()
        if step_num == self.stop_step:
            self.profiler.stop()
            run_context.request_stop()


class StopAtEpoch(Callback):
    """for onconditions_ascend StopAtEpoch."""

    def __init__(self, start_epoch, stop_epoch, **kwargs):
        """init"""
        self.profiler_dir = kwargs.get('profiler_dir', './profiler_data')
        self.start_profile = kwargs.get('start_profile', False)
        self.profile_communication = kwargs.get('profile_communication', False)
        self.profile_memory = kwargs.get('profile_memory', False)
        super().__init__()
        self.start_epoch = start_epoch
        self.stop_epoch = stop_epoch
        self.profiler = Profiler(output_path=self.profiler_dir,
                                 start_profile=self.start_profile,
                                 profile_communication=self.profile_communication,
                                 profile_memory=self.profile_memory)

    def on_train_epoch_begin(self, run_context):
        """epoch_begin"""
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        if epoch_num == self.start_epoch:
            self.profiler.start()

    def on_train_epoch_end(self, run_context):
        """epoch_stop"""
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        if epoch_num == self.stop_epoch:
            self.profiler.stop()
            self.profiler.analyse()


class StopAtStep(Callback):
    """for onconditions_ascend StopAtStep"""

    def __init__(self, start_step, stop_step, **kwargs):
        """init"""
        self.profiler_dir = kwargs.get('profiler_dir', './profiler_data')
        self.start_profile = kwargs.get('start_profile', False)
        self.profile_communication = kwargs.get('profile_communication', False)
        self.profile_memory = kwargs.get('profile_memory', False)
        super().__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        self.profiler = Profiler(output_path=self.profiler_dir,
                                 start_profile=self.start_profile,
                                 profile_communication=self.profile_communication,
                                 profile_memory=self.profile_memory)

    def on_train_step_begin(self, run_context):
        """step_begin"""
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step:
            self.profiler.start()

    def on_train_step_end(self, run_context):
        """epoch_stop"""
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.stop_step:
            self.profiler.stop()
            self.profiler.analyse()


class DynProfileCtrler(threading.Thread):
    """DynProfileCtrler"""

    def __init__(self, cfg_path, delay_list, cfg_list):
        super().__init__()
        self.cfg_path = cfg_path
        self.delay_list = delay_list
        self.cfg_list = cfg_list
        self._is_running = True

    def run(self):
        """run"""
        for i, delay in enumerate(self.delay_list):
            while not os.path.exists(self.cfg_path) and self._is_running:
                time.sleep(1)
            time.sleep(delay)
            with open(self.cfg_path, 'r', encoding='utf-8') as f:
                src_cfg = json.load(f)
                for k, v in self.cfg_list[i].items():
                    src_cfg[k] = v
            with open(self.cfg_path, 'w', encoding='utf-8') as f2:
                json.dump(src_cfg, f2)

    def finish(self):
        """set is running"""
        self._is_running = False


def check_is_directory_empty(path):
    """check is directory empty."""
    if len(os.listdir(path)) == 0:
        raise FileNotFoundError(f"{path}下目录为空")


def check_file_not_empty(file_path):
    """check file not empty"""
    if os.path.getsize(file_path) == 0:
        raise FileNotFoundError(f"{file_path}文件为空")


def get_directory_from_check_dir(file_dir_path):
    """Retrieve the directory under the specified path, without retrieving subdirectories."""
    directories = []
    for entry in os.scandir(file_dir_path):
        if entry.is_dir():
            directories.append(entry.path)
    return directories


def find_files_with_start_string(directory, start_string, file_format):
    """Retrieve files starting with a specified character, used for obtaining file paths related to mindata."""
    get_result = []
    for root, dirs, files in os.walk(directory):
        logger.warning(dirs)
        for file in files:
            if file.startswith(start_string) and file.endswith(file_format):
                file_path = os.path.abspath(os.path.join(root, file))
                logger.warning(f"file path is {file_path}")
                if os.path.exists(file_path):
                    get_result.append(file_path)
    if not get_result:
        raise FileNotFoundError(f"No files found with start string '{start_string}' and "
                                f"format '{file_format}' in directory '{directory}'")
    return get_result


def check_profiler_path(file_name, directory):
    """check profiler path"""
    full_path = ''
    if not os.path.isdir(directory):
        return False
    for root, _, _ in os.walk(directory):
        profiler_dir_name = "ASCEND_PROFILER_OUTPUT"
        if os.path.basename(root) == profiler_dir_name:
            full_path = os.path.join(root, file_name)
            logger.warning(full_path)
    assert full_path != ''
    return full_path


def check_profiler_path_out(file_name, directory):
    """check profiler path out"""
    get_paths = ''
    for root, _, files in os.walk(directory):
        if file_name in files:
            get_paths = os.path.join(root, file_name)
            check_file_not_empty(get_paths)
    assert get_paths != ''
    return get_paths


def find_matching_keys(data, pattern):
    """find matching keys"""
    matching_keys = []

    def search_keys(obj, pattern):
        """search keys"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if re.match(pattern, key):
                    matching_keys.append(key)
                search_keys(value, pattern)
        elif isinstance(obj, list):
            for item in obj:
                search_keys(item, pattern)

    search_keys(data, pattern)
    return matching_keys


def get_child_value(data, key, sub_key):
    """get child value"""
    if key in data:
        if sub_key in data[key]:
            return data[key][sub_key]
        return None
    for _, v in data.items():
        if isinstance(v, dict):
            results = get_child_value(v, key, sub_key)
            if results is not None:
                return results
    return None


def run_train_cmd(command):
    """run train cmd"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.info(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}: {e.stderr.strip()}")
        return None


def set_pythonpath(execute_path):
    """set PYTHONPATH"""
    env_vars = os.environ
    if "PYTHONPATH" in env_vars:
        env_vars["PYTHONPATH"] += ":" + execute_path
    else:
        env_vars["PYTHONPATH"] = execute_path
    os.environ = env_vars


def check_host_info_file(output_path):
    """ check host_info_{}.csv,host_memory_{}.csv file"""
    df = check_profiler_path("dataset.csv", output_path)
    assert os.path.exists(df)
    dataset_info = pd.read_csv(df)

    headers = ['Operation']

    for header in headers:
        if dataset_info[header].isnull().values.any():
            return False
        return True


def check_communication_step_id(communication_files, step_numbers, actives_num):
    with open(communication_files, 'r', encoding='utf-8') as files:
        json_data = json.load(files)
    found_count = sum(1 for key in json_data if key in step_numbers)
    logger.warning(f"found_count is {found_count}")
    return found_count == actives_num


def analyse_time(file_path):
    """analyse time"""
    # analyse time
    pattern = re.compile(r'Done Elapsed:\s*(\d+)s')

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            match = pattern.search(content)
            if match:
                return match.group(1)

            return None
    except FileNotFoundError:
        logger.warning(f"文件 {file_path} 未找到")
        return None
    except Exception as e:  # pylint: disable=W0718
        logger.warning(f"处理文件时发生错误：{e}")
        return None


def get_directory_size(path):
    """get directory size"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        logging.warning(dirnames)
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def check_msleaks_dump(out_path):
    """check msleaks dump"""
    check_leaks_dump = find_files_with_start_string(out_path, "leaks_dump_", ".csv")
    logger.warning(f"check_leaks_dump is {check_leaks_dump}")
    dump_file_size = os.path.getsize(check_leaks_dump[0])
    assert int(dump_file_size) > 0
    table_head = ["ID", "Event", "Event Type", "Name", "Timestamp(us)", "Process Id", "Thread Id", "Device Id",
                  "Ptr", "Attr"]

    with open(check_leaks_dump[0], 'r', encoding='utf-8') as f:
        table = csv.reader(f)
        act_table_head = next(table)
        if not table_head[:10] == act_table_head[:10]:
            return False

    kernel_details_data = pd.read_csv(check_leaks_dump[0])
    header_check = "Event Type"
    char_value = "Mindspore"
    values_get = set(kernel_details_data[header_check].tolist())
    if char_value not in values_get:
        logger.warning('values_get check_kernel_details: {}'.format(values_get))
        return False
    return True


def check_cpu_trace(out_path):
    """check cpu trace"""
    check_cpu_path = find_files_with_start_string(out_path, "cpu_trace_", ".json")
    logger.warning(f"check_cpu_trace is {check_cpu_path}")
    cpu_file_size = os.path.getsize(check_cpu_path[0])
    assert int(cpu_file_size) > 0

    target_chars = ["pin memory size", "memory size"]
    with open(check_cpu_path[0], 'r', encoding='utf-8') as files:
        data = json.load(files)
        for char in target_chars:
            logger.warning('char check_trace_view: {}'.format(char))
            if char not in json.dumps(data):
                return False
    return True


def check_npu_trace(out_path, nums):
    """check npu trace"""
    for i in range(nums):
        check_npu_trace_path = find_files_with_start_string(out_path, f"npu{i}_trace_", ".json")
        logger.warning(f"check_npu_trace is {check_npu_trace_path}")

        cpu_file_size = os.path.getsize(check_npu_trace_path[0])
        assert int(cpu_file_size) > 0

        target_chars = ["mindspore allocated memory size", "mindspore reserved memory size", "mstx_mark"]
        with open(check_npu_trace_path[0], 'r', encoding='utf-8') as files:
            data = json.load(files)
            for char in target_chars:
                logger.warning('char check_trace_view: {}'.format(char))
                if char not in json.dumps(data):
                    return False
    return True


def check_files_permission_and_size(file_list, expect_mode="640", is_removed=False):
    """
    check permission and size
    :param file_list: file list
    :param expect_mode: permission right
    :return:
    """

    if is_removed:
        for file_path in file_list:
            if os.path.exists(file_path):
                raise Exception(f"file: {file_path} should not exists.")
    else:
        for file_path in file_list:
            if oct(os.stat(file_path).st_mode)[-3:] > expect_mode:
                raise Exception(f"file: {file_path} permission check failed")
            file_size = os.path.getsize(file_path)
            if file_size <= 0:
                raise Exception(f"file: {file_path} is empty")


class MSProfilerChecker:
    """profile check base function"""

    def __init__(self, profile_config, worker_num, **kwargs):
        self.profile_config = profile_config
        self.rank_list = list(range(worker_num))
        self.check_step_id = kwargs.get("check_step_id", [0])
        self.repeat = kwargs.get("repeat", 1)
        self.step_numbers = kwargs.get("step_numbers", ["step0"])
        self.active = kwargs.get("active", 0)

    def __call__(self):
        if ms.get_context("device_target") != 'CPU':
            assert self.check_trace_view()
            assert self.check_step_trace_time()
            assert self.check_operator_memory()
            assert self.check_operator_details()
            assert self.check_op_statistic()
            assert self.check_npu_module_mem()
            assert self.check_memory_record()
            assert self.check_kernel_details()
            assert self.check_api_statistic()
            assert self.check_minddata_pipeline_summary()
            assert self.check_minddata_pipeline_summary_json()
            assert self.check_minddata_pipeline_raw()
            assert self.check_dataset()
            assert self.check_l2_cache()
            assert self.check_communication()
            assert self.check_communication_matrix()
            assert self.check_add_metadata()
            assert self.check_get_export_type()
            assert self.check_nic_file()
            assert self.check_hccs_file()
            assert self.check_pcie_file()
            assert self.check_roce_file()
        else:
            assert self.check_cpu_file()

    def check_trace_view(self):
        """02 not have AI Core Freq， Stars Soc Info"""
        if self.profile_config.get("export_type", "text") in ["text", "dbtext"]:
            trace_view_path = check_profiler_path("trace_view.json", self.profile_config.get("output_path"))
            target_chars = ["MindSpore", "CANN", "Ascend Hardware", "Overlap Analysis"]
            logger.warning(f"trace_view_path is {trace_view_path}")
            with open(trace_view_path, 'r', encoding='utf-8') as files:
                data = files.read()
                if self.profile_config.get("hbm") and self.profile_config.get("pcie"):
                    if JIT_LEVEL == "" or JIT_LEVEL == "O2":
                        append_list = ["HBM", "LLC", "NPU MEM", "HCCS", "DDR"]
                    else:
                        append_list = ["AI Core Freq", "HBM", "LLC", "NPU MEM", "Stars Soc Info", "Acc PMU", "SIO",
                                       "QoS"]
                    target_chars.extend(append_list)
                if self.profile_config.get("heterogeneous"):
                    target_chars.append("CPU OP")
                if len(self.rank_list) > 1:
                    target_chars.append("Communication")
                if self.profile_config.get("with_stack"):
                    append_list = ["mindspore/train/", "mindspore/common/", "mindspore/nn/"]
                    target_chars.extend(append_list)
                if self.profile_config.get("with_stack_not_normal"):
                    append_list = ["RuntimeFramework", "RunGraph"]
                    target_chars.extend(append_list)
                if self.profile_config.get("activities") == "NPU":
                    target_chars.remove("MindSpore")
                if self.profile_config.get("host_sys", []):
                    if "cpu" in self.profile_config.get("host_sys", []):
                        target_chars.append("CPU Usage")
                    if "mem" in self.profile_config.get("host_sys", []):
                        target_chars.append("Memory Usage")
                    if "disk" in self.profile_config.get("host_sys", []):
                        target_chars.append("Disk Usage")
                    if "network" in self.profile_config.get("host_sys", []):
                        target_chars.append("Network Usage")
                    if "osrt" in self.profile_config.get("host_sys", []):
                        target_chars.append("OS Runtime")
                logger.warning('target_chars check_trace_view: {}'.format(target_chars))
                for char in target_chars:
                    if char not in data:
                        logger.warning('char check_trace_view: {} error'.format(char))
                        return False
        return True

    def check_step_trace_time(self):
        """check step trace time"""
        if self.profile_config.get("export_type", "text") in ["text", "dbtext"]:
            step_trace_time_path = check_profiler_path("step_trace_time.csv", self.profile_config.get("output_path"))
            logger.warning(f"step_trace_time_path is {step_trace_time_path}")
            table_head = ["Step", "Computing", "Communication(Not Overlapped)", "Overlapped", "Communication", "Free",
                          "Stage", "Bubble", "Communication(Not Overlapped and Exclude Receive)", "Preparing"]
            with open(step_trace_time_path, 'r', encoding='utf-8') as f:
                table = csv.reader(f)
                act_table_head = next(table)
                if not table_head[:9] == act_table_head[:9]:
                    return False
                table_first_line = next(table)

                if '' in table_first_line:
                    return False
                if not all(float(table_first_line[i]) >= 0 for i in range(0, 9)):
                    return False
                if self.profile_config.get("activities") != "NPU":
                    if float(table_first_line[9]) <= 0:
                        return False

        return True

    def check_operator_memory(self):
        """check operator memory"""
        if (self.profile_config.get("export_type", "text") != "db" and self.profile_config.get("profile_memory") and
                self.profile_config.get("activities", "CPU_NPU") != "NPU"):
            table_head = ["Name", "Size(KB)", "Allocation Time(us)", "Release Time(us)", "Active Release Time(us)",
                          "Duration(us)", "Active Duration(us)", "Allocation Total Allocated(MB)",
                          "Allocation Total Reserved(MB)", "Allocation Total Active(MB)", "Release Total Allocated(MB)",
                          "Release Total Reserved(MB)",
                          "Release Total Active(MB)", "Stream Ptr", "Device Type"]
            operator_memory_path = check_profiler_path("operator_memory.csv", self.profile_config.get("output_path"))
            logger.warning(f"operator_memory_path is {operator_memory_path}")
            operator_list = []
            with open(operator_memory_path, 'r', encoding='utf-8') as f:
                table = csv.reader(f)
                act_table_head = next(table)
                if not table_head[:9] == act_table_head[:9]:
                    return False

                for line in table:
                    operator_list.append(line[0])
            if not any(self.check_fullscope_name(x) for x in operator_list):
                return False
        return True

    @staticmethod
    def check_fullscope_name(target_str):
        """check fullscope name"""
        return re.search(r'^[a-zA-Z0-9]+', target_str)

    def check_op_statistic(self):
        """check op statistic"""
        if (self.profile_config.get("profile_level") != 0 and
                self.profile_config.get("export_type", "text") in ["text", "dbtext"]):
            table_head = ["Device_id", "OP Type", "Core Type", "Count", "Total Time(us)", "Min Time(us)",
                          "Avg Time(us)", "Max Time(us)", "Ratio(%)"]
            op_statistic_path = check_profiler_path("op_statistic.csv", self.profile_config.get("output_path"))
            logger.warning(f"op_statistic_path is {op_statistic_path}")
            with open(op_statistic_path, 'r', encoding='utf-8') as f:
                table = csv.reader(f)
                act_table_head = next(table)
                if not table_head[:9] == act_table_head[:9]:
                    return False
                for line in table:
                    if int(line[0]) < 0 or int(line[3]) < 1 or any(float(i) < 0 for i in line[4:]):
                        return False
        return True

    def check_npu_module_mem(self):
        """check npu module mem"""
        if (self.profile_config.get("profile_level") != 0 and self.profile_config.get("profile_memory") and
                self.profile_config.get("export_type", "text") in ["text", "dbtext"]):
            table_head = ["Device_id", "Component", "Timestamp(us)", "Total Reserved(KB)", "Device"]
            npu_module_mem_path = check_profiler_path("npu_module_mem.csv", self.profile_config.get("output_path"))
            logger.warning(f"npu_module_mem_path is {npu_module_mem_path}")
            with open(npu_module_mem_path, 'r', encoding='utf-8') as f:
                table = csv.reader(f)
                act_table_head = next(table)
                if not table_head[:5] == act_table_head[:5]:
                    return False
            dataset_info = pd.read_csv(npu_module_mem_path)

            for i in range(5):
                if dataset_info[table_head[i]].isnull().values.any():
                    logger.warning('dataset_info check_npu_module_mem: {}'.format(dataset_info[table_head[i]]))
                    return False

        return True

    def check_memory_record(self):
        """check memory record"""
        if (self.profile_config.get("profile_memory") and
                self.profile_config.get("export_type", "text") in ["text", "dbtext"]):
            table_head = ["Component", "Timestamp(us)", "Total Allocated(MB)", "Total Reserved(MB)", "Total Active(MB)",
                          "Device Type"]
            memory_record_path = check_profiler_path("memory_record.csv", self.profile_config.get("output_path"))
            logger.warning(f"memory_record_path is {memory_record_path}")
            with open(memory_record_path, 'r', encoding='utf-8') as f:
                table = csv.reader(f)
                act_table_head = next(table)
                if not table_head[:6] == act_table_head[:6]:
                    return False
        return True

    def check_operator_details(self):
        """check operator details"""
        if self.profile_config.get("record_shapes"):
            table_head = ["Name", "Input Shapes"]
            operatoer_details_path = check_profiler_path("operator_details.csv", self.profile_config.get("output_path"))
            logger.warning(f"memory_record_path is {operatoer_details_path}")
            with open(operatoer_details_path, 'r', encoding='utf-8') as f:
                table = csv.reader(f)
                act_table_head = next(table)
                if not table_head[:2] == act_table_head[:2]:
                    return False
            dataset_info = pd.read_csv(operatoer_details_path)

            if dataset_info[table_head[0]].isnull().values.any():
                logger.warning('check values operator_details_path: {}'.format(dataset_info[table_head[0]]))
                return False

        return True

    def check_data_preprocess(self):
        """check data preprocess"""
        if (self.profile_config.get("aicpu_ops") is True and
                self.profile_config.get("export_type", "text") in ["text", "dbtext"]):
            table_head = ["Device_id", "Timestamp(us)", "Node", "Compute_time(us)", "Memcpy_time(us)", "Task_time(us)",
                          "Dispatch_time(us)", "Total_time(us)", "Stream ID", "Task ID"]
            data_preprocess_path = check_profiler_path("data_preprocess.csv", self.profile_config.get("output_path"))
            with open(data_preprocess_path, 'r', encoding='utf-8') as f:
                table = csv.reader(f)
                act_table_head = next(table)
                if not table_head[:10] == act_table_head[:10]:
                    return False
            dataset_info = pd.read_csv(data_preprocess_path)
            for i in range(10):
                if dataset_info[table_head[i]].isnull().values.any():
                    logger.warning('dataset_info check_data_preprocess: {}'.format(dataset_info[table_head[i]]))
                    return False
        return True

    def check_kernel_details(self):
        """
        910A does not have Mix Block Dim, 'Context ID' is not supported, verify shared fields;
        The step ID is only generated when the step is specified in the dynamic profiler of the dynamic graph and in
        the for loop scenario of the dynamic graph scene;
        The data_preprocessing file is only generated when the AI CPU is included in the Accelerator Core;
        When ACTIVATE is npu, there is no step ID stored and displayed; The graph mode does not store step IDs,
        but will display them; kernel&communicate
        """
        if self.profile_config.get("export_type", "text") in ["text", "dbtext"]:
            if self.profile_config.get("profile_level") == 0:
                table_head = ["Model ID", "Task ID", "Stream ID", "Name", "Type", "OP State", "Accelerator Core",
                              "Start Time(us)", "Duration(us)", "Wait Time(us)", "Block Dim"]
            else:
                table_head = ["Model ID", "Task ID", "Stream ID", "Name", "Type", "OP State", "Accelerator Core",
                              "Start Time(us)", "Duration(us)", "Wait Time(us)", "Block Dim", "Mix Block Dim",
                              "HF32 Eligible", "Input Shapes", "Input Data Types", "Input Formats", "Output Shapes",
                              "Output Data Types", "Output Formats"]
                if de != "ascend910b":
                    table_head.remove("Mix Block Dim")
            kernel_details_path = check_profiler_path("kernel_details.csv", self.profile_config.get("output_path"))
            logger.warning(f"kernel_details_path is {kernel_details_path}")
            kernel_details_data = pd.read_csv(kernel_details_path)
            act_table_head = kernel_details_data.columns.tolist()
            if self.profile_config.get("dynamic_shape"):
                header_check = "OP State"
                char_value = "dynamic"
                values_get = set(kernel_details_data[header_check].tolist())
                if char_value not in values_get:
                    logger.warning('values_get check_kernel_details: {}'.format(values_get))
                    return False
            if self.profile_config.get("pynative_step") and not (
                    JIT_LEVEL == "O2" or self.profile_config.get("activities") == "NPU"):
                header_check = "Step ID"
                values_get = set(kernel_details_data[header_check].tolist())
                logger.warning(f"get step id {values_get}")
                logger.warning(f"set step id {set(self.check_step_id)}")
                table_head.insert(0, header_check)
                if self.repeat <= 1:
                    assert set(self.check_step_id) == values_get
                else:
                    assert values_get.issubset(self.check_step_id)
            if self.profile_config.get("aicpu_ops"):
                header_check = "Accelerator Core"
                char_value = "AI_CPU"
                values_get = set(kernel_details_data[header_check].tolist())
                if char_value not in values_get and not self.check_data_preprocess():
                    logger.warning('values_get AI_CPU check_kernel_details: {}'.format(values_get))
                    return False
            if self.profile_config.get('aic_metrics') == 'memoryaccess' and self.profile_config.get(
                    "profile_level") != 0:
                memoryaccess_list = ['Context ID', 'aicore_time(us)', 'aic_total_cycles',
                                     'aic_read_main_memory_datas(KB)',
                                     'aic_write_main_memory_datas(KB)', 'aic_GM_to_L1_datas(KB)',
                                     'aic_L0C_to_L1_datas(KB)',
                                     'aic_L0C_to_GM_datas(KB)', 'aic_GM_to_UB_datas(KB)', 'aic_UB_to_GM_datas(KB)',
                                     'aiv_time(us)', 'aiv_total_cycles', 'aiv_read_main_memory_datas(KB)',
                                     'aiv_write_main_memory_datas(KB)', 'aiv_GM_to_L1_datas(KB)',
                                     'aiv_L0C_to_L1_datas(KB)',
                                     'aiv_L0C_to_GM_datas(KB)', 'aiv_GM_to_UB_datas(KB)', 'aiv_UB_to_GM_datas(KB)']
                table_head.extend(memoryaccess_list)
            range_id = len(table_head)

            if self.profile_config.get("aicore_metrics", "AiCoreNone") != "AiCoreNone":
                if not table_head[:10] == act_table_head[:10]:
                    logger.error(f"{table_head} {act_table_head}列不一致")
                    return False
            else:
                if not table_head[:range_id] == act_table_head[:range_id]:
                    logger.error(f"{table_head} {act_table_head}列不一致")
                    return False

            not_nan_headers = ["Model ID", "Name", "Start Time(us)", "Duration(us)",
                               "Wait Time(us)", "Block Dim"]
            if self.profile_config.get("profile_level") != 0:
                add_list_name = ["Type", "Accelerator Core"]
                not_nan_headers.extend(add_list_name)
            for headers in not_nan_headers:
                if any(kernel_details_data[headers].isnull().tolist()):
                    logger.error(f"{kernel_details_path} {headers}列存在空值")
                    return False
        return True

    def check_api_statistic(self):
        """api_statistic.csv"""
        if self.profile_config.get("export_type", "text") in ["text", "dbtext"]:
            table_head = ["Device_id", "Level", "API Name", "Time(us)", "Count", "Avg(us)", "Min(us)", "Max(us)",
                          "Variance"]
            api_statistic_path = check_profiler_path("api_statistic.csv", self.profile_config.get("output_path"))
            logger.warning(f"api_statistic_path is {api_statistic_path}")

            with open(api_statistic_path, 'r', encoding='utf-8') as f:
                table = csv.reader(f)
                act_table_head = next(table)
                if not table_head[:9] == act_table_head[:9]:
                    return False
            dataset_info = pd.read_csv(api_statistic_path)
            for i in range(9):
                if dataset_info[table_head[i]].isnull().values.any():
                    logger.warning('dataset_info check_api_statistic: {}'.format(dataset_info[table_head[i]]))
                    return False

        return True

    def check_minddata_pipeline_summary(self):
        """minddata_pipeline_summary_{self.check_id}.csv"""
        if self.profile_config.get("minddata") and self.profile_config.get("export_type", "text") in ["text", "dbtext"]:
            minddata_pipeline_summary_path = find_files_with_start_string(
                self.profile_config.get("output_path"), "minddata_pipeline_summary_", ".csv")
            specified_list = ['op_ids', 'op_names', 'pipeline_ops', 'num_workers', 'queue_average_size',
                              'queue_utilization_pct',
                              'queue_empty_freq_pct', 'children_ids', 'parent_id', 'avg_cpu_pct', 'per_pipeline_time',
                              'per_push_queue_time',
                              'per_batch_time', 'avg_cpu_pct_per_worker', 'cpu_analysis_details',
                              'queue_analysis_details', 'bottleneck_warning', 'bottleneck_suggestion']
            with open(minddata_pipeline_summary_path[0], encoding='utf-8') as csv_file:
                table = csv.reader(csv_file)
                for row in table:
                    if row and row[0] not in specified_list:
                        logger.warning('check_minddata_pipeline_summary row[0]: {}'.format(row[0]))
                        return False
        return True

    def check_minddata_pipeline_summary_json(self):
        """minddata_pipeline_summary_{self.check_id}.json"""
        if self.profile_config.get("minddata") and self.profile_config.get("export_type", "text") in ["text", "dbtext"]:
            check_minddata_pipeline_summary_json_path = find_files_with_start_string(
                self.profile_config.get("output_path"), "minddata_pipeline_summary_", ".json")
            logger.warning("check minddata_pipeline_summary_json list")
            specified_list = ['op_ids', 'op_names', 'pipeline_ops', 'num_workers', 'queue_average_size',
                              'queue_utilization_pct', 'queue_empty_freq_pct', 'children_ids', 'parent_id',
                              'avg_cpu_pct', 'per_pipeline_time', 'per_push_queue_time', 'per_batch_time',
                              'avg_cpu_pct_per_worker',
                              'cpu_analysis_details', 'queue_analysis_details', 'bottleneck_warning',
                              'bottleneck_suggestion']
            with open(check_minddata_pipeline_summary_json_path[0], 'r', encoding='utf-8') as files:
                data = json.load(files)
                logger.warning(data)
                for key in data.keys():
                    if key not in specified_list:
                        logger.warning("Unexpected key found in minddata_pipeline_summary_json: %s", key)
                        return False
        return True

    def check_minddata_pipeline_raw(self):
        """minddata_pipeline_raw_0.csv"""
        if self.profile_config.get("minddata") and self.profile_config.get("export_type", "text") in ["text", "dbtext"]:
            table_head = ["op_id", "op_type", "num_workers", "output_queue_size", "output_queue_average_size",
                          "output_queue_length", "output_queue_usage_rate", "sample_interval",
                          "parent_id", "children_id"]
            minddata_pipeline_raw_path = find_files_with_start_string(
                self.profile_config.get("output_path"), "minddata_pipeline_raw_", ".csv")

            logger.warning(f"minddata_pipeline_raw is {minddata_pipeline_raw_path}")

            with open(minddata_pipeline_raw_path[0], 'r', encoding='utf-8') as f:
                table = csv.reader(f)
                act_table_head = next(table)
                if not table_head[:10] == act_table_head[:10]:
                    return False
                for line in table:
                    if int(line[0]) < 0 or int(line[2]) < 1:
                        return False
        return True

    def check_l2_cache(self):
        """l2_cache.csv"""

        if (de != "ascend910b" and JIT_LEVEL != "O2" and self.profile_config.get("l2_cache")
                and self.profile_config.get("export_type", "text") in ["text", "dbtext"]):

            table_head = ["Device_id", "Stream Id", "Task Id", "Hit Rate", "Victim Rate", "Op Name"]
            l2_cache_path = check_profiler_path("l2_cache.csv", self.profile_config.get("output_path"))
            logger.warning(f"l2_cache_path is {l2_cache_path}")

            with open(l2_cache_path, 'r', encoding='utf-8') as f:
                table = csv.reader(f)
                act_table_head = next(table)
                if not table_head[:6] == act_table_head[:6]:
                    return False
            dataset_info = pd.read_csv(l2_cache_path)
            for i in range(3):
                if dataset_info[table_head[i]].isnull().values.any():
                    logger.warning('check_l2_cache dataset_info: {}'.format(dataset_info[table_head[i]]))
                    return False
        return True

    def check_dataset(self):
        """dataset.csv"""
        if self.profile_config.get("export_type", "text") in ["text", "dbtext"]:
            if not self.profile_config.get("minddata"):
                return True
            table_head = ["Operation", "Stage", "Occurrences", "Avg. time (us)", "Custom Info"]
            dataset_path = check_profiler_path("dataset.csv", self.profile_config.get("output_path"))
            logger.warning(f"dataset_path is {dataset_path}")

            with open(dataset_path, 'r', encoding='utf-8') as f:
                table = csv.reader(f)
                act_table_head = next(table)
                if not table_head[:5] == act_table_head[:5]:
                    return False
        return True

    def check_communication(self):
        """check communication"""
        if (self.profile_config.get("profile_level") != 0 and len(self.rank_list) > 1 and
                self.profile_config.get("export_type", "text") in ["text", "dbtext"]):
            communication_path = check_profiler_path("communication.json", self.profile_config.get("output_path"))
            logger.warning(f"communication_path is {communication_path}")
            with open(communication_path, 'r', encoding='utf-8') as f:
                communication_res = json.load(f)
            check_res = []

            check_res.append(get_child_value(communication_res, 'Communication Time Info', 'Start Timestamp(us)') >= 0)
            check_res.append(get_child_value(communication_res, 'Communication Time Info', 'Elapse Time(ms)') >= 0)
            check_res.append(get_child_value(communication_res, 'Communication Time Info', 'Transit Time(ms)') >= 0)
            check_res.append(get_child_value(communication_res, 'Communication Time Info', 'Wait Time(ms)') >= 0)
            check_res.append(
                get_child_value(communication_res, 'Communication Time Info', 'Synchronization Time(ms)') >= 0)
            check_res.append(get_child_value(communication_res, 'Communication Time Info', 'Idle Time(ms)') >= 0)
            check_res.append(get_child_value(communication_res, 'Communication Time Info', 'Wait Time Ratio') >= 0)
            check_res.append(
                get_child_value(communication_res, 'Communication Time Info', 'Synchronization Time Ratio') >= 0)
            check_res.append(get_child_value(communication_res, 'RDMA', 'Transit Size(MB)') >= 0)
            check_res.append(get_child_value(communication_res, 'RDMA', 'Transit Time(ms)') >= 0)
            check_res.append(get_child_value(communication_res, 'RDMA', 'Bandwidth(GB/s)') >= 0)
            check_res.append(get_child_value(communication_res, 'RDMA', 'Large Packet Ratio') >= 0)
            check_res.append(get_child_value(communication_res, 'HCCS', 'Transit Size(MB)') >= 0)
            check_res.append(get_child_value(communication_res, 'HCCS', 'Transit Time(ms)') >= 0)
            check_res.append(get_child_value(communication_res, 'HCCS', 'Bandwidth(GB/s)') >= 0)
            check_res.append(get_child_value(communication_res, 'HCCS', 'Large Packet Ratio') >= 0)
            check_res.append(get_child_value(communication_res, 'PCIE', 'Transit Size(MB)') >= 0)
            check_res.append(get_child_value(communication_res, 'PCIE', 'Transit Time(ms)') >= 0)
            check_res.append(get_child_value(communication_res, 'PCIE', 'Bandwidth(GB/s)') >= 0)
            check_res.append(get_child_value(communication_res, 'PCIE', 'Large Packet Ratio') >= 0)
            check_res.append(get_child_value(communication_res, 'SDMA', 'Transit Size(MB)') >= 0)
            check_res.append(get_child_value(communication_res, 'SDMA', 'Transit Time(ms)') >= 0)
            check_res.append(get_child_value(communication_res, 'SDMA', 'Bandwidth(GB/s)') >= 0)
            check_res.append(get_child_value(communication_res, 'SDMA', 'Large Packet Ratio') >= 0)
            check_res.append(get_child_value(communication_res, 'SIO', 'Transit Size(MB)') >= 0)
            check_res.append(get_child_value(communication_res, 'SIO', 'Transit Time(ms)') >= 0)
            check_res.append(get_child_value(communication_res, 'SIO', 'Bandwidth(GB/s)') >= 0)
            check_res.append(get_child_value(communication_res, 'SIO', 'Large Packet Ratio') >= 0)
            if not all(check_res):
                return False
            if self.profile_config.get("schedule") and not (JIT_LEVEL == "O2" or
                                                            self.profile_config.get("activities") == "NPU"):
                assert check_communication_step_id(communication_path, self.step_numbers, self.active)

        return True

    def check_communication_matrix(self):
        """check communication matrix"""
        if (self.profile_config.get("profile_level") != 0 and len(self.rank_list) > 1 and
                self.profile_config.get("export_type", "text") in ["text", "dbtext"]):
            communication_matrix_path = check_profiler_path("communication_matrix.json",
                                                            self.profile_config.get("output_path"))
            logger.warning(f"communication_matrix_path is {communication_matrix_path}")
            with open(communication_matrix_path, 'r', encoding='utf-8') as f:
                communication_res = json.load(f)
            re_par = r"^\w+?\-(\w+?)@\d+$"
            matching_keys = find_matching_keys(communication_res, re_par)
            logger.warning(matching_keys)
            length_matching_keys = len(matching_keys)
            for i in range(int(length_matching_keys)):
                order_type = re.findall(r"^\w+?\-(\w+?)@\d+$", matching_keys[i])[0]
                if order_type not in ["top1", "middle", "total", "bottom1", "bottom2", "bottom3"]:
                    logger.warning(order_type)
                    return False
            re_par2 = r"^\d+?\-\d+?$"
            matching_keys2 = find_matching_keys(communication_res, re_par2)
            logger.warning(matching_keys2)
            list_ckeck = ["Transit Size(MB)", "Transit Time(ms)", "Bandwidth(GB/s)"]
            length_matching_keys2 = len(matching_keys2)
            length_list_ckeck = len(list_ckeck)
            for i in range(int(length_matching_keys2)):
                for j in range(int(length_list_ckeck)):
                    list_ckeck_value = get_child_value(communication_res, str(matching_keys2[i]), str(list_ckeck[j]))
                    assert list_ckeck_value >= 0
            if self.profile_config.get("schedule") and not (JIT_LEVEL == "O2" or
                                                            self.profile_config.get("activities") == "NPU"):
                assert check_communication_step_id(communication_matrix_path, self.step_numbers, self.active)
        return True

    def check_add_metadata(self):
        """check add metadata"""
        file_list = []
        if (self.profile_config.get("add_metadata") and
                self.profile_config.get("export_type", "text") in ["text", "dbtext"]):
            add_metadata_jsons_path = check_profiler_path_out("profiler_metadata.json",
                                                              self.profile_config.get("output_path"))
            logger.warning(f"add_metadata_jsons_path：{add_metadata_jsons_path}")
            file_list.extend([add_metadata_jsons_path])
            check_files_permission_and_size(file_list)

        return True

    def check_get_export_type(self):
        """check get export type"""
        file_list = []
        if self.profile_config.get("export_type", "text") in ["db", "dbtext"]:
            export_type_path = find_files_with_start_string(
                self.profile_config.get("output_path"), "ascend_mindspore_profiler", ".db")
            file_list.extend([export_type_path])
            logger.warning(f"export filer list {file_list}")
            length_file = len(file_list)
            for i in range(length_file):
                check_files_permission_and_size(file_list[i])

        return True

    def check_cpu_file(self):
        """check cpu file"""
        file_list = []
        if ms.get_context("device_target") == "CPU":
            cpu_framework = check_profiler_path_out("cpu_framework_0.txt", self.profile_config.get("output_path"))
            cpu_ms_memory_record = check_profiler_path_out("cpu_ms_memory_record_0.txt",
                                                           self.profile_config.get("output_path"))
            cpu_op_detail_info = check_profiler_path_out("cpu_op_detail_info_0.csv",
                                                         self.profile_config.get("output_path"))
            cpu_op_execute_timestamp = check_profiler_path_out("cpu_op_execute_timestamp_0.txt",
                                                               self.profile_config.get("output_path"))
            cpu_op_type_info = check_profiler_path_out("cpu_op_type_info_0.csv",
                                                       self.profile_config.get("output_path"))
            file_list.extend([cpu_framework, cpu_ms_memory_record, cpu_op_detail_info,
                              cpu_op_execute_timestamp, cpu_op_type_info])
            check_files_permission_and_size(file_list)
        return True

    def check_nic_file(self):
        """check nic file"""
        if self.profile_config.get("sys_io"):
            table_head = ["Device_id", "Timestamp(us)", "Bandwidth(MB/s)", "Rx Bandwidth efficiency(%)", "rxPacket/s",
                          "rxError rate(%)", "rxDropped rate(%)", "Tx Bandwidth efficiency(%)", "txPacket/s",
                          "txError rate(%)", "txDropped rate(%)", "funcId"]
            nic_file_path = check_profiler_path("nic.csv", self.profile_config.get("output_path"))
            logger.warning(f"nic_file_path is {nic_file_path}")

            with open(nic_file_path, 'r', encoding='utf-8') as f:
                table = csv.reader(f)
                act_table_head = next(table)
                if not table_head[:12] == act_table_head[:12]:
                    return False
            dataset_info = pd.read_csv(nic_file_path)
            for i in range(12):
                logger.warning('nic_file_path values: {}'.format(dataset_info[table_head[i]]))
                if dataset_info[table_head[i]].isnull().values.any():
                    return False
        return True

    def check_roce_file(self):
        """check roce file"""
        if self.profile_config.get("sys_io"):
            table_head = ["Device_id", "Timestamp(us)", "Bandwidth(MB/s)", "Rx Bandwidth efficiency(%)", "rxPacket/s",
                          "rxError rate(%)", "rxDropped rate(%)", "Tx Bandwidth efficiency(%)", "txPacket/s",
                          "txError rate(%)", "txDropped rate(%)", "funcId"]
            roce_file_path = check_profiler_path("roce.csv", self.profile_config.get("output_path"))
            logger.warning(f"nic_file_path is {roce_file_path}")

            with open(roce_file_path, 'r', encoding='utf-8') as f:
                table = csv.reader(f)
                act_table_head = next(table)
                if not table_head[:12] == act_table_head[:12]:
                    return False
            dataset_info = pd.read_csv(roce_file_path)
            for i in range(12):
                logger.warning('nic_file_path values: {}'.format(dataset_info[table_head[i]]))
                if dataset_info[table_head[i]].isnull().values.any():
                    return False
        return True

    def check_pcie_file(self):
        """check pcie file"""
        if self.profile_config.get("sys_interconnection") or self.profile_config.get("pcie"):
            table_head = ["Device_id", "Mode", "Min", "Max", "Avg"]
            pcie_file_path = check_profiler_path("pcie.csv", self.profile_config.get("output_path"))
            logger.warning(f"pcie_file_path is {pcie_file_path}")

            with open(pcie_file_path, 'r', encoding='utf-8') as f:
                table = csv.reader(f)
                act_table_head = next(table)
                if not table_head[:5] == act_table_head[:5]:
                    return False
            dataset_info = pd.read_csv(pcie_file_path)
            for i in range(5):
                if dataset_info[table_head[i]].isnull().values.any():
                    logger.warning('pcie_file_path values: {}'.format(dataset_info[table_head[i]]))
                    return False
        return True

    def check_hccs_file(self):
        """check hccs file"""
        if self.profile_config.get("sys_interconnection") or self.profile_config.get("pcie"):
            table_head = ["Device_id", "Mode", "Max", "Min", "Average"]
            hccs_file_path = check_profiler_path("hccs.csv", self.profile_config.get("output_path"))
            logger.warning(f"hccs_file_path is {hccs_file_path}")

            with open(hccs_file_path, 'r', encoding='utf-8') as f:
                table = csv.reader(f)
                act_table_head = next(table)
                if not table_head[:5] == act_table_head[:5]:
                    return False
            dataset_info = pd.read_csv(hccs_file_path)
            for i in range(5):
                if dataset_info[table_head[i]].isnull().values.any():
                    logger.warning('hccs_file_path values: {}'.format(dataset_info[table_head[i]]))
                    return False
        return True


class TimeLineChecker:
    """check time."""

    def __init__(self, timeline_file):
        with open(timeline_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.dataframe = pd.DataFrame(data)

    def check_mstx_dot(self, host_to_python=True, python_to_device=False):
        """check mstx dot"""
        if host_to_python and self.dataframe[self.dataframe['name'] == "mindspore_to_mstx"].empty:
            return False
        if python_to_device and \
                self.dataframe[self.dataframe['cat'] == "MsTx"][self.dataframe['ph'] == "f"][
                    self.dataframe['name'] != "mindspore_to_mstx"].empty:
            return False
        return True

    def check_mstx_msg(self, msg_list):
        """check mstx msg."""
        for msg_name in msg_list:
            if self.dataframe[self.dataframe['name'] == msg_name].empty:
                return False
        return True

    def check_keyword_msg(self, key_name, msg_list):
        """check keyword msg."""
        for msg_name in msg_list:
            if self.dataframe[self.dataframe[key_name] == msg_name].empty:
                return False
        return True

    def check_mstx_msg_dict(self, msg_list):
        """check mstx msg dict."""
        columns = self.dataframe.to_dict(orient='records')
        for msg in msg_list:
            for column in columns:
                if column['name'] == msg and msg in ['dataloader', 'save_checkpoint']:
                    if 'domain' not in column['args'].keys():
                        return False
                    if column['args']['domain'] != 'default':
                        return False
                if msg == 'communication':
                    if (isinstance(column['args'], dict) and len(column['args']) > 0 and msg in
                            column['args'].values() and 'domain' in column['args'].keys()):
                        if column['args']['event_type'] != "start/end":
                            return False

        return True
