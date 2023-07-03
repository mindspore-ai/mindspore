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
"""msprof PROF data export api file"""
import os
from subprocess import CalledProcessError, TimeoutExpired
from subprocess import Popen, PIPE
from typing import List
import json
from json.decoder import JSONDecodeError
import csv
import mindspore as ms
from mindspore import log as logger, context
from mindspore.communication import get_rank
from mindspore.profiler.common.util import get_file_path


class AscendMsprofExporter:
    """
    msprof exporter. export cann edge profiling data.

    args:
        prof_root_dir: the root path of PROF_* files

    files under prof_root_dir is like:
        profiler/PROF_*/device_{id}/data/xxx
        profiler/PROF_*/host/data/xxx
        ...

    usage:
        >> ms_exporter = AscendMsprofExporter("path/to/profiler/data")
        >> ms_exporter.export(start_time)
    """

    _null_info = ""
    _csv_header_model_id = "Model ID"
    _csv_header_iter_id = "Iteration ID"
    _profiling_prefix = "PROF"
    _summary_dir = "summary"
    _step_trace_mark = "step_trace"
    _op_summary_mark = "op_summary"
    _op_statistic_mark = "op_statistic"
    _device_mark = "device"
    _msprof_cmd = "msprof"
    _info_prefix = "info.json"
    _start_log = "start_info"
    _rank_id_mark = "rank_id"
    _dev_info = "DeviceInfo"
    _dev_index = 0
    _dev_id = "id"
    _ascend_mark = "Ascend"
    _hiai_msprof_tail = "Ascend/latest/tools/profiler/bin"

    def __init__(self, prof_root_dir: str) -> None:
        self._prof_root_dir = prof_root_dir
        self._start_time = 0
        self._prof_paths = []
        self._output_path = None
        self._device_path = None
        self._model_ids = []
        self._iter_ids = []
        self._mode = ms.get_context("mode")
        self._check_msprof_env()

    @staticmethod
    def _check_readable(file_path: str):
        """Check whether the file is readable"""
        if not os.access(file_path, os.R_OK):
            msg = "The file {} is not readable.".format(file_path)
            raise PermissionError(msg)

    @staticmethod
    def _parse_start_info(input_file: str):
        """Get profiler start time from start info file."""
        start_time = -1
        try:
            with open(input_file, "r") as f:
                start_time = json.load(f).get("collectionTimeBegin")
        except (JSONDecodeError, FileNotFoundError, TypeError, PermissionError) as err:
            logger.warning(err)
        return int(start_time)

    def export(self, start_time=0):
        """start_time is the time to collect PROF data"""
        self._start_time = start_time
        self._init_output_path()
        if not self._output_path:
            raise FileNotFoundError("Do not found valid profiling directory")
        trace_file = self._get_device_trace_file(self._output_path, self._device_path)
        if trace_file and self._mode == context.GRAPH_MODE:
            self._export_whole_prof(self._output_path, trace_file)
            self._check_export_files(self._device_path, trace_file)

    def get_job_dir(self):
        """Return matched PROF directory path. Call this function after exporting profiling data."""
        return self._output_path

    def _run_cmd(self, cmd: List[str], raise_error=True):
        """run msprof tool shell command"""
        try:
            proc = Popen(cmd, stdout=PIPE, stderr=PIPE, text=True)
        except (FileNotFoundError, PermissionError, CalledProcessError) as exc:
            raise RuntimeError(exc)
        try:
            outs, errs = proc.communicate(timeout=300)
        except TimeoutExpired:
            proc.kill()
            msg = "The possible cause is that too much data is collected " \
                "and the export time is too long."
            logger.error(msg)
            raise TimeoutError(msg)
        logger.info(outs)
        if raise_error and errs != self._null_info:
            raise RuntimeError(errs)
        return outs

    def _export_helper(self, **kwargs):
        """msprof export helper"""
        export_cmd = [self._msprof_cmd, "--export=on"]
        output = kwargs.get("output")
        model_id = kwargs.get("model_id")
        iter_id = kwargs.get("iter_id")
        if output:
            export_cmd.append("--output={}".format(output))
        if model_id:
            export_cmd.append("--model-id={}".format(model_id))
        if iter_id:
            export_cmd.append("--iteration-id={}".format(iter_id))
        _ = self._run_cmd(export_cmd)

    def _get_model_iter_ids(self, trace_file: str):
        """Read the step_trace file and get all the model_ids and iteration_ids"""
        if self._model_ids and self._iter_ids:
            return
        self._check_readable(trace_file)
        with open(trace_file, "r") as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx == 0:
                    model_idx = row.index(self._csv_header_model_id)
                    iter_idx = row.index(self._csv_header_iter_id)
                else:
                    self._model_ids.append(int(row[model_idx]))
                    self._iter_ids.append(int(row[iter_idx]))

    def _check_export_files(self, device_path: str, trace_file: str):
        """Check the existence of op_summary & op_statistic files."""
        summary_path = os.path.join(device_path, self._summary_dir)

        if not os.path.isdir(summary_path):
            raise RuntimeError("Path {} is not a existing directory.".format(summary_path))

        sumary_filess = os.listdir(summary_path)

        dev_id = ms.get_context("device_id")
        self._get_model_iter_ids(trace_file)

        op_summary = set()
        op_statistic = set()

        for summary_file in sumary_filess:
            if summary_file.startswith(self._op_summary_mark):
                op_summary.add(summary_file)
            elif summary_file.startswith(self._op_statistic_mark):
                op_statistic.add(summary_file)

        if not op_summary:
            raise RuntimeError("No op_summary file is exported!")
        if not op_statistic:
            raise RuntimeError("No op_statistic file is exported!")

        for model_id, iter_id in zip(self._model_ids, self._iter_ids):
            tag = "_{}_{}_{}.csv".format(dev_id, model_id, iter_id)
            op_sum_file = self._op_summary_mark + tag
            op_sta_file = self._op_statistic_mark + tag
            if op_sum_file not in op_summary:
                logger.warning("The file {} can not be found!".format(op_sum_file))
            if op_sta_file not in op_statistic:
                logger.warning("The file {} can not be found!".format(op_sta_file))

        logger.info("Finish checking files.")

    def _export_whole_prof(self, prof: str, trace_file: str):
        """export all the data under PROF directory"""
        self._get_model_iter_ids(trace_file)

        for model_id, iter_id in zip(self._model_ids, self._iter_ids):
            single_kwargs = {"output": prof, "model_id": model_id, "iter_id": iter_id}
            self._export_helper(**single_kwargs)

    def _get_device_trace_file(self, prof_path: str, device_path: str):
        """search the step trace csv file under device directory"""

        summary_path = os.path.join(device_path, self._summary_dir)

        if not os.path.exists(summary_path):
            default_kwargs = {"output": prof_path}
            self._export_helper(**default_kwargs)

        if not os.path.isdir(summary_path):
            msg = "Path {} is not a existing directory. Make sure there is " \
                "valid profiling data directory!".format(summary_path)
            raise FileNotFoundError(msg)

        step_trace_file = get_file_path(summary_path, self._step_trace_mark)

        if not step_trace_file and self._mode == context.GRAPH_MODE:
            msg = "Do not found step trace csv file in {}.".format(self._output_path)
            raise FileNotFoundError(msg)

        return step_trace_file

    def _check_msprof_env(self):
        """Check the existence of msprof binary tool"""
        msprof_cmd = ["which", self._msprof_cmd]
        outs = self._run_cmd(msprof_cmd, raise_error=False)
        if outs != self._null_info:
            return
        logger.warning("The msprof command was not found. Searching from environment variables...")
        self._search_and_add()

    def _search_and_add(self):
        """Search msprof and add it to PATH"""
        def _check_msprof(temp_path: str):
            if not os.path.isdir(temp_path):
                return False
            sub_files = os.listdir(temp_path)
            if self._msprof_cmd in sub_files:
                return True
            return False

        msprof_path = None
        envs = os.environ
        if envs.get("ASCEND_TOOLKIT_HOME"):
            temp_path = os.path.join(envs.get("ASCEND_TOOLKIT_HOME"), "bin")
            if _check_msprof(temp_path):
                msprof_path = temp_path

        if not msprof_path and envs.get("PATH"):
            path_list = envs.get("PATH").split(":")
            for path in path_list:
                if self._ascend_mark in path:
                    prefix = path.split(self._ascend_mark)[0]
                    tail = self._hiai_msprof_tail
                    temp_path = os.path.join(prefix, tail)
                    if _check_msprof(temp_path):
                        msprof_path = temp_path
                        break
        if msprof_path:
            envs["PATH"] = msprof_path + ":" + envs.get("PATH", "")
        else:
            raise FileNotFoundError("The msprof command was not found!")

        logger.info("The msprof command has been added to the path!")

    def _init_output_path(self):
        """find all the directories start with PROF"""
        self._prof_paths = []

        if not os.path.isdir(self._prof_root_dir):
            msg = "Path {} is not a existing directory.".format(self._prof_root_dir)
            raise RuntimeError(msg)

        for loc_root, loc_dirs, _ in os.walk(self._prof_root_dir):
            for loc_dir in loc_dirs:
                if loc_dir.startswith(self._profiling_prefix):
                    self._prof_paths.append(os.path.join(loc_root, loc_dir))

        if not self._prof_paths:
            msg = "Do not found profiling data. Make sure there are directories start with PROF."
            raise FileNotFoundError(msg)

        # consider there may exists old PROF data, new PROF data have higher priority.
        self._prof_paths.sort()               # sort by created time

        device_path = self._search_by_rank_id(self._prof_paths)
        if device_path:
            dev_par = os.path.join(device_path, os.path.pardir)
            abs_dev_par = os.path.abspath(dev_par)
            self._output_path = abs_dev_par
            self._device_path = device_path

    def _search_by_rank_id(self, prof_paths: str):
        """search valid device path through rank_id"""
        device_paths = []

        for prof_path in prof_paths:
            if not os.path.isdir(prof_path):
                continue
            devices = os.listdir(prof_path)
            for device in devices:
                if not device.startswith(self._device_mark):
                    continue
                dev_path = os.path.join(prof_path, device)
                device_paths.append(dev_path)

        # search by rank id
        find_device_path = None
        rank_id = None
        device_id = ms.get_context("device_id")

        try:
            rank_id = get_rank()
        except RuntimeError:
            logger.warning("Do not get rank_id in the environment variable, use device_id instead.")

        for dev_path in device_paths:
            if not os.path.isdir(dev_path):
                continue
            start_log = get_file_path(dev_path, self._start_log)
            if not start_log:
                continue
            start_time = self._parse_start_info(start_log)
            if start_time < self._start_time:
                continue
            info_json = get_file_path(dev_path, self._info_prefix)
            if not info_json:
                continue
            temp_rank_id, temp_dev_id = self._parse_info_json(info_json)
            if rank_id is not None and rank_id == temp_rank_id:
                find_device_path = dev_path
                break
            if (rank_id is None or temp_rank_id == -1) and device_id == temp_dev_id:
                find_device_path = dev_path

        return find_device_path

    def _parse_info_json(self, info_file: str):
        """get rank_id from info.json.{device_id} file"""
        rank_id = -1
        dev_id = -1
        info_dict = {}
        try:
            with open(info_file, "r") as f:
                info_dict = json.load(f)
        except (JSONDecodeError, FileNotFoundError, TypeError, PermissionError) as err:
            logger.warning(err)
        if info_dict.get(self._rank_id_mark) is None:
            msg = "There is no rank_id key in file {}".format(info_file)
            logger.warning(msg)
        else:
            rank_id = info_dict.get(self._rank_id_mark)

        if not info_dict.get(self._dev_info):
            return rank_id, dev_id

        dev_info = info_dict.get(self._dev_info)
        dev_id = dev_info[self._dev_index].get(self._dev_id, -1)

        return rank_id, dev_id
