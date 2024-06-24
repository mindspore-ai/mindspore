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
import glob
import shutil
import json
from json import JSONDecodeError
from collections import defaultdict
from subprocess import CalledProcessError, TimeoutExpired
from subprocess import Popen, PIPE
import csv
from mindspore import log as logger
from mindspore.profiler.common.util import get_newest_file


class AscendMsprofExporter:
    """
    msprof exporter. export cann edge profiling data.

    args:
       source_path: the root path of PROF_* files

    files under prof_root_dir is like:
        profiler/PROF_*/device_{id}/data/xxx
        profiler/PROF_*/host/data/xxx
        ...

    usage:
        >> ms_exporter = AscendMsprofExporter("path/to/profiler/data")
        >> ms_exporter.export(start_time)
    """
    DRV_VERSION = 467473
    _hiai_msprof_tail = "Ascend/latest/tools/profiler/bin"
    _msprof_cmd = "msprof"
    _ascend_mark = "Ascend"
    _step_trace_mark = "step_trace"
    _op_summary_mark = "op_summary"
    _op_statistic_mark = "op_statistic"

    def __init__(self, mindstudio_profiler_output, time_out=3600):
        self._time_out = time_out
        self.mindstudio_profiler_output = mindstudio_profiler_output  # mindstudio_profiler_output dir
        # PROF* dir
        self.prof_root_dir = os.path.abspath(os.path.join(self.mindstudio_profiler_output, os.path.pardir))

        AscendMsprofExporter.check_msprof_env()

    @classmethod
    def check_msprof_env(cls):
        """Check the existence of msprof binary tool"""

        def _check_msprof(temp_path: str):
            if not os.path.isdir(temp_path):
                return False
            sub_files = os.listdir(temp_path)
            if cls._msprof_cmd in sub_files:
                return True
            return False

        outs, _ = cls.run_cmd(["which", cls._msprof_cmd])
        if outs != "":
            return
        logger.warning("[Profiler]The msprof command was not found. Searching from environment variables...")

        msprof_path = None
        envs = os.environ
        if envs.get("ASCEND_TOOLKIT_HOME"):
            temp_path = os.path.join(envs.get("ASCEND_TOOLKIT_HOME"), "bin")
            if _check_msprof(temp_path):
                msprof_path = temp_path

        if not msprof_path and envs.get("PATH"):
            path_list = envs.get("PATH").split(":")
            for path in path_list:
                if cls._ascend_mark in path:
                    prefix = path.split(cls._ascend_mark)[0]
                    temp_path = os.path.join(prefix, cls._hiai_msprof_tail)
                    if _check_msprof(temp_path):
                        msprof_path = temp_path
                        break
        if msprof_path:
            envs["PATH"] = msprof_path + ":" + envs.get("PATH", "")
        else:
            raise FileNotFoundError("The msprof command was not found!")

        logger.info("The msprof command has been added to the path!")

    @classmethod
    def get_msprof_script_path(cls, script_path=""):
        """Check the existence of msprof script"""
        if not script_path:
            return ""
        outs, _ = AscendMsprofExporter.run_cmd(["which", cls._msprof_cmd])
        if not outs:
            return ""
        msprof_path = os.path.realpath(outs.strip())
        sup_path = msprof_path.split("tools")[0]
        script_path = os.path.join(sup_path, script_path)
        if not os.path.exists(script_path):
            return ""
        return script_path

    @classmethod
    def get_msprof_info_path(cls):
        """Check the existence of get_msprof_info.py script"""
        return cls.get_msprof_script_path("tools/profiler/profiler_tool/analysis/interface/get_msprof_info.py")

    @classmethod
    def get_msprof_py_script_path(cls):
        """Check the existence of msprof.py script"""
        return cls.get_msprof_script_path("tools/profiler/profiler_tool/analysis/msprof/msprof.py")

    @classmethod
    def run_cmd(cls, cmd, timeout=300):
        """run shell command"""
        try:
            proc = Popen(cmd, stdout=PIPE, stderr=PIPE, text=True)
        except (FileNotFoundError, PermissionError, CalledProcessError) as exc:
            raise RuntimeError(exc) from exc
        try:
            outs, errs = proc.communicate(timeout=timeout)
        except TimeoutExpired as err:
            proc.kill()
            msg = "The possible cause is that too much data is collected " \
                  "and the export time is too long."
            logger.error(msg)
            raise TimeoutError(msg) from err
        logger.info(outs)
        return outs, errs

    def get_drv_version(self):
        """Get the drv_version for choosing the export mode."""
        try:
            script_path = AscendMsprofExporter.get_msprof_info_path()
            if not script_path:
                logger.warning("Can`t find get_msprof_info.py path, use single-export mode instead.")
                return False
            logger.info("get_msprof_info.py path is : %s", script_path)
            host_dir = os.path.join(self.prof_root_dir, 'host')
            cmd = ['python', script_path, '-dir', host_dir]
            outs, _ = AscendMsprofExporter.run_cmd(cmd)
            if not outs:
                logger.warning('Check the drvVersion can`t find the result, use single-export mode instead.')
                return False
            result = json.loads(outs)
            logger.info('get drv_version result is : %s', result)
            status = result.get('status', 1)
            if status == 1:
                return False
            drv_version = result.get('data', {}).get('version_info', {}).get('drv_version', 0)
            if drv_version >= self.DRV_VERSION:
                return True
            return False
        except (RuntimeError, JSONDecodeError, AttributeError, TimeoutError) as err:
            logger.warning('Get the drvVersion error, use single-export mode instead. detail : %s', err)
            return False

    def export(self, model_iteration_dict=None):
        """start_time is the time to collect PROF data"""

        flag = self.get_drv_version()
        if not flag or model_iteration_dict:
            flag = False
            if not model_iteration_dict:
                model_iteration_dict = self._generate_step_trace()

            if model_iteration_dict:
                for model_id, iter_list in model_iteration_dict.items():
                    self._run_msprof_export_cmd(self.prof_root_dir, model_id, iter_list)
        else:
            msprof_export_cmd = self._msprof_command_generator(self.prof_root_dir)
            AscendMsprofExporter.run_cmd(msprof_export_cmd, self._time_out)

        self._check_export_files()

        msprof_analyze_cmd = [self._msprof_cmd, "--analyze=on", "--rule=communication,communication_matrix",
                              "--output={}".format(self.prof_root_dir)]
        AscendMsprofExporter.run_cmd(msprof_analyze_cmd, self._time_out)

        return flag

    def _run_msprof_export_cmd(self, prof_root_dir, model_id, iter_list):
        """run msprof.py export cmd"""
        script_path = AscendMsprofExporter.get_msprof_py_script_path()
        if not script_path:
            raise FileNotFoundError("Can not find msprof.py path, please check the cann environment.")
        export_cmd = ['python', script_path]
        iter_param = []
        if isinstance(model_id, int) and model_id >= 0:
            iter_param.extend(["--model-id", str(model_id)])
        if iter_list and isinstance(iter_list, list):
            iter_list.sort()
            iter_param.extend(["--iteration-id", str(iter_list[0]), "--iteration-count", str(len(iter_list))])
        for export_type in ("timeline", "summary"):
            cmd = export_cmd + ["export", export_type, "-dir", prof_root_dir] + iter_param
            AscendMsprofExporter.run_cmd(cmd, self._time_out)

    def _msprof_command_generator_old(self, output, model_id=None, iter_id=None):
        """msprof export helper"""
        export_cmd = [self._msprof_cmd, "--export=on", "--output={}".format(output)]
        if isinstance(model_id, int) and model_id >= 0:
            export_cmd.append("--model-id={}".format(model_id))
        if isinstance(iter_id, int) and iter_id >= 0:
            export_cmd.append("--iteration-id={}".format(iter_id))
        return export_cmd

    def _msprof_command_generator(self, output):
        """msprof export helper"""
        return [self._msprof_cmd, "--export=on", "--output={}".format(output)]

    def _generate_step_trace(self):
        """"generate model_id iteration_id dict"""

        AscendMsprofExporter.run_cmd(self._msprof_command_generator_old(self.prof_root_dir), self._time_out)

        if not os.path.isdir(self.mindstudio_profiler_output):
            msg = "Path {} is not a existing directory. Make sure there is " \
                  "valid profiling data directory!".format(self.mindstudio_profiler_output)
            raise FileNotFoundError(msg)

        step_trace_name = fr'{self.mindstudio_profiler_output}/step_trace_*.csv'
        step_trace_file = get_newest_file(glob.glob(step_trace_name))

        if not step_trace_file:
            logger.info("Do not found step trace csv file in {} .".format(self.mindstudio_profiler_output))
            return None

        step_trace = defaultdict(list)
        with os.fdopen(os.open(step_trace_file[0], os.O_RDONLY, 0o600), newline='', mode='r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for index, value in enumerate(next(reader)):
                if value == 'Model ID':
                    model_id = index
                if value == 'Iteration ID':
                    iteration_id = index
            for row in reader:
                step_trace[int(row[model_id])].append(int(row[iteration_id]))

        shutil.rmtree(self.mindstudio_profiler_output)

        return step_trace

    def _check_export_files(self):
        """Check the existence of op_summary & op_statistic files."""
        if not os.path.isdir(self.mindstudio_profiler_output):
            raise RuntimeError("Path {} is not a existing directory.".format(self.mindstudio_profiler_output))
        op_summary = set()
        op_statistic = set()
        msprof_json = set()

        for f in os.listdir(self.mindstudio_profiler_output):
            if f.endswith('.csv'):
                if f.startswith(self._op_summary_mark):
                    op_summary.add(f)
                elif f.startswith(self._op_statistic_mark):
                    op_statistic.add(f)

            elif f.endswith('.json') and f.startswith('msprof'):
                msprof_json.add(f)

        if not op_summary:
            logger.warning("The op_summary csv file was not found, perhaps the original data was not collected.")
        if not op_statistic:
            logger.warning("The op_statistics csv file was not found, perhaps the original data was not collected.")
        if not msprof_json:
            logger.warning("The msprof json file was not found, perhaps the original data was not collected.")

        logger.info("Finish checking files.")
