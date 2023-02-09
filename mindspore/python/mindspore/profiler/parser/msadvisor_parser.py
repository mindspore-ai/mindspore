# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
MSAdvisor AICPU model parser.
"""

import os
import stat
import shutil
import json

from mindspore import log as logger
from mindspore.profiler.common.exceptions.exceptions import ProfilerFileNotFoundException
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path

MIN_TO_US = 60000000  # 1 min to us
MS_TO_US = 1000  # 1 ms to us
AICPU_STREAM_ID = 9000  # aicpu stream id in profiler


class MsadvisorParser:
    """
    Data format conversion for MSAdvisor AICPU model.
    """

    def __init__(self, job_id, device_id, rank_id, output_path):
        self._job_id = job_id
        self._device_id = device_id
        self._rank_id = rank_id
        self._output_path = output_path
        self._aicore_path = ""
        self._aicpu_path = ""
        self._time_start = 0
        self._time_end = 0

    @staticmethod
    def check_clear_make_dir(dir_path):
        """
        Check if dir exists, then clear the dir and make a new dir.

        Args:
            dir_path (str): dir path is needed to clear and make.

        Return:
            str, new dir path.
        """
        dir_path = validate_and_normalize_path(dir_path)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, stat.S_IRWXU)
        return dir_path

    @staticmethod
    def generate_aicore_json(aicore_info, tid):
        """
        Generate dict of operation information which be dumped into json file.

        Args:
            aicore_info (str): str read from aicore timeline file.
            tid (int): Task Id.

        Return:
            dict, dict of operation information which can be dumped into json file.

        Raises:
            ValueError: If the value of aicore attrributes cannot be converted to float.
        """
        op = aicore_info.split(",")
        name = op[0]
        pid = 0
        tid = tid - 1
        task_type = "AI_CORE"

        try:
            ts, dur, sid = float(op[2]) * MS_TO_US, float(op[3]) * MS_TO_US, float(op[1])
        except ValueError as err:
            logger.warning("The aicore timeline file content is abnormal. Failed to format aicore timeline file")
            raise err
        finally:
            pass

        op = {"name": name, "pid": pid, "ts": ts, "dur": dur,
              "args": {"Task Type": task_type, "Stream Id": sid, "Task Id": tid}, "ph": "X"}
        return op

    @staticmethod
    def generate_aicpu_json(aicpu_info, tid):
        """
        Generate dict of operation information which be dumped into json file.

        Args:
            aicpu_info (str): str read from aicpu timeline file.
            tid (int): Task Id.

        Return:
            dict, dict of operation information which can be dumped into json file.

        Raises:
            ValueError: If the value of aicpu attrributes cannot be converted to float.
        """
        op = aicpu_info.split(",")
        name = op[1]
        pid = 1
        sid = AICPU_STREAM_ID
        tid = tid - 1
        task_type = "AI_CPU"

        try:
            ts = float(op[5])
            dur = float(op[4]) * MS_TO_US
        except ValueError as err:
            logger.warning("The aicpu timeline file content is abnormal. Failed to format aicpu timeline file")
            raise err
        finally:
            pass

        op = {"name": name, "pid": pid, "ts": ts, "dur": dur,
              "args": {"Task Type": task_type, "Stream Id": sid, "Task Id": tid}, "ph": "X"}
        return op

    def get_input_file(self):
        """
        Get aicore and aicpu information file from specific path and rank id.

        Raises:
            ProfilerFileNotFoundException: If aicore timeline file does not exist.
            ProfilerFileNotFoundException: If aicpu timeline file does not exist.
        """
        self._aicore_path = "output_timeline_data_{}.txt".format(self._rank_id)
        self._aicore_path = os.path.join(self._output_path, self._aicore_path)
        self._aicore_path = validate_and_normalize_path(self._aicore_path)

        self._aicpu_path = "aicpu_intermediate_{}.csv".format(self._rank_id)
        self._aicpu_path = os.path.join(self._output_path, self._aicpu_path)
        self._aicpu_path = validate_and_normalize_path(self._aicpu_path)

        if not os.path.exists(self._aicore_path):
            logger.warning('The aicore timeline file does not exist!')
            raise ProfilerFileNotFoundException(msg=self._aicore_path)
        if not os.path.exists(self._aicpu_path):
            logger.warning('The aicpu timeline file does not exist!')
            raise ProfilerFileNotFoundException(msg=self._aicpu_path)

    def get_output_file(self):
        """
        Get output path needed by MSAdvisor and created dir.
        """
        msprof_file = os.path.join(self._output_path, "msadvisor")
        msprof_file = os.path.join(msprof_file, "device_" + self._rank_id)
        msprof_file = os.path.join(msprof_file, "profiling")
        msprof_file = MsadvisorParser.check_clear_make_dir(msprof_file)

        msprof_file = os.path.join(msprof_file, self._job_id)
        msprof_file = os.path.join(msprof_file, "device_0", "timeline")
        msprof_file = validate_and_normalize_path(msprof_file)
        os.makedirs(msprof_file, stat.S_IRWXU)

        msprof_file = os.path.join(msprof_file, "task_time_0_1_1.json")
        self._output_path = msprof_file

    def write_aicore(self):
        """
        Read aicore information from file created by profiler and generate new file needed by MSAdvisor.
        """
        aicore_file = self._aicore_path
        output_file = self._output_path

        with os.fdopen(os.open(output_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                               stat.S_IRUSR | stat.S_IWUSR), "w") as output_file:
            output_file.write("[")
            with os.fdopen(os.open(aicore_file, os.O_RDONLY,
                                   stat.S_IRUSR | stat.S_IWUSR), "r") as aicore_file:
                for tid, aicore in enumerate(aicore_file):
                    if tid == 0:
                        continue
                    op = MsadvisorParser.generate_aicore_json(aicore, tid)
                    if tid == 1:
                        self._time_start = op.get("ts")
                    total_duration = op.get("ts") - self._time_start
                    if total_duration > 1 * MIN_TO_US or tid > 10000:
                        self._time_end = op.get("ts")
                        break
                    if tid > 1:
                        output_file.write(",")
                    json.dump(op, output_file)

    def write_aicpu(self):
        """
        Read aicpu information from file created by profiler and write into new file needed by MSAdvisor.
        """
        aicpu_file = self._aicpu_path
        output_file = self._output_path

        with os.fdopen(os.open(output_file, os.O_WRONLY | os.O_APPEND,
                               stat.S_IRUSR | stat.S_IWUSR), "a") as output_file:
            with os.fdopen(os.open(aicpu_file, os.O_RDONLY,
                                   stat.S_IRUSR | stat.S_IWUSR), "r") as aicpu_file:
                for tid, aicpu in enumerate(aicpu_file):
                    if tid == 0:
                        continue
                    op = MsadvisorParser.generate_aicpu_json(aicpu, tid)
                    if op is None:
                        continue
                    if op.get("ts") > self._time_end:
                        break
                    output_file.write(",")
                    json.dump(op, output_file)
            output_file.write("]")

    def parse(self):
        """
        Interface to call all function in the class. Generated data for AICpu model in MSAdvisor.
        """
        self.get_input_file()
        self.get_output_file()
        self.write_aicore()
        self.write_aicpu()
