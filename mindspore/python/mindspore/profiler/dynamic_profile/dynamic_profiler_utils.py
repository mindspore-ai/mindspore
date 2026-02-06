# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""Dynamic Monitor proxy"""
import os
from enum import Enum
from mindspore import log as logger
from mindspore.profiler.common.constant import DynoMode
from mindspore.communication import get_rank


class DynamicProfilerUtils:
    """
    Class for dynamic profiler utils.
    """
    CFG_BUFFER_SIZE = 1024 * 1024

    class ProfilerStatus(Enum):
        UNINITIALIZED = -1
        IDLE = 0
        RUNNING = 1
        READY = 2

    PROFILER_STATUS = "profiler_status"
    CURRENT_STEP = "current_step"
    START_STEP = "start_step"
    STOP_STEP = "stop_step"
    REPORT_INTERVAL = 1.0

    @classmethod
    def is_dyno_mode(cls):
        """Check whether it is dyno mode"""
        dyno_enable_flag = os.getenv(DynoMode.DYNO_DAEMON, "0")
        try:
            dyno_enable_flag = int(dyno_enable_flag)
        except ValueError:
            logger.error(f"Environment variable '{DynoMode.DYNO_DAEMON}' value not valid, will be set to 0 !")
            dyno_enable_flag = 0

        return dyno_enable_flag == 1

    @classmethod
    def get_real_rank(cls):
        """get rank id"""
        try:
            return get_rank()
        except RuntimeError:
            return int(os.getenv("RANK_ID", "0"))

    @classmethod
    def dyno_str_to_dict(cls, res: str):
        """ Convert dyno str to json """
        res_dict = {}
        pairs = str(res).split("\n")
        char_equal = '='
        for pair in pairs:
            str_split = pair.split(char_equal)
            if len(str_split) == 2:
                if any(keyword in str_split[0] for keyword in ["PROFILE", "ACTIVITIES"]):
                    res_dict[str_split[0].strip().split('_', 1)[-1].lower()] = str_split[1].strip()
                else:
                    res_dict[str_split[0].strip()] = str_split[1].strip()

        return res_dict
