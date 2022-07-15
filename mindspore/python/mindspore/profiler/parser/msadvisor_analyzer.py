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
The msadvisor analyzer.
"""

import os
import subprocess

from mindspore import log as logger
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path
from mindspore.profiler.parser.msadvisor_parser import MsadvisorParser


class Msadvisor:
    """
    The interface to call msadvisor(CANN) by command line.
    """
    def __init__(self, job_id, rank_id, output_path):
        self._job_id, self._device_id = job_id.split("/")
        self._rank_id = rank_id
        self._output_path = output_path

    def call_msadvisor(self):
        """
        Call Msadvisor by command line.
        """
        output_path = os.path.join(self._output_path, "msadvisor")
        output_path = os.path.join(output_path, self._rank_id)
        output_path = validate_and_normalize_path(output_path)
        logger.info("Msadvisor is running. Log and result files are saved in %s", output_path)
        subprocess.run(["msadvisor", "-d", output_path, "-c", "all"])
        logger.info("Msadvisor is over. If command not found, please check if intalled ascend-toolkit"
                    "and add environment path.")

    def analyse(self):
        """
        Execute the msadvisor parser, generate timeline file and call msadvisor by command line.
        """
        reformater = MsadvisorParser(self._job_id, self._device_id, self._rank_id, self._output_path)
        reformater.parse()
        self.call_msadvisor()
