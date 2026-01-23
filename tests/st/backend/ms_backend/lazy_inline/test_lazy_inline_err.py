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
"""
Test module for lazy inline err.
"""
import os
from tests.mark_utils import arg_mark


def run_command(cmd, log_path):
    if os.path.isfile(log_path):
        os.remove(log_path)
    os.system(cmd)
    sub_str = "Please remove the @lazy_inline decorator and try again"
    with open(log_path, 'r') as file:
        for line in file:
            if 'RuntimeError' in line:
                assert sub_str in line
    if os.path.isfile(log_path):
        os.remove(log_path)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_lazy_inline_err():
    """
    Feature: lazy inline err
    Description: test lazy inline err
    Expectation: test pass
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_command(f"bash {sh_path}/run.sh", f"{sh_path}/lazy.log")
