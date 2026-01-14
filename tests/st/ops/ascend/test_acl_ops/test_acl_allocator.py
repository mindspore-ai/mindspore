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
Test module for parallel training of Llama models using Mindformers at jit_level O2.
"""
import os
import subprocess
from tests.mark_utils import arg_mark


def run_command(cmd, log_path, enable_acl_allocator):
    if os.path.isfile(log_path):
        os.remove(log_path)
    os.system(cmd)

    log_para = "Register AclAllocator"
    log_output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (log_para, log_path)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    print(log_cnt, flush=True)
    if enable_acl_allocator:
        assert log_cnt != str(0)
    else:
        assert log_cnt == str(0)

    if os.path.isfile(log_path):
        os.remove(log_path)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_disable_acl_allocator():
    """
    Feature: Acl Allocator Config
    Description: test disable acl allocator
    Expectation: no register acl allocator
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    test_case = "test_disable_acl_allocator"
    run_command(f"bash {sh_path}/test_config_acl_allocator/{test_case}.sh {test_case}.log",
                f"{sh_path}/{test_case}.log", False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_enable_acl_allocator():
    """
    Feature: Acl Allocator Config
    Description: test enable acl allocator
    Expectation: register acl allocator
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    test_case = "test_enable_acl_allocator"
    run_command(f"bash {sh_path}/test_config_acl_allocator/{test_case}.sh {test_case}.log",
                f"{sh_path}/{test_case}.log", True)
