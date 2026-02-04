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
Test module for dynamic allocator.
"""
import os
import subprocess
from tests.mark_utils import arg_mark


def get_para_cnt(param, log_path):
    para_output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (param, log_path)],
        shell=True)
    para_cnt = str(para_output, 'utf-8').strip()
    return int(para_cnt)


def run_command(cmd, log_path, is_oom, is_two_pointer):
    if os.path.isfile(log_path):
        os.remove(log_path)
    os.system(cmd)

    for line in open(log_path):
        print(line, end='', flush=True)

    if is_two_pointer is False:
        malloc_cnt = get_para_cnt("MallocStaticDevMem success", log_path)
        free_cnt = get_para_cnt("Free memory success", log_path)
        assert malloc_cnt > 0
        assert malloc_cnt == free_cnt
        print("malloc and free:", malloc_cnt, free_cnt, flush=True)
    else:
        two_pointer_cnt = get_para_cnt("ascend_two_pointer_mem_adapter", log_path)
        assert two_pointer_cnt > 0
        print("two_pointer:", two_pointer_cnt, flush=True)

    if is_oom:
        oom_cnt = get_para_cnt("memory isn't enough", log_path)
        assert oom_cnt > 0
        print("oom:", oom_cnt, flush=True)

    if os.path.isfile(log_path):
        os.remove(log_path)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_no_vmm_kbk():
    """
    Feature: Test dynamic allocator
    Description: Test dynamic allocator with no vmm and kbk.
    Expectation: Success
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_command(f"bash {sh_path}/run_no_vmm_1p.sh no_vmm_kbk", f"{sh_path}/no_vmm_kbk.log", False, False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_no_vmm_ge():
    """
    Feature: Test dynamic allocator
    Description: Test dynamic allocator with no vmm and ge.
    Expectation: Success
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_command(f"bash {sh_path}/run_no_vmm_1p.sh no_vmm_ge", f"{sh_path}/no_vmm_ge.log", False, False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_no_vmm_ge_two_pointer():
    """
    Feature: Test two pointer allocator
    Description: Test two pointer allocator with no vmm and ge.
    Expectation: Success
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_command(f"bash {sh_path}/run_no_vmm_1p.sh no_vmm_ge_two_pointer",
                f"{sh_path}/no_vmm_ge_two_pointer.log", False, True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_no_vmm_kbk_mempool_block():
    """
    Feature: Test dynamic allocator
    Description: Test dynamic allocator with no vmm and kbk mempool block.
    Expectation: Success
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_command(f"bash {sh_path}/run_no_vmm_1p.sh no_vmm_kbk_mempool_block",
                f"{sh_path}/no_vmm_kbk_mempool_block.log", False, False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_no_vmm_kbk_oom():
    """
    Feature: Test dynamic allocator
    Description: Test dynamic allocator with no vmm and OOM.
    Expectation: Success
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_command(f"bash {sh_path}/run_no_vmm_1p.sh no_vmm_kbk_oom", f"{sh_path}/no_vmm_kbk_oom.log", False, False)
