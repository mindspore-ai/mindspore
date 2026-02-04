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

def run_command_semi_compile(cmd, log_path, backend_time):
    if os.path.isfile(log_path):
        os.remove(log_path)
    os.system(cmd)

    log_backend = "compile_backend_graph costs"
    log_output = subprocess.check_output(
        ["grep -r '%s' %s | head -1 | awk '{print $3}'" % (log_backend, log_path)],
        shell=True)
    log_time = str(log_output, 'utf-8').strip()
    assert float(log_time) <= backend_time * 1.1


def run_command_auto_compile(cmd, log_path, sharding_time):
    if os.path.isfile(log_path):
        os.remove(log_path)
    os.system(cmd)

    log_sharding = "parallel_strategy_search costs"
    log_output = subprocess.check_output(
        ["grep -r '%s' %s | awk '{print $3}'" % (log_sharding, log_path)],
        shell=True)
    log_time = str(log_output, 'utf-8').strip()
    assert float(log_time) <= sharding_time

    if os.path.isfile(log_path):
        os.remove(log_path)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_train_semi_compile():
    """
    Feature: Trainer.train()
    Description: Test llama2 70b semi compile time when parallel_mode=SEMI_AUTO_PARALLEL.
    Expectation: Throw AssertionError when compile_backend_graph time > 60000 ms
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_command_semi_compile(f"bash {sh_path}/dry_compile.sh semi compile", f"{sh_path}/compile.log", 60000)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_train_auto_compile():
    """
    Feature: refactor sharding propagation when AUTO_PARALLEL.
    Description: Test llama2 70b compile time when parallel_mode=AUTO_PARALLEL.
    Expectation: Throw AssertionError when parallel_strategy_search time > 11000 ms
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_command_auto_compile(f"bash {sh_path}/dry_compile.sh auto compile", f"{sh_path}/compile_auto.log", 11000)
