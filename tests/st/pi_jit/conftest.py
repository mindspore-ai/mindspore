# Copyright 2020-2025 Huawei Technologies Co., Ltd
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
@File   : conftest.py
@Desc   : common fixtures for pytest
"""

import os
import pytest
import subprocess
import sys
from typing import Dict

from mindspore._c_expression import update_pijit_default_config
from mindspore.graph._parse import compile_config


def pytest_runtest_setup(item):
    if sys.version_info >= (3, 11):
        pytest.skip("Skipping PIJit tests for Python >= 3.11.")
    update_pijit_default_config(compile_with_try=False)
    compile_config.JIT_ENABLE_AUGASSIGN_INPLACE = '0'


def run_in_subprocess(env_vars: Dict[str, str], *, timeout: int = 60):
    """
    A decorator that runs testcase in a subprocess with specific environment variables.
    This decorator is helpful for debugging randomly failing testcases.
    You can change the log level to INFO or DEBUG by setting the environment variable
    'GLOG_v' before executing the testcase.
    """
    assert isinstance(env_vars, dict), "env_vars must be a dictionary"

    def decorator(func):
        func.__run_in_subprocess__ = True
        func.__env_vars__ = env_vars
        func.__timeout__ = timeout
        return func

    return decorator


@pytest.fixture(autouse=True)
def _run_in_subprocess_if_decorated(request):
    """
    If the test function is decorated with @run_in_subprocess, run the test in
    a subprocess with the specific environment variables.
    If the testcase failed in subprocess, the output will be printed to the console.
    """
    if (
        not hasattr(request.node.function, "__run_in_subprocess__")
        or os.getenv('_MS_PYTEST_FIXTURE_IN_SUBPROCESS', '') == '1'
    ):
        yield
        return

    env_vars = getattr(request.node.function, "__env_vars__", {})
    # It is a guard to prevent recursive call (where testing in a subprocess re-triggers the subprocess)
    env_vars['_MS_PYTEST_FIXTURE_IN_SUBPROCESS'] = '1'

    sub_env = os.environ.copy()
    sub_env.update(env_vars)

    timeout = getattr(request.node.function, "__timeout__", 60)

    cmd = [sys.executable, "-m", "pytest", request.node.nodeid, "-s"]
    try:
        subprocess.run(
            cmd,
            env=sub_env,
            timeout=timeout,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            cwd=str(request.config.rootdir),
        )
    except subprocess.TimeoutExpired as e:
        print(f"\n--- Subprocess output for {request.node.nodeid} ---\n")
        print(e.stdout, flush=True)
        pytest.fail(f"Test execution in subprocess timed out after {timeout} seconds.", pytrace=False)
    except subprocess.CalledProcessError as e:
        print(f"\n--- Subprocess output for {request.node.nodeid} ---\n")
        print(e.stdout, flush=True)
        pytest.fail(f"Test failed in subprocess with retcode {e.returncode}. See the subprocess output.", pytrace=False)

    pytest.skip("This test was successfully executed in a subprocess.")
