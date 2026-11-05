# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Regression for loading complete JIT hook graphs from the compilation cache."""
import os
from pathlib import Path
import subprocess
import sys

from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux', 'platform_ascend910b'], level_mark='level1', card_mark='onecard', mem_peak=0.01)
def test_compile_cache_grad_jit_with_hook(tmp_path):
    """
    Feature: Compile cache for JIT gradient hooks.
    Description: Execute a hook with control flow in two fresh processes sharing the cache.
    Expectation: Both gradients are correct; the second process loads caches without recompilation.
    """
    script = Path(__file__).with_name("run_compile_cache_hook.py")
    cache_path = tmp_path / "cache"
    env = os.environ.copy()
    env.update(MS_COMPILER_CACHE_ENABLE="1", MS_COMPILER_CACHE_PATH=str(cache_path), GLOG_v="2")
    for run in (1, 2):
        log_path = tmp_path / f"run_{run}.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run([sys.executable, str(script)], env=env, cwd=tmp_path,
                                    stdout=log_file, stderr=subprocess.STDOUT, check=False, timeout=180)
        log = log_path.read_text(encoding="utf-8")
        assert result.returncode == 0, log
        assert "hook gradient: [4]" in log
        if run == 1:
            assert list(cache_path.rglob("*.mindir")), log
            assert "Use the compilation cache and execute the backend actions only." not in log
        else:
            assert "Use the compilation cache and execute the backend actions only." in log
            assert "Failed to load the compilation cache file" not in log
            assert "Execute all the compilation actions" not in log
