# Copyright 2025 Huawei Technologies Co., Ltd
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
"""profiler with acl ops."""
import os
import tempfile

from tests.mark_utils import arg_mark
from tests.st.tools.profiler.daily_test.profiler_check import MSProfilerChecker


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_func_profiler_env_acl_unique_ops_01():
    """
    Feature: Profiler ACL Operations
    Description: Test profiler with ACL unique operations and dynamic pipeline.
    Expectation: Generate profiling data with memory, L2 cache, HBM, and PCIe metrics.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ret = os.system(f"python ./run_net_with_dynamic_getnext_pipeline.py --output_path {tmpdir}")
        assert ret == 0
        prof_config = {"output_path": tmpdir, "profiler_memory": True, "profiler_level": 2, "l2_cache": True,
                       "hbm": True, "pcie": True}
        prof_check = MSProfilerChecker(prof_config, 1)
        prof_check()
