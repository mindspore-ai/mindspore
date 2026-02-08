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

"""test hccl allreduce with 2p profiling"""
import os
import glob
import json
import tempfile
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.context as context
from mindspore import Profiler
from mindspore import Tensor
from mindspore.communication.management import init
from model_zoo import AllReduceNet

np.random.seed(1)
os.environ['HCCL_WHITELIST_DISABLE'] = str(1)
context.set_context(jit_level='O0')
context.set_context(device_target="Ascend")
init()


def test_allreduce():
    """
    Feature: hccl operator test.
    Description: msrun hccl all_reduce 2P case.
    Expectation: success
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = Profiler(output_path=tmpdir)
        net = AllReduceNet()
        input_x = np.ones([3, 4]).astype(np.float32)
        net(Tensor(input_x, mstype.float32))
        profiler.stop()
        ascend_ms_dir = glob.glob(f"{tmpdir}/*_ascend_ms")[0]
        assert "profiler_metadata.json" in os.listdir(ascend_ms_dir)
        with open(os.path.join(ascend_ms_dir, "profiler_metadata.json"), "r") as f:
            data = json.load(f)
            assert "parallel_group_info" in data
            for item in data["parallel_group_info"].values():
                if item["group_name"] == "hccl_world_group":
                    assert len(item.get("global_ranks", [])) == 2
