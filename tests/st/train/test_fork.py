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
""" test fork. """
import os
import multiprocessing as mp
import platform
import pytest
import mindspore as ms
import mindspore.ops as ops
import numpy as np


def subprocess(mode, subprocess_id):
    print(f"id:{subprocess_id} enter")
    ms.set_context(mode=mode)
    x = ms.Tensor(2, dtype=ms.float32)
    y = ops.log(x)
    assert np.allclose(y.asnumpy(), np.log(2), 1e-3), "subprocess id:{subprocess_id}"
    print(f"id:{subprocess_id} exit")


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fork(mode):
    """
    Feature: Fork test
    Description: Test multiprocessing with fork
    Expectation: No exception
    """
    if platform.system() != 'Linux':
        return
    os.environ['MS_ENABLE_FORK_UTILS'] = '1'
    ms.set_context(mode=mode)
    x = ms.Tensor(2, dtype=ms.float32)
    y = ops.log(x)
    assert np.allclose(y.asnumpy(), np.log(2), 1e-3)

    mp.set_start_method('fork', force=True)
    processes = []
    for i in range(4):
        p = mp.Process(target=subprocess, args=(mode, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join(5)  # timeout:10s
        if p.is_alive() is not True:
            # exitcode info:
            # None: subprocess still running
            # 0: exit successfully
            # 1: exit with exception
            # -N: exit with signal N
            assert p.exitcode == 0
    del os.environ['MS_ENABLE_FORK_UTILS']
