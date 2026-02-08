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

"""test hccl AllGather and all_gather with 8p"""

import os
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.communication.management import init, get_group_size
from mindspore.ops.communication import reduce_scatter, set_comm_ops_inplace
from mindspore import context, jit

np.random.seed(1)
context.set_context(jit_level='O0')
os.environ['HCCL_WHITELIST_DISABLE'] = str(1)
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
init()
#msrun --worker_num=2 --local_worker_num=2 --master_port=1128 --log_dir=msrun_log --join=True --cluster_time_out=300  test_ops_reduce_scatter.py
size = get_group_size()


def test_hccl_reduce_scatter_func_dynstatic1():
    """
    Feature: test 'reduce_scatter' communication function.
    Description: test 'reduce_scatter' communication function.
    Expectation: expect correct result.
    """
    @jit(jit_level="O0")
    def test_inplace(out_tensor, in_tensor):
        """
        Feature: test 'reduce_scatter' communication function.
        Description: test 'reduce_scatter' communication function.
        Expectation: expect correct result.
        """
        h = reduce_scatter(out_tensor, in_tensor, async_op=True)
        if isinstance(h, tuple):
            h[1].wait()
        else:
            h.wait()
        return h

    @jit(jit_level="O0")
    def test_inplace1(out_tensor, in_tensor):
        """
        Feature: test 'reduce_scatter' communication function.
        Description: test 'reduce_scatter' communication function.
        Expectation: expect correct result.
        """
        h = reduce_scatter(out_tensor, in_tensor, async_op=True)
        return h

    set_comm_ops_inplace(True)
    context.set_context(save_graphs=3, save_graphs_path="ir_dump_path")
    input_tensor = []
    for _ in range(size):
        input_tensor.append(Tensor(np.ones([3, 40]).astype(np.float32)))
    y = Tensor(np.zeros([3, 40]).astype(np.float32), mstype.float32)
    expect_output = np.ones([3, 40]).astype(np.float32) * 2
    h = test_inplace(y, input_tensor)
    assert np.allclose(y.asnumpy(), expect_output)
    y = Tensor(np.zeros([3, 40]).astype(np.float32), mstype.float32)
    h = test_inplace1(y, input_tensor)
    if isinstance(h, tuple):
        h[1].wait()
    else:
        h.wait()
    assert np.allclose(y.asnumpy(), expect_output)
