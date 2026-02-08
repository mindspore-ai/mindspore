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
from mindspore.communication.management import init
from mindspore.ops.communication import all_gather_into_tensor, set_comm_ops_inplace
from mindspore import context, jit

np.random.seed(1)
context.set_context(jit_level='O0')
os.environ['HCCL_WHITELIST_DISABLE'] = str(1)
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
init()
#msrun --worker_num=2 --local_worker_num=2 --master_port=1128 --log_dir=msrun_log --join=True --cluster_time_out=300  test_ops_all_gather_into_tensor.py


def test_hccl_all_gather_into_tensor_func_dynstatic1():
    """
    Feature: test 'all_gather_into_tensor' communication function.
    Description: test 'all_gather_into_tensor' communication function.
    Expectation: expect correct result.
    """
    @jit(jit_level="O0")
    def test(out_tensor, in_tensor):
        """
        Feature: test 'all_gather_into_tensor' communication function.
        Description: test 'all_gather_into_tensor' communication function.
        Expectation: expect correct result.
        """
        output, h = all_gather_into_tensor(out_tensor, in_tensor, async_op=True)
        h.wait()
        return output

    @jit(jit_level="O0")
    def test1(out_tensor, in_tensor):
        """
        Feature: test 'all_gather_into_tensor' communication function.
        Description: test 'all_gather_into_tensor' communication function.
        Expectation: expect correct result.
        """
        output, h = all_gather_into_tensor(out_tensor, in_tensor, async_op=True)
        return output, h

    set_comm_ops_inplace(False)
    context.set_context(save_graphs=3, save_graphs_path="ir_dump_path")
    x = np.ones([3, 4]).astype(np.float32)
    expect_output = np.ones([6, 4]).astype(np.float32)
    output = test(None, Tensor(x, mstype.float32))
    assert np.allclose(output.asnumpy(), expect_output)

    output, h = test1(None, Tensor(x, mstype.float32))
    h.wait()
    assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_all_gather_into_tensor_func_dynstatic2():
    """
    Feature: test 'all_gather_into_tensor' communication function.
    Description: test 'all_gather_into_tensor' communication function.
    Expectation: expect correct result.
    """
    @jit(jit_level="O0")
    def test_inplace(out_tensor, in_tensor):
        """
        Feature: test 'all_gather_into_tensor' communication function.
        Description: test 'all_gather_into_tensor' communication function.
        Expectation: expect correct result.
        """
        h = all_gather_into_tensor(out_tensor, in_tensor, async_op=True)
        if isinstance(h, tuple):
            h[1].wait()
        else:
            h.wait()
        return h

    @jit(jit_level="O0")
    def test_inplace1(out_tensor, in_tensor):
        """
        Feature: test 'all_gather_into_tensor' communication function.
        Description: test 'all_gather_into_tensor' communication function.
        Expectation: expect correct result.
        """
        h = all_gather_into_tensor(out_tensor, in_tensor, async_op=True)
        return h

    set_comm_ops_inplace(True)
    context.set_context(save_graphs=3, save_graphs_path="ir_dump_path")
    x = Tensor(np.ones([6, 40]).astype(np.float32), mstype.float32)
    y = Tensor(np.zeros([12, 40]).astype(np.float32), mstype.float32)
    expect_output = np.ones([12, 40]).astype(np.float32)
    h = test_inplace(y, x)
    assert np.allclose(y.asnumpy(), expect_output)
    y = Tensor(np.zeros([12, 40]).astype(np.float32), mstype.float32)
    h = test_inplace1(y, x)
    if isinstance(h, tuple):
        h[1].wait()
    else:
        h.wait()
    assert np.allclose(y.asnumpy(), expect_output)
