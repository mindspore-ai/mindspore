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

"""test hccl isend and irecv with 2p"""

import os
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.communication.management import init, get_rank
from mindspore.ops.communication import isend, irecv, set_comm_ops_inplace
from mindspore import context, jit

np.random.seed(1)
context.set_context(jit_level='O0')
os.environ['HCCL_WHITELIST_DISABLE'] = str(1)
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
init()
this_rank = get_rank()
#msrun --worker_num=2 --local_worker_num=2 --master_port=1128 --log_dir=msrun_log --join=True --cluster_time_out=300  test_ops_all_reduce.py


def test_hccl_isend_irecv_func_dynstatic1():
    """
    Feature: test 'all_reduce' communication function.
    Description: test 'all_reduce' communication function.
    Expectation: expect correct result.
    """
    @jit(jit_level="O0")
    def test_send(in_tensor):
        """
        Feature: test 'all_reduce' communication function.
        Description: test 'all_reduce' communication function.
        Expectation: expect correct result.
        """
        h = isend(in_tensor, dst=1)
        return h


    @jit(jit_level="O0")
    def test_recv(in_tensor):
        """
        Feature: test 'all_reduce' communication function.
        Description: test 'all_reduce' communication function.
        Expectation: expect correct result.
        """
        h = irecv(in_tensor, src=0)
        return h

    set_comm_ops_inplace(True)
    context.set_context(save_graphs=3, save_graphs_path="ir_dump_path")
    x = np.ones([3, 4]).astype(np.float32)
    y = Tensor(np.zeros([3, 4]).astype(np.float32), mstype.float32)
    expect_output = np.ones([3, 4]).astype(np.float32)
    if this_rank == 0:
        h = test_send(Tensor(x, mstype.float32))
        h.wait()
    if this_rank == 1:
        h = test_recv(y)
        h.wait()
        assert np.allclose(y.asnumpy(), expect_output)


def test_hccl_isend_irecv_func_dynstatic2():
    """
    Feature: test 'all_reduce' communication function.
    Description: test 'all_reduce' communication function.
    Expectation: expect correct result.
    """
    @jit(jit_level="O0")
    def test_send1(in_tensor):
        """
        Feature: test 'all_reduce' communication function.
        Description: test 'all_reduce' communication function.
        Expectation: expect correct result.
        """
        h = isend(in_tensor, dst=1)
        return h

    @jit(jit_level="O0")
    def test_recv1(in_tensor):
        """
        Feature: test 'all_reduce' communication function.
        Description: test 'all_reduce' communication function.
        Expectation: expect correct result.
        """
        out, h = irecv(in_tensor, src=0)
        return out, h

    set_comm_ops_inplace(False)
    context.set_context(save_graphs=3, save_graphs_path="ir_dump_path")
    x = np.ones([3, 4]).astype(np.float32)
    y = Tensor(np.zeros([3, 4]).astype(np.float32), mstype.float32)
    expect_output = np.ones([3, 4]).astype(np.float32)
    if this_rank == 0:
        h = test_send1(Tensor(x, mstype.float32))
        h.wait()
    if this_rank == 1:
        out, h = test_recv1(y)
        h.wait()
        assert np.allclose(out.asnumpy(), expect_output)
