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

"""test hccl ReduceScatter and reduce_scatter_tensor with 8p"""

import os
import numpy as np
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.communication.management import init, get_rank
from mindspore.communication.comm_func import reduce_scatter_tensor
from mindspore import context
from mindspore.ops import ReduceOp

np.random.seed(1)
context.set_context(jit_level='O0')
os.environ['HCCL_WHITELIST_DISABLE'] = str(1)
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
init()


class ReduceScatterNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reduce_scatter = P.ReduceScatter()

    def construct(self, x):
        return self.reduce_scatter(x)


class ReduceScatterFuncNet(nn.Cell):
    def __init__(self, op=ReduceOp.SUM):
        super().__init__()
        self.op = op

    def construct(self, x):
        return reduce_scatter_tensor(x)


def test_hccl_reduce_scatter_tensor_8p():
    """
    Feature: test 'ReduceScatter' communication operator.
    Description: test 'ReduceScatter' communication operator.
    Expectation: expect correct result.
    """
    input_tensor = Tensor(np.ones([8, 8]).astype(np.float32))
    net = ReduceScatterNet()
    output = net(input_tensor)
    expect_output = (np.ones([1, 8]) * 8).astype(np.float32)
    print("all_gather_into_tensor func output is", output)
    assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_reduce_scatter_tensor_func_net_8p():
    """
    Feature: test 'reduce_scatter_tensor' communication function in cell.
    Description: test 'reduce_scatter_tensor' communication function in cell.
    Expectation: expect correct result.
    """
    input_tensor = Tensor(np.ones([8, 8]).astype(np.float32))
    net = ReduceScatterFuncNet()
    output = net(input_tensor)
    expect_output = (np.ones([1, 8]) * 8).astype(np.float32)
    print("all_gather_into_tensor func output is", output)
    assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_reduce_scatter_func_8p():
    """
    Feature: test 'reduce_scatter_tensor' communication function.
    Description: test 'reduce_scatter_tensor' communication function.
    Expectation: expect correct result.
    """
    input_tensor = Tensor(np.ones([8, 8]).astype(np.float32))
    output = reduce_scatter_tensor(input_tensor)
    expect_output = (np.ones([1, 8]) * 8).astype(np.float32)
    print("all_gather_into_tensor func output is", output)
    assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_reduce_scatter_diff_reduce_type():
    """
    Feature: test reduce_scatter with different reducing type.
    Description: test reduce_scatter with different reducing type.
    Expectation: expect type checking exception.
    """
    input_tensor = Tensor(np.ones([8, 8]).astype(np.float32))
    try:
        if get_rank() in [0, 1, 2, 3]:
            input_tensor = Tensor(np.ones([8, 8]).astype(np.float32))
            _ = reduce_scatter_tensor(input_tensor)
        else:
            input_tensor = Tensor(np.ones([8, 8]).astype(np.float16))
            _ = reduce_scatter_tensor(input_tensor)
    except RuntimeError as e:
        error_msg = str(e)
        print(error_msg)
        assert "inconsistent" in error_msg
