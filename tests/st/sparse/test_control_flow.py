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
"""smoke tests for Sparse control flow cases"""

import pytest
import numpy as np

from mindspore import Tensor, CSRTensor, Parameter, nn, context
from mindspore.common import dtype as mstype
import mindspore.ops.operations as P

from .sparse_utils import compare_csr, get_csr_tensor, csr_add, get_csr_components, get_csr_from_scalar, \
    forward_grad_net
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_while_tensor_as_condition_forward_and_backward():
    """
    Feature: Test CSRTensor in while.
    Description: Test CSRTensor computation in while loop.
    Expectation: Success.
    """
    class Net(nn.Cell):

        def construct(self, x, y):
            out = y
            while x < 1:
                out = csr_add(out, out.values)
                x += 1
            return out

    x = Tensor(-2, dtype=mstype.int32)
    y = get_csr_tensor()
    net = Net()

    csr1, grad_py = forward_grad_net(net, x, y, mode=context.PYNATIVE_MODE)
    csr2, grad_graph = forward_grad_net(net, x, y, mode=context.GRAPH_MODE)

    # Compare results
    compare_csr(csr1, csr2)
    assert (csr1.values.asnumpy() == np.array([8, 16], dtype=np.float32)).all()
    assert len(grad_py) == 2
    assert len(grad_graph) == 2
    assert isinstance(grad_graph[1], CSRTensor)


@pytest.mark.skip(reason="Sparse tensor can not run renormalize")
@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_control_flow_while_if_continue_not_relevant_gt():
    """
    Feature: Test CSRTensor in while.
    Description: Test CSRTensor computation in while loop.
    Expectation: Success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.addn = P.AddN()

        def construct(self, x):
            s = x
            t = csr_add(x, 1)
            tensor_list = [x.values, x.values]
            while len(tensor_list) < 4:
                tensor_list.append(x.values)
                a = self.addn(tensor_list)
                x = csr_add(x, 1)
                if t.values in tensor_list:
                    continue
                s = csr_add(s, a)
            return s

    num = Tensor(-2, dtype=mstype.float32)
    x = get_csr_from_scalar(num)
    net = Net()

    csr1, grad_py = forward_grad_net(net, x, mode=context.PYNATIVE_MODE)
    csr2, grad_graph = forward_grad_net(net, x, mode=context.GRAPH_MODE)

    # Compare results
    compare_csr(csr1, csr2)
    assert (csr1.values.asnumpy() == np.array([-8], dtype=np.float32)).all()
    assert len(grad_py) == 1
    assert len(grad_graph) == 1
    assert isinstance(grad_graph[0], CSRTensor)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_control_flow_for_while_return_in_while_no():
    """
    Feature: Test CSRTensor in while.
    Description: Test CSRTensor computation in while loop.
    Expectation: Success.
    """
    class Net(nn.Cell):

        def construct(self, x, y):
            out = y
            for _ in range(3):
                out = csr_add(out, y.values)
            while x < 5:
                out = csr_add(out, out.values)
                if x > 1:
                    return out
                x += 1
            return out

    x = Tensor(-2, dtype=mstype.int32)
    y = get_csr_tensor()
    net = Net()

    csr1, grad_py = forward_grad_net(net, x, y, mode=context.PYNATIVE_MODE)
    csr2, grad_graph = forward_grad_net(net, x, y, mode=context.GRAPH_MODE)

    # Compare results
    compare_csr(csr1, csr2)
    assert (csr1.values.asnumpy() == np.array([128, 256], dtype=np.float32)).all()
    assert len(grad_py) == 2
    assert len(grad_graph) == 2
    assert isinstance(grad_graph[1], CSRTensor)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_control_flow_for_enumerate_if_continue():
    """
    Feature: Test CSRTensor in while.
    Description: Test CSRTensor computation in while loop.
    Expectation: Success.
    """
    class Net(nn.Cell):
        def __init__(self, t1, t2):
            super().__init__()
            self.p1 = Parameter(Tensor(t1, mstype.float32), name="a")
            self.p2 = Parameter(Tensor(t2, mstype.float32), name="b")
            self.assignadd = P.AssignAdd()

        def construct(self, x):
            plist = [self.p1, self.p2]
            out = x
            for i, t in enumerate(plist):
                if t > 2:
                    continue
                self.assignadd(t, 1)
                out = csr_add(out, i * t)
            return out

    t1 = 1
    t2 = 2
    x = get_csr_tensor()

    csr1, grad_py = forward_grad_net(Net(t1, t2), x, mode=context.PYNATIVE_MODE)
    csr2, grad_graph = forward_grad_net(Net(t1, t2), x, mode=context.GRAPH_MODE)

    # Compare results
    compare_csr(csr1, csr2)
    assert (csr1.values.asnumpy() == np.array([4, 5], dtype=np.float32)).all()
    assert len(grad_py) == 1
    assert len(grad_graph) == 1
    assert isinstance(grad_graph[0], CSRTensor)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_multi_csr_in_if_else():
    """
    Feature: Test multiple CSRTensors in if-else.
    Description: Test CSRTensor computation in control flow.
    Expectation: Success.
    """
    class Net(nn.Cell):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape

        def construct(self, indptr, indices, values, a, b):
            x = CSRTensor(indptr, indices, values, self.shape)
            if a > b:
                x1 = x.abs()
                x2 = x.astype(mstype.float16)
                x3 = x.to_tuple()
            else:
                x1 = x.abs()
                x2 = x.astype(mstype.float16)
                x3 = x.to_tuple()
            return x1, x2, x3

    a = Tensor(1, mstype.float32)
    b = Tensor(0, mstype.float32)
    indptr, indices, values, shape = get_csr_components()
    net = Net(shape)

    forward_grad_net(net, indptr, indices, values, a, b, mode=context.PYNATIVE_MODE)
    forward_grad_net(net, indptr, indices, values, a, b, mode=context.GRAPH_MODE)
