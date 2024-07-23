# Copyright 2021-2023 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np

from mindspore import Tensor, Parameter, context, jit
from mindspore.ops import operations as P
from mindspore.nn import Cell
import mindspore as ms
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_zip_operation_args_size():
    """
    Feature: Check the size of inputs of ZipOperation.
    Description: The inputs of ZipOperation must not be empty.
    Expectation: The size of inputs of ZipOperation must be greater than 0.
    """
    class AssignInZipLoop(Cell):
        def __init__(self):
            super().__init__()
            self.conv1 = ms.nn.Conv2d(3, 2, 1, weight_init="zero")
            self.conv2 = ms.nn.Conv2d(3, 2, 1, weight_init="zero")
            self.params1 = self.conv1.trainable_params()
            self.params2 = self.conv2.trainable_params()

        def construct(self, x):
            for p1, p2 in zip():
                P.Assign()(p2, p1 + x)

            out = 0
            for p1, p2 in zip(self.params1, self.params2):
                out = p1 + p2

            return out

    x = Tensor.from_numpy(np.ones([1], np.float32))
    net = AssignInZipLoop()
    with pytest.raises(Exception, match="The zip operator must have at least 1 argument"):
        out = net(x)
        assert np.all(out.asnumpy() == 1)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_zip_operation_args_type():
    """
    Feature: Check the type of inputs of ZipOperation.
    Description: Check whether all inputs in zip is sequeue.
    Expectation: All inputs in zip must be sequeue.
    """
    class AssignInZipLoop(Cell):
        def __init__(self):
            super().__init__()
            self.conv1 = ms.nn.Conv2d(3, 2, 1, weight_init="zero")
            self.conv2 = ms.nn.Conv2d(3, 2, 1, weight_init="zero")
            self.params1 = self.conv1.trainable_params()
            self.params2 = self.conv2.trainable_params()
            self.param = Parameter(Tensor(5, ms.float32), name="param")

        def construct(self, x):
            for p1, p2 in zip(self.params1, self.params2, self.param):
                P.Assign()(p2, p1 + x)

            out = 0
            for p1, p2 in zip(self.params1, self.params2):
                out = p1 + p2

            return out

    x = Tensor.from_numpy(np.ones([1], np.float32))
    net = AssignInZipLoop()
    with pytest.raises(TypeError, match="Cannot iterate over a scalar tensor."):
        out = net(x)
        assert np.all(out.asnumpy() == 1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_zip_operation_with_tensor_input():
    """
    Feature: Check the type of inputs of ZipOperation.
    Description: Check whether all inputs in zip is sequeue.
    Expectation: All inputs in zip must be sequeue.
    """
    @jit
    def foo(x):
        return tuple(zip(x, x+1))

    ret = foo(Tensor([1, 2, 3]))
    assert ret == ((Tensor([1]), Tensor([2])), (Tensor([2]), Tensor([3])), (Tensor([3]), Tensor([4])))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_zip_operation_with_tensor_input_2():
    """
    Feature: Check the type of inputs of ZipOperation.
    Description: Check whether all inputs in zip is sequeue.
    Expectation: All inputs in zip must be sequeue.
    """
    @jit
    def foo(x):
        return tuple(zip(x, (1, 2, 3)))

    ret = foo(Tensor([1, 2, 3]))
    assert ret == ((Tensor([1]), 1), (Tensor([2]), 2), (Tensor([3]), 3))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_zip_with_numpy_and_tensor():
    """
    Feature: JIT Fallback
    Description: Test zip in graph mode with numpy and tensor input.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = np.array([1, 2])
        y = Tensor([10, 20])
        ret = zip(x, y)
        return tuple(ret)

    out = foo()
    assert out == ((1, Tensor([10])), (2, Tensor([20])))
