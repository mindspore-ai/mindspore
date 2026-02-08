# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
import os
import numpy as np
import mindspore as ms
from mindspore import jit
from mindspore import Tensor
from mindspore import mutable

reserved_env = None


def setup_module():
    global reserved_env
    reserved_env = os.getenv('MS_DEV_PRECOMPILE_ONLY')
    os.environ['MS_DEV_PRECOMPILE_ONLY'] = '1'


def teardown_module():
    if reserved_env is None:
        os.unsetenv('MS_DEV_PRECOMPILE_ONLY')
    else:
        os.environ['MS_DEV_PRECOMPILE_ONLY'] = reserved_env


def test_initial_scalar_body_tensor2():
    """
    Feature: While specialize.
    Description: Test scalar arg when first entry of while and set to tensor in body.
    Expectation: No exception in infer process.
    """

    class Net(ms.nn.Cell):

        def construct(self, x, a, b):
            y = 1
            while a < b:
                y = Tensor(2, ms.float32)
                a += 1
            return x + y

    ms.context.set_context(mode=ms.context.GRAPH_MODE)
    input_np_x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_me_x = Tensor(input_np_x)
    input_me_a = Tensor(2, ms.float32)
    input_me_b = Tensor(4, ms.float32)
    net = Net()
    net(input_me_x, input_me_a, input_me_b)


def test_initial_scalar_body_tensor3():
    """
    Feature: While specialize.
    Description: Test scalar arg when first entry of while and set to tensor in body.
    Expectation: No exception in infer process.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.ini_flg = Tensor(False, ms.bool_)
            self.true_flg = Tensor([True], ms.bool_)
            self.false_flg = Tensor([False], ms.bool_)

        def construct(self, x, y):
            finish_flg = self.ini_flg
            while finish_flg:
                x = x + 1
                y = y - 1
                if x > y:
                    finish_flg = self.false_flg
                else:
                    finish_flg = self.true_flg
            return x, y

    ms.context.set_context(mode=ms.context.GRAPH_MODE)
    x_arg = Tensor([0], ms.int32)
    y_arg = Tensor([10], ms.int32)
    net = Net()

    net(x_arg, y_arg)


def test_initial_tensor_body_ref():
    """
    Feature: While specialize.
    Description: Test constant tensor arg when first entry of while and set to RefTensor in body.
    Expectation: No exception in infer process.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight = ms.Parameter(Tensor([1]))

        def construct(self, a, b):
            y_param = Tensor([1])
            while a < b:
                y_param = self.weight
                a += 1
            out = y_param + b
            return out

    test_net = Net()
    ms.context.set_context(mode=ms.context.GRAPH_MODE)
    input_a = Tensor([2])
    input_b = Tensor([6])
    test_net(input_a, input_b)

    @jit
    def test_grad_net(a, b):
        return ms.ops.grad(test_net)(a, b)

    input_a = Tensor([2])
    input_b = Tensor([6])
    test_grad_net(input_a, input_b)


def test_initial_ref_body_tensor():
    """
    Feature: While specialize.
    Description: Test constant RefTensor arg when first entry of while and set to Tensor in body.
    Expectation: No exception in infer process.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight = ms.Parameter(Tensor([1]))

        def construct(self, a, b):
            y_param = self.weight
            while a < b:
                y_param = Tensor([1])
                a += 1
            out = y_param + b
            return out

    test_net = Net()
    ms.context.set_context(mode=ms.context.GRAPH_MODE)
    input_a = Tensor([2])
    input_b = Tensor([6])
    test_net(input_a, input_b)

    @jit
    def test_grad_net(a, b):
        return ms.ops.grad(test_net)(a, b)

    input_a = Tensor([2])
    input_b = Tensor([6])
    test_grad_net(input_a, input_b)


def test_initial_emtpy_list_body_list():
    """
    Feature: While specialize.
    Description: Test constant mutable([]) arg when first entry of while and set to mutable with not empty list in body.
    Expectation: No exception in infer process.
    """

    @ms.jit
    def test_net():
        x = mutable([1, 2, 3, 4], True)
        y = mutable([], True)
        for i in x:
            y.append(i)
        return y

    ms.context.set_context(mode=ms.context.GRAPH_MODE)
    test_net()
