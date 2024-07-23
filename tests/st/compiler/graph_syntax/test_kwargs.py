# Copyright 2023 Huawei Technologies Co., Ltd
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
""" test kwargs with side effect. """
import pytest
import numpy as np
from mindspore.ops import operations as P

import mindspore as ms
from mindspore import Tensor, Parameter, context, nn, jit, ops
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_kwargs_has_side_effect():
    """
    Feature: Support kwargs has side effect.
    Description: Support kwargs has side effect.
    Expectation: No exception.
    """

    def multi_forward(input_x, call_func=None):
        return call_func(input_x)

    class KwargsTestNet(nn.Cell):
        def __init__(self):
            super(KwargsTestNet, self).__init__()
            self.param = Parameter(Tensor([1.0], ms.float32), name="para1")
            self.assign = P.Assign()

        def my_assign_value(self, value):
            self.assign(self.param, value * 2)
            return self.param + 2

        def construct(self, x):
            return multi_forward(x, call_func=self.my_assign_value)

    net = KwargsTestNet()
    out = net(Tensor([10], ms.float32))
    print(out)
    assert out == 22


@pytest.mark.skip("the key in kwargs is any")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_kwargs_key_value_both_is_custom_class_attr():
    """
    Feature: Support the kwargs is any.
    Description: Graph syntax resolve support custom class input is kwargs.
    Expectation: No error.
    """

    class Config:
        def __init__(self, **kwargs):
            self.aaa = kwargs.pop("aaa", 2.0)
            self.input1 = "input1"

    class Model(ms.nn.Cell):
        def construct(self, input1, input2):
            return input1 * input2

    class Net(ms.nn.Cell):
        def __init__(self, net_config):
            super().__init__()
            self.config = net_config
            self.model = Model()

        def construct(self):
            arg_dict = {self.config.input1: self.config.aaa + 1, "input2": self.config.aaa}
            return self.model(**arg_dict)

    config = Config()
    net = Net(config)
    output = net()
    assert output == 6


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_useless_kwargs():
    """
    Feature: Support the kwargs is not used in function.
    Description: Graph syntax support kwargs.
    Expectation: No error.
    """
    x = Tensor([1, 2])

    @jit
    def func(*args, **conf):
        def ff(x, *args, **conf):
            return ops.mul(*x, *args)

        return ff(*args)

    res = func((x,), x, a=x)

    assert np.allclose(res.asnumpy(), np.array([1, 4]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_use_partial_kwargs():
    """
    Feature: Support Not all key-value parameters are fully utilized in the iinput function.
    Description: Graph syntax support kwargs.
    Expectation: No error.
    """
    x = Tensor([1, 2])

    @jit
    def func(*args, **conf):
        def ff(x, *args, **conf):
            res = ops.mul(*x, *args)
            res = res + conf.get('a', 0)
            return res

        return ff(*args)

    res = func((x,), x, a=x, b=x)

    assert np.allclose(res.asnumpy(), np.array([1, 4]))
