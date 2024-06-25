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

import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, jit, context, nn, Parameter
from mindspore import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_getattr_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test getattr in fallback runtime
    Expectation: No exception.
    """
    @jit
    def foo():
        x = Tensor(np.array([1, 2, 3, 4])).asnumpy()
        len_func1 = getattr(x, "__len__", Tensor([-1]))
        attr = "__len__"
        len_func2 = getattr(x, attr)
        return len_func1(), len_func2()

    out = foo()
    assert out[0] == out[1] == 4


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_getattr_asnumpy_custom_class():
    """
    Feature: getattr for custom class.
    Description: Support getattr for custom class.
    Expectation: No exception.
    """
    class GetattrClass():
        def __init__(self):
            self.attr1 = Tensor(np.array([1, 2, 3, 4])).asnumpy()
            self.attr2 = 1

    class GetattrClassNet(nn.Cell):
        def __init__(self):
            super(GetattrClassNet, self).__init__()
            self.cls = GetattrClass()

        def construct(self):
            attr = "__len__"
            len_func1 = getattr(self.cls.attr1, attr)
            len_func2 = getattr(self.cls.attr1, "__len__")
            return len_func1(), len_func2()

    net = GetattrClassNet()
    out = net()
    assert out[0] == out[1] == 4


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_getattr_numpy_array_2():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support numpy array input.
    Expectation: TypeError
    """

    @jit
    def foo():
        x = 1
        return getattr(x, "shape", np.array([0, 1, 2, 3, 4]))

    out = foo()
    assert (out == np.array([0, 1, 2, 3, 4])).all()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_get_attr_form_param():
    """
    Feature: Graph mode do not support getattr on Parameter.
    Description: Graph mode do not support getattr on Parameter.
    Expectation: success.
    """

    param_attr_fg = C.MultitypeFuncGraph("param_attr_fg")

    @param_attr_fg.register("Tensor", "Tensor")
    def param_attr_fg_for_tensor(x, param):  # pylint: disable=unused-variable
        if param.requires_grad:
            return P.Square()(x * 2)
        return P.Square()(x)

    class HyperMapNet(nn.Cell):
        def __init__(self, fg):
            super(HyperMapNet, self).__init__()
            self.hyper_map = C.HyperMap()
            self.parameter = Parameter(Tensor([1], ms.float32), name="name_a")
            self.fg = fg

        def construct(self, x):
            return self.hyper_map(self.fg, x, self.parameter)

    with pytest.raises(RuntimeError) as ex:
        x = Tensor(np.array([1]), mstype.float32)
        net = HyperMapNet(param_attr_fg)
        output = net(x)
        print("output:", output)
    assert "Failed to compile in GRAPH_MODE" in str(ex.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_get_attr_form_param_2():
    """
    Feature: Graph mode do not support getattr on Parameter.
    Description: Graph mode do not support getattr on Parameter.
    Expectation: success.
    """

    param_attr_fg = C.MultitypeFuncGraph("param_attr_fg")

    @param_attr_fg.register("Tensor", "Bool")
    def param_attr_fg_for_tensor_2(x, requires_grad):  # pylint: disable=unused-variable
        if requires_grad:
            return P.Square()(x * 2)
        return P.Square()(x)

    class HyperMapNet(nn.Cell):
        def __init__(self, fg):
            super(HyperMapNet, self).__init__()
            self.hyper_map = C.HyperMap()
            self.parameter = Parameter(Tensor([1], ms.float32), name="name_a")
            self.fg = fg
            self.requires_grad = self.parameter.requires_grad

        def construct(self, x):
            return self.hyper_map(self.fg, x, self.requires_grad)

    x = Tensor(np.array([1]), mstype.float32)
    net = HyperMapNet(param_attr_fg)
    output = net(x)
    assert output == 4


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_attr_in_control_flow_no_check():
    """
    Feature: test  the validity of the attribute in graph mode.
    Description: Do not check the validity of the attribute in the variable scenario.
    Expectation: success
    """
    class GetattrClass():
        def __init__(self):
            self.attr1 = 99
            self.attr2 = 1

        def method1(self, x):
            return x + self.attr2

    class SizeNet(ms.nn.Cell):
        def __init__(self):
            super(SizeNet, self).__init__()
            self.cls = GetattrClass()

        def construct(self, x):
            if isinstance(self.cls.method1(x), dict):
                return x.keys()
            return x

    net = SizeNet()
    x = Tensor([4])
    out = net(x)
    assert out == 4
