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
"""Test basic operation with one stage"""
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.common import dtype
from mindspore.common.api import jit
from tests.mark_utils import arg_mark

cfg = {
    "replace_nncell_by_construct": True,
    "print_after_all": False,
    "compile_by_trace": True,
    "print_bb": False,
    "MAX_INLINE_DEPTH": 10,
    "allowed_inline_modules": ["mindspore"],  # buildsubgraph
}


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_make_tuple():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            return (x, x+1, x+2)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a)
    assert isinstance(ret, tuple)
    assert len(ret) == 3
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])
    assert ret[2] == Tensor([3])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_make_list():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            return [x, x+1, x+2]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a)
    assert isinstance(ret, list)
    assert len(ret) == 3
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])
    assert ret[2] == Tensor([3])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_return_add_result_tuple():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return x + y

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = (1, 2, 3)
    b = (4, 5, 6)
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a, b)
    assert ret == (1, 2, 3, 4, 5, 6)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_return_add_result_list():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return x + y

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = [1, 2, 3]
    b = [4, 5, 6]
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a, b)
    assert ret == [1, 2, 3, 4, 5, 6]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_empty_tuple_input():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.a = Parameter(Tensor([1, 2, 3]))
            self.b = Parameter(Tensor([1, 1, 1]))

        @jit(mode="PIJit", jit_config=cfg)
        def construct(self, x):
            return self.a + self.b

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    ret = net(())
    assert np.all(ret.asnumpy() == np.array([2, 3, 4]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_empty_list_input():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.a = Parameter(Tensor([1, 2, 3]))
            self.b = Parameter(Tensor([1, 1, 1]))

        @jit(mode="PIJit", jit_config=cfg)
        def construct(self, x):
            return self.a + self.b

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    ret = net([])
    assert np.all(ret.asnumpy() == np.array([2, 3, 4]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_empty_dict_input():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.a = Parameter(Tensor([1, 2, 3]))
            self.b = Parameter(Tensor([1, 1, 1]))

        @jit(mode="PIJit", jit_config=cfg)
        def construct(self, x):
            return self.a + self.b

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    ret = net({})
    assert np.all(ret.asnumpy() == np.array([2, 3, 4]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tuple_slice():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = (x, x+1, x+2)
            return m[0:2:1]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a)
    assert isinstance(ret, tuple)
    assert len(ret) == 2
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_slice():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = [x, x+1, x+2]
            return m[0:2:1]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a)
    assert isinstance(ret, list)
    assert len(ret) == 2
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_slice_with_default_parameter():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = [x, x+1, x+2]
            return m[0:2]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a)
    assert isinstance(ret, list)
    assert len(ret) == 2
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_slice_with_default_parameter_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = [x, x+1, x+2]
            return m[::]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a)
    assert isinstance(ret, list)
    assert len(ret) == 3
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])
    assert ret[2] == Tensor([3])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_slice_with_default_parameter_3():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = [x, x+1, x+2]
            return m[:]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a)
    assert isinstance(ret, list)
    assert len(ret) == 3
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])
    assert ret[2] == Tensor([3])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_make_dict():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = {"x": x, "y": x+1}
            return m["x"]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a)
    assert ret == Tensor([1])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_make_dict_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = {}
            m["x"] = x
            return m["x"]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a)
    assert ret == Tensor([1])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_make_dict_3():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = {"x": x+1}
            return m["x"]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a)
    assert ret == Tensor([2])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tuple_input():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return x/y

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = (1.0, 2.0, 3.0)
    b = Tensor(np.ones([2, 3]).astype(np.float32))
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a, b)
    expect = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert np.allclose(ret.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_input():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return x/y

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = [1.0, 2.0, 3.0]
    b = Tensor(np.ones([2, 3]).astype(np.float32))
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(a, b)
    expect = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert np.allclose(ret.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_handle_constant():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            a, b = x
            return (a, b)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    m = (1, 2)
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(m)
    assert ret == (1, 2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_handle_constant_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            a, b = x
            return (a, b)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    m = [1, 2]
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(m)
    assert ret == (1, 2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_handle_mutable_kwargs_args():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, a, *args, b=1, **kwargs):
            return a + b + args[0] + kwargs["s"]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(1, 10, 100, s=1000)
    assert ret == 1012


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_handle_mutable_kwargs_args_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, a, *args, b=1, **kwargs):
            return a + b + args[0]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret = net(1, 10, 100, s=1000)
    assert ret == 12


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_use_free_variable():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            mod = 2
            return any(i % mod == 0 for i in x)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    jit(net.construct, mode="PIJit", jit_config={"loop_unrolling": True, "compile_by_trace": True})
    input1 = (1, 2, 3, 4)
    assert net(input1)
    input2 = (1, 1, 1, 1, 1)
    assert not net(input2)


@pytest.mark.skip(reason="When disable loop_unrolling, check guard failed.")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_use_free_variable_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            mod = 2
            return any(i % mod == 0 for i in x)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    jit(net.construct, mode="PIJit", jit_config={"compile_by_trace": True})
    input1 = (1, 2, 3, 4)
    assert net(input1)
    input2 = (1, 1, 1, 1, 1)
    assert not net(input2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_guard_for_getattr():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.a = 1

        def construct(self, x):
            return self.a + x

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret1 = net(1)
    net.a = 2
    ret2 = net(2)
    assert ret1 == 2
    assert ret2 == 4


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_guard_for_getattr_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.a = 1

        def construct(self, x):
            return self.a + x

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    jit(net.construct, mode="PIJit", jit_config=cfg)
    ret1 = net(1)
    net.a = 2
    ret2 = net(1)
    assert ret1 == 2
    assert ret2 == 3


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cycle_container_structure():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            return x

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    jit(net.construct, mode="PIJit", jit_config=cfg)
    a = [1, 2]
    a += [a]
    ret = net(a)
    assert ret == a


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cycle_container_structure_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            return x

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    jit(net.construct, mode="PIJit", jit_config=cfg)
    a = {"1": 1}
    a["2"] = a
    ret = net(a)
    assert ret == a


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cycle_container_structure_3():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            return x

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    jit(net.construct, mode="PIJit", jit_config=cfg)
    a = [1, 2, 3]
    b = [4, 5, 6]
    a[0] = b
    b[0] = a
    ret1 = net(a)
    assert ret1 == a
    ret2 = net(b)
    assert ret2 == b


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_guard_parameter():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.w = Parameter(Tensor(np.random.rand(2, 2), dtype.float32), name='w')

        @jit(mode="PIJit")
        def construct(self, x):
            return self.w * x

    context.set_context(mode=context.PYNATIVE_MODE)
    m = Tensor([[1, 1], [2, 2]], dtype.float32)
    net1 = Net()
    ret1 = net1(m)
    net2 = Net()
    ret2 = net2(m)
    assert np.allclose(ret1.asnumpy(), (net1.w * m).asnumpy())
    assert np.allclose(ret2.asnumpy(), (net2.w * m).asnumpy())
