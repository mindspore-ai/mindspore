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
import os
import pytest
import types
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import ops
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.common import dtype
from mindspore.common import mutable
from mindspore.graph.api import jit
from tests.mark_utils import arg_mark
from mindspore._c_expression import get_code_extra
from tests.st.pi_jit.share.utils import pi_jit_with_config, assert_equal, match_array
from mindspore._c_expression import TensorPy as CppTensor
from tests.st.pi_jit.share.utils import assert_graph_compile_status

cfg = {
    "print_after_all": False,
    "print_bb": False,
}

def assert_executed_by_graph_mode(func):
    jcr = get_code_extra(getattr(func, "__wrapped__", func))
    assert jcr is not None
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0, f'break_count expect: 0, actual: {jcr["break_count_"]}'
    if 'phase_' in jcr['code']:
        assert len(jcr['code']['phase_']) > 0
    else:
        checked = False
        for item in jcr['code']['compiled_code_'].co_consts:
            if isinstance(item, types.CodeType):
                j = get_code_extra(item)
                assert len(j['code']['phase_']) > 0
                checked = True
                break
        assert checked

@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(ret, tuple)
    assert len(ret) == 3
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])
    assert ret[2] == Tensor([3])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, a, b)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, a, b)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
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

        def construct(self, x):
            return self.a + self.b

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, ())
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
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

        def construct(self, x):
            return self.a + self.b

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, [])
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
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

        def construct(self, x):
            return self.a + self.b

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, {})
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert np.all(ret.asnumpy() == np.array([2, 3, 4]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(ret, tuple)
    assert len(ret) == 2
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(ret, list)
    assert len(ret) == 2
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(ret, list)
    assert len(ret) == 3
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])
    assert ret[2] == Tensor([3])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(ret, list)
    assert len(ret) == 3
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])
    assert ret[2] == Tensor([3])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='essential')
def test_variable_number_in_container():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            return x.shape

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor(CppTensor(shape=[-1, 12], dtype=mindspore.float32))
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, a)
    assert ret == (-1, 12)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='essential')
def test_variable_number_in_container_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            a = x.shape[0]
            b = {"1": a}
            return b["1"]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor(CppTensor(shape=[-1, 12], dtype=mindspore.float32))
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, a)
    assert ret == -1
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert ret == Tensor([1])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert ret == Tensor([1])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert ret == Tensor([2])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, a, b)
    expect = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert np.allclose(ret.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, a, b)
    expect = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, m)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, m)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, 1, 10, 100, s=1000)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
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
    ret = pi_jit_with_config(Net.construct, jit_config=cfg)(net, 1, 10, 100, s=1000)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert ret == 12


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_use_free_variable():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @pi_jit_with_config(jit_config={"loop_unrolling": True})
        def construct(self, x):
            mod = 2
            return any(i % mod == 0 for i in x)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    input1 = (1, 2, 3, 4)
    assert net(input1)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    input2 = (1, 1, 1, 1, 1)
    assert not net(input2)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_use_free_variable_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x):
            mod = 2
            return any(i % mod == 0 for i in x)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
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

        @pi_jit_with_config(jit_config=cfg)
        def construct(self, x):
            return self.a + x

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    ret1 = net(1)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    net.a = 2
    ret2 = net(2)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0
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

        @pi_jit_with_config(jit_config=cfg)
        def construct(self, x):
            return self.a + x

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    ret1 = net(1)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    net.a = 2
    ret2 = net(1)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    assert ret1 == 2
    assert ret2 == 3


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_env_get():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    os.environ["abc"] = "1"

    @pi_jit_with_config(jit_config=cfg)
    def foo(x):
        ret = os.environ.get("abc")
        return ret, x + 1

    context.set_context(mode=context.PYNATIVE_MODE)
    ret = foo(Tensor([1, 2, 3]))
    assert isinstance(ret, tuple)
    assert ret[0] == "1"
    jcr = get_code_extra(getattr(foo, "__wrapped__", foo))
    assert jcr is not None
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert len(jcr['code']['phase_']) > 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cycle_container_structure():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @pi_jit_with_config(jit_config=cfg)
        def construct(self, x):
            return x

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
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
        @pi_jit_with_config(jit_config=cfg)
        def construct(self, x):
            return x

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
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
        @pi_jit_with_config(jit_config=cfg)
        def construct(self, x):
            return x

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = [1, 2, 3]
    b = [4, 5, 6]
    a[0] = b
    b[0] = a
    ret1 = net(a)
    assert ret1 == a
    ret2 = net(b)
    assert ret2 == b


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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

        @jit(capture_mode="bytecode")
        def construct(self, x):
            return self.w * x

    context.set_context(mode=context.PYNATIVE_MODE)
    m = Tensor([[1, 1], [2, 2]], dtype.float32)
    net1 = Net()
    ret1 = net1(m)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    net2 = Net()
    ret2 = net2(m)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    assert np.allclose(ret1.asnumpy(), (net1.w * m).asnumpy())
    assert np.allclose(ret2.asnumpy(), (net2.w * m).asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dict_items_call_in_control_flow():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x, y, z):
            m = {"1": x + z, "2": y - z}
            ret = 0
            for _, v in m.items():
                ret = ret + v
            return ret

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    c = Tensor([2, 2, 2])
    net = Net()
    ret = net(a, b, c)
    assert np.all(ret.asnumpy() == np.array([2, 3, 4]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_method_by_ast():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x):
            return x.view((2, 2))

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3, 4])
    net = Net()
    net(a)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_method_by_ast_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x):
            return x.var()

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3, 4])
    net = Net()
    net(a)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_method_by_ast_3():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x):
            return x.get("1")

    context.set_context(mode=context.PYNATIVE_MODE)
    a = {"1": Tensor([1, 2, 3, 4])}
    net = Net()
    net(a)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_method_by_ast_4():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x):
            return x.contiguous()

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3, 4])
    net = Net()
    net(a)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_function_parse_by_ast():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x):
            return ops.split(x, [3, ])

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor(np.ones((3, 128, 352, 224)))
    net = Net()
    net(a)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_function_parse_by_ast_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x):
            return ops.full_like(x, 1, dtype=mindspore.float32)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([[1, 2], [3, 4]])
    net = Net()
    net(a)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_function_parse_by_ast_3():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x, axis):
            return x.permute(*axis)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], mindspore.float32)
    axis = (0, 2, 1)
    net = Net()
    net(a, axis)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_constant_flod_for_variable():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            if all(x > y):
                out = x + y
            else:
                out = x * y
            return out

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([3, 4, 5])
    b = Tensor([1, 2, 3])
    net = Net()
    ret = net(a, b)
    assert np.all(ret.asnumpy() == np.array([4, 6, 8]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_attr_as_inputs():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.a = 1

        @jit(capture_mode="bytecode")
        def construct(self, x):
            self.a = self.a + 1
            return self.a + x

    context.set_context(mode=context.PYNATIVE_MODE)
    m = Tensor([1, 2, 3])
    net = Net()
    ret = net(m)
    assert np.all(ret.asnumpy() == np.array([3, 4, 5]))
    assert_graph_compile_status(Net.construct.__wrapped__, 0)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_attr_as_inputs_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.a = 1
            self.b = Parameter(Tensor([1, 1, 1]), name='b')

        @jit(capture_mode="bytecode")
        def construct(self, x):
            b = self.b
            self.a = self.a + 1
            return b + self.a + x

    context.set_context(mode=context.PYNATIVE_MODE)
    m = Tensor([1, 2, 3])
    net = Net()
    ret = net(m)
    assert np.all(ret.asnumpy() == np.array([4, 5, 6]))
    assert_graph_compile_status(Net.construct.__wrapped__, 0)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_attr_as_inputs_3():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.a = 1
            self.b = Parameter(Tensor([1, 1, 1]), name='b')

        @jit(capture_mode="bytecode")
        def construct(self, x):
            b = self.b
            self.a = self.a + 1
            return b + self.a + x

    m = Parameter(Tensor([1, 2, 3]), name='m')
    net = Net()
    ret = net(m)
    assert np.all(ret.asnumpy() == np.array([4, 5, 6]))
    assert_graph_compile_status(Net.construct.__wrapped__, 0)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_attr_as_inputs_4():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x):
            return self.a + x

    net = Net()
    for i in range(10):
        net.a = i
        x = Tensor(np.random.rand(4,4))
        y = net(x)
        assert np.all((x + i == y).asnumpy())

    assert_graph_compile_status(Net.construct.__wrapped__, 0, 8, 3)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_attr_as_inputs_5():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = 1

        def construct(self, x: Tensor) -> Tensor:
            self.x = self.x + 2
            return self.x + x

    pynative_model = Model()
    pijit_model = Model()
    pijit_model.construct = jit(pijit_model.construct, capture_mode='bytecode')

    pynative_outputs = []
    pijit_outputs = []
    for i in range(10):
        x = Tensor([i])
        y = pynative_model(x)
        pynative_outputs.append(y)
    for i in range(10):
        x = Tensor([i])
        y = pijit_model(x)
        pijit_outputs.append(y)

    assert_equal(pynative_model.x, pijit_model.x)
    assert_equal(pynative_outputs, pijit_outputs)
    assert_graph_compile_status(pijit_model.construct, 0, 8, 3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_attr_as_inputs_6():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x):
            return self.x + x

    net = Net()
    cond = [Tensor([99]), Tensor([61]), Tensor([32])]
    for i in range(10):
        net.x = cond[i % 3]
        x = Tensor(np.random.rand(4,4))
        y = net(x)
        assert np.all((x + net.x == y).asnumpy())

    assert_graph_compile_status(Net.construct.__wrapped__, 0, 9, 2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_attr_as_inputs_7():
    """
    Feature: Dynamic attribute support.
    Description: Test dynamic type.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x):
            return self.x + x

    net = Net()
    cond = [Tensor([-1]), 1, 2.2, np.float16(3.3)]
    for i in cond:
        net.x = i
        x = Tensor(np.random.rand(1))
        y = net(x)
        assert (net.x + x) == y

    assert_graph_compile_status(Net.construct, 0, 1, 4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_attr_as_inputs_8():
    """
    Feature: Dynamic attribute support.
    Description: Test dynamic shape.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @pi_jit_with_config(jit_config={"_symbolic": 1})
        def construct(self, x):
            return self.x + x

    net = Net()
    cond = []
    for i in range(1, 16):
        cond.append(Tensor(np.resize(np.array([i], np.float32).repeat(i), (i, 1))))
    for i in cond:
        net.x = i
        x = Tensor(np.random.rand(4))
        y = net(x)
        assert ((net.x + x) == y).all()

    assert_graph_compile_status(Net.construct, 0, 8, 8)

    # validate dynamic shape check
    i = Tensor(np.random.rand(4,4), dtype=mindspore.float32)
    net.x = i
    y = net(x)
    assert ((net.x + x) == y).all()
    assert_graph_compile_status(Net.construct, 0, 1, 9)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_symbolic_input_1():
    """
    Feature: Dynamic scalar mutable.
    Description: Test scalar input mutable.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            return x + y

    net = Net()
    for x in range(10):
        y = Tensor(np.random.rand(1))
        z = net(x, y)
        assert x + y == z

    assert_graph_compile_status(Net.construct, 0, 8, 3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_symbolic_input_2():
    """
    Feature: Dynamic shape input.
    Description: Test dynamic shape tensor as input.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @pi_jit_with_config(jit_config={"_symbolic": 1})
        def construct(self, x):
            return x + x

    net = Net()
    cond = []
    for i in range(1, 16):
        cond.append(Tensor(np.resize(np.array([i], np.float32).repeat(i), (i, 1))))
    for x in cond:
        y = net(x)
        assert ((x + x) == y).all()

    assert_graph_compile_status(Net.construct, 0, 9, 7)

    # validate dynamic shape check
    x = Tensor(np.random.rand(4,4), dtype=mindspore.float32)
    y = net(x)
    assert ((x + x) == y).all()
    assert_graph_compile_status(Net.construct, 0, 1, 8)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_attr_as_inputs_config():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @pi_jit_with_config(jit_config={"_symbolic": 2})
        def construct(self, x):
            return self.a + x

    net = Net()
    for i in range(100, 110):
        net.a = i
        x = Tensor(np.random.rand(4,4))
        y = net(x)
        assert np.all((x + i == y).asnumpy())

    assert_graph_compile_status(Net.construct.__wrapped__, 0, 6, 5)


@pytest.mark.skip
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_global_as_inputs_1():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    global magic

    class Net(nn.Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x):
            return magic + x

    net = Net()
    x = Tensor(np.random.rand(4,4))
    for _ in range(10):
        magic = np.random.randint(0, 10e7)
        y = net(x)
        assert np.all((magic + x == y).asnumpy())

    assert_graph_compile_status(Net.construct.__wrapped__, 0, 8, 3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_unpack_sequence_with_variable():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    def inner(x, y):
        return x + y, y

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        a, b = inner(x, y)
        return a, b

    input_x = Tensor([1, 2, 3])
    input_y = mutable(3)
    ret = foo(input_x, input_y)
    assert isinstance(ret, tuple)
    assert len(ret) == 2
    assert np.all(ret[0].asnumpy() == np.array([4, 5, 6]))
    assert ret[1] == 3
    assert_executed_by_graph_mode(foo)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_unpack_sequence_with_variable_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    def inner(x, y):
        return x + y, y

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        a, b = inner(x, y)
        return a, b

    input_x = mutable(1)
    input_y = mutable(3)
    ret = foo(input_x, input_y)
    assert isinstance(ret, tuple)
    assert len(ret) == 2
    assert ret[0] == 4
    assert ret[1] == 3
    assert_executed_by_graph_mode(foo)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_unpack_sequence_with_variable_3():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    def inner(x, y):
        return {"1": x + y, "2": y}

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        a, b = inner(x, y)
        return a, b, x + 1

    input_x = mutable(1)
    input_y = mutable(3)
    ret = foo(input_x, input_y)
    assert isinstance(ret, tuple)
    assert len(ret) == 3
    assert ret[0] == "1"
    assert ret[1] == "2"
    assert_executed_by_graph_mode(foo)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_empty_container_input():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y, z):
        return len(x), y + z

    a = [[], []]
    b = Tensor([1, 2, 3])
    c = Tensor([1, 1, 1])
    ret = foo(a, b, c)
    assert isinstance(ret, tuple)
    assert len(ret) == 2
    assert ret[0] == 2
    assert np.all(ret[1].asnumpy() == np.array([2, 3, 4]))
    assert_executed_by_graph_mode(foo)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_empty_container_input_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y, z):
        return len(x), y + z

    a = [([], []), []]
    b = Tensor([1, 2, 3])
    c = Tensor([1, 1, 1])
    ret = foo(a, b, c)
    assert isinstance(ret, tuple)
    assert len(ret) == 2
    assert ret[0] == 2
    assert np.all(ret[1].asnumpy() == np.array([2, 3, 4]))
    assert_executed_by_graph_mode(foo)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_empty_container_input_3():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y, z):
        return len(x), y + z

    a = {"1": [], "2": ()}
    b = Tensor([1, 2, 3])
    c = Tensor([1, 1, 1])
    ret = foo(a, b, c)
    assert isinstance(ret, tuple)
    assert len(ret) == 2
    assert ret[0] == 2
    assert np.all(ret[1].asnumpy() == np.array([2, 3, 4]))
    assert_executed_by_graph_mode(foo)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_subgraph_with_primitive_output():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """

    add_func = ops.Add()
    add_func.ref = add_func

    def cal_func():
        return add_func.ref

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x):
        return cal_func()(x, x)

    a = Tensor([1, 2, 3])
    ret = foo(a)
    assert np.all(ret.asnumpy() == np.array([2, 4, 6]))
    assert_executed_by_graph_mode(foo)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_subgraph_with_primitive_output_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """

    add_func = ops.Add()

    def cal_func():
        return add_func

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x):
        return cal_func()(x, x)

    a = Tensor([1, 2, 3])
    ret = foo(a)
    assert np.all(ret.asnumpy() == np.array([2, 4, 6]))
    assert_executed_by_graph_mode(foo)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_function_decorated_with_PSJIT_run_ast():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """

    @jit
    def cal_func(x, y):
        if x > 1:
            return y + 1
        return y + 2

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        return cal_func(x, y)

    a = Tensor([2])
    b = Tensor([1, 2, 3])
    ret = foo(a, b)
    assert np.all(ret.asnumpy() == np.array([2, 3, 4]))
    assert_executed_by_graph_mode(foo)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_function_decorated_with_PSJIT_run_ast_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """

    @jit
    def cal_func(*vargs):
        x = vargs[0]
        y = vargs[1]
        if x > 1:
            return y + 1
        return y + 2

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        return cal_func(x, y)

    a = Tensor([2])
    b = Tensor([1, 2, 3])
    ret = foo(a, b)
    assert np.all(ret.asnumpy() == np.array([2, 3, 4]))
    assert_executed_by_graph_mode(foo)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_function_decorated_with_PSJIT_run_ast_3():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """

    @jit
    def cal_func(**kwargs):
        x = kwargs["x"]
        y = kwargs["y"]
        if x > 1:
            return y + 1
        return y + 2

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(m, n):
        return cal_func(y=n, x=m)

    a = Tensor([2])
    b = Tensor([1, 2, 3])
    ret = foo(a, b)
    assert np.all(ret.asnumpy() == np.array([2, 3, 4]))
    assert_executed_by_graph_mode(foo)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_function_decorated_with_PSJIT_run_ast_4():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit
        def construct(self, x, y):
            if x > 1:
                return y + 1
            return y + 2

    net = Net()

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(m, n):
        return net(m, n)

    a = Tensor([2])
    b = Tensor([1, 2, 3])
    ret = foo(a, b)
    assert np.all(ret.asnumpy() == np.array([2, 3, 4]))
    assert_executed_by_graph_mode(foo)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_function_decorated_with_PSJIT_run_ast_5():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit
        def construct(self, *vargs):
            x = vargs[0]
            y = vargs[1]
            if x > 1:
                return y + 1
            return y + 2

    net = Net()

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(m, n):
        return net(m, n)

    a = Tensor([2])
    b = Tensor([1, 2, 3])
    ret = foo(a, b)
    assert np.all(ret.asnumpy() == np.array([2, 3, 4]))
    assert_executed_by_graph_mode(foo)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_function_decorated_with_PSJIT_run_ast_6():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit
        def construct(self, **kwargs):
            x = kwargs["x"]
            y = kwargs["y"]
            if x > 1:
                return y + 1
            return y + 2

    net = Net()

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(m, n):
        return net(y=n, x=m)

    a = Tensor([2])
    b = Tensor([1, 2, 3])
    ret = foo(a, b)
    assert np.all(ret.asnumpy() == np.array([2, 3, 4]))
    assert_executed_by_graph_mode(foo)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_sync_constant_stub_tensor():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            a = Tensor([1])
            b = Tensor([2])
            self.stub_tensor_attr = ops.add(a, b)

        @pi_jit_with_config(jit_config={"compile_with_try": False})
        def construct(self, x):
            if self.stub_tensor_attr == 0:
                return x + 1
            return x - 1

    net = Net()
    input_x = Tensor([1, 2, 3])
    ret = net(input_x)
    assert np.all(ret.asnumpy() == np.array([0, 1, 2]))
    assert_executed_by_graph_mode(net.construct)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_unpack_for_variable_tensor():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    def inner(x, y):
        return x + y

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        m, n = inner(x, y)
        return m, n

    a = Tensor([1, 1])
    b = Tensor([2, 3])
    ret = foo(a, b)
    assert len(ret) == 2
    assert np.all(ret[0].asnumpy() == np.array([3]))
    assert np.all(ret[1].asnumpy() == np.array([4]))
    jcr = get_code_extra(getattr(foo, "__wrapped__", foo))
    assert jcr is not None
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert len(jcr['code']['phase_']) > 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_return_function():
    """
    Feature: ms.jit returning function closure.
    Description: Return a python function from nn.Cell and execute it under jit bytecode mode.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_use.py::test_pijit_return_func
    """

    class PlainNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.k = 1

        def func(self, a):
            def increment(x):
                return (x + a) * self.k

            return increment

        def construct(self, x):
            inc = self.func(3)
            return inc(x)

    class JitNet(PlainNet):
        # TODO: should support fullgraph=True
        @jit(capture_mode="bytecode")
        def construct(self, x):
            inc = self.func(3)  # graph break
            return inc(x)

    input_tensor = Tensor(np.ones((2, 3), np.float32))
    pynative_net = PlainNet()
    jit_net = JitNet()

    pynative_result = pynative_net(input_tensor)
    jit_result = jit_net(Tensor(np.ones((2, 3), np.float32)))

    match_array(pynative_result, jit_result, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_return_bool():
    """
    Feature: ms.jit returning boolean branch.
    Description: Return bool value in helper and use it inside jit bytecode construct.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_use.py::test_pijit_return_bool
    """

    class PlainNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.k = 3

        def func(self, x):
            data = x.asnumpy()
            return bool((data > self.k).all())

        def construct(self, x):
            if self.func(x):
                return x + x
            return x * x

    class JitNet(PlainNet):
        @jit(capture_mode="bytecode")
        def construct(self, x):
            if self.func(x):
                return x + x
            return x * x

    input_np = np.ones((2, 3), np.float32)
    pynative_net = PlainNet()
    jit_net = JitNet()

    pynative_result = pynative_net(Tensor(input_np))
    jit_result = jit_net(Tensor(input_np))

    match_array(pynative_result, jit_result, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_return_none():
    """
    Feature: ms.jit returning None helper.
    Description: Update Parameter inside helper that returns None and use it under jit bytecode mode.
    Expectation: JIT result matches pynative result and keeps parameter synchronized.
    Migrated from: test_pijit_use.py::test_pijit_return_none
    """

    class PlainNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.p = Parameter(Tensor([1, 2], dtype.int32))

        def func(self):
            self.p += 3

        def construct(self, x):
            self.func()
            return x * self.p

    class JitNet(PlainNet):
        @jit(capture_mode="bytecode", fullgraph=True)
        def construct(self, x):
            self.func()
            return x * self.p

    input_tensor = Tensor([2, 3], dtype.float32)
    pynative_net = PlainNet()
    jit_net = JitNet()

    pynative_result = pynative_net(input_tensor)
    jit_result = jit_net(Tensor([2, 3], dtype.float32))

    match_array(pynative_result, jit_result, error=5)
    # TODO: fix this test, pijit does not support Parameter inplace add
    # match_array(pynative_net.p, jit_net.p)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_to_float_preserves_dtype():
    """
    Feature: Bytecode JIT with to_float conversion.
    Description: Compile a cell with bytecode JIT after converting it to float16 and ensure dtype is preserved.
    Expectation: JIT result matches pynative result and keeps float16 dtype.
    Migrated from: test_pijit_ai4sci.py::test_pijit_ai4sci_net_to_float32
    """
    class ScaleAddNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.scale = Tensor([1, 1, 1], dtype.float32)

        def construct(self, x, y):
            product = self.scale * x
            return product + y

    input_np = np.ones((1, 2, 3), np.float16)
    other_np = np.ones((1, 2, 3), np.float16)

    pynative_net = ScaleAddNet().to_float(dtype.float16)
    pynative_result = pynative_net(Tensor(input_np), Tensor(other_np))

    @jit(capture_mode="bytecode")
    def run(a, b):
        net = ScaleAddNet().to_float(dtype.float16)
        return net(Tensor(a), Tensor(b))

    jit_result = run(input_np, other_np)
    assert jit_result.dtype == dtype.float16
    match_array(pynative_result, jit_result)
