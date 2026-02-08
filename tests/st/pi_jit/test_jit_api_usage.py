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
"""Test jit api usage"""

import numpy as np

import mindspore as ms
from mindspore import Tensor

from tests.mark_utils import arg_mark
from tests.st.pi_jit.one_stage.test_utils import save_graph_ir, check_ir_num
from tests.st.pi_jit.share.utils import assert_equal, match_array


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_wrap_cell_class():
    """
    Feature: ms.jit(class) usage.
    Description: Apply ms.jit with capture_mode='bytecode' as a class decorator for nn.Cell.
    Expectation: JIT compiled class produces the same result as pynative execution.
    Migrated from: test_pijit_use.py::test_pijit_wrap_cell_class
    """

    class PopNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.idx = -1

        def construct(self, items):
            return items.pop(self.idx)

    pynative_net = PopNet()
    pynative_result = pynative_net([1, 2, 3])

    JitPopNet = ms.jit(PopNet, capture_mode="bytecode")
    jit_net = JitPopNet()
    jit_result = jit_net([1, 2, 3])

    assert_equal(pynative_result, jit_result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_wrap_cell_instance():
    """
    Feature: ms.jit(Cell instance) usage.
    Description: Wrap an nn.Cell instance with ms.jit using capture_mode='bytecode'.
    Expectation: JIT wrapped instance matches pynative execution.
    Migrated from: test_pijit_use.py::test_pijit_wrap_cell_instance
    """

    class PopNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.idx = -1

        def construct(self, items):
            return items.pop(self.idx)

    pynative_net = PopNet()
    pynative_result = pynative_net([1, 2, 3])

    jit_net = ms.jit(pynative_net, capture_mode="bytecode")
    jit_result = jit_net([1, 2, 3])

    assert_equal(pynative_result, jit_result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_wrap_net_method():
    """
    Feature: ms.jit(method) usage.
    Description: Wrap the construct method of nn.Cell with ms.jit capture_mode='bytecode'.
    Expectation: JIT wrapped method matches pynative execution.
    Migrated from: test_pijit_use.py::test_pijit_wrap_net_method
    """

    class PopNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.idx = -1

        def construct(self, items):
            return items.pop(self.idx)

    net = PopNet()
    pynative_result = net([1, 2, 3])

    jit_construct = ms.jit(net.construct, capture_mode="bytecode")
    jit_result = jit_construct([1, 2, 3])

    assert_equal(pynative_result, jit_result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_wrap_callable_object():
    """
    Feature: ms.jit(callable object) usage.
    Description: Wrap a callable object with ms.jit capture_mode='bytecode'.
    Expectation: JIT wrapped callable matches pynative execution.
    Migrated from: test_pijit_use.py::test_pijit_wrap_other_instance
    """

    class PopCallable:
        def __init__(self):
            self.idx = -1

        def __call__(self, items):
            return items.pop(self.idx)

    callable_obj = PopCallable()
    pynative_result = callable_obj([1, 2, 3])

    jit_callable = ms.jit(callable_obj, capture_mode="bytecode")
    jit_result = jit_callable([1, 2, 3])

    assert_equal(pynative_result, jit_result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_with_custom_config():
    """
    Feature: ms.jit configuration.
    Description: Use ms.jit with capture_mode='bytecode' and jit_level='O0' inside nn.Cell.
    Expectation: JIT compiled network matches pynative execution.
    Migrated from: test_pijit_use.py::test_pijit_jit_config
    """

    @ms.jit(capture_mode="bytecode", jit_level="O0")
    def double_tensor(x):
        return x + x

    class ConfigNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.scale = 1

        def construct(self, x):
            mid = double_tensor(x) * self.scale
            return mid + mid

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    tensor_input = Tensor(input_np)

    pynative_net = ConfigNet()
    pynative_result = pynative_net(tensor_input)

    jit_net = ConfigNet()
    jit_net.construct = ms.jit(jit_net.construct, capture_mode="bytecode")
    jit_result = jit_net(tensor_input)

    assert_equal(pynative_result, jit_result)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_capture_function_matches_pynative():
    """
    Feature: ms.jit function capture.
    Description: Wrap a standalone function with ms.jit capture_mode='bytecode' and compare with pynative execution.
    Expectation: JIT result matches pynative result and generates one graph.
    Migrated from: test_pijit_catch.py::test_pijit_catch_pfunc
    """

    def func(x):
        return x + x

    input_np = np.random.rand(2, 3).astype(np.float32)
    tensor = Tensor(input_np)

    pynative_result = func(tensor)

    jit_func = ms.jit(func, capture_mode="bytecode")
    jit_result = jit_func(tensor)

    match_array(pynative_result, jit_result)
    check_ir_num('graph_before_compile', 1)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_capture_cell_instance_with_wrapper():
    """
    Feature: ms.jit(Cell instance) with shared context.
    Description: Wrap an nn.Cell instance with ms.jit while another Cell keeps using the original instance in pynative mode.
    Expectation: JIT result matches pynative result and generates one graph.
    Migrated from: test_pijit_catch.py::test_pijit_catch_decorate_cell
    """

    class SquareNet(ms.nn.Cell):
        def construct(self, x):
            return x * x

    class ForwardWrapper(ms.nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x):
            return self.net(x)

    input_main = np.random.rand(2, 3).astype(np.float32)
    tensor_main = Tensor(input_main)
    input_aux = np.random.rand(3, 2).astype(np.float32)
    tensor_aux = Tensor(input_aux)

    pynative_net = SquareNet()
    pynative_result = pynative_net(tensor_main)

    jit_net = ms.jit(SquareNet(), capture_mode="bytecode")
    jit_result = jit_net(tensor_main)

    match_array(pynative_result, jit_result)

    wrapper = ForwardWrapper(SquareNet())
    wrapper_result = wrapper(tensor_aux)
    expected_wrapper = Tensor(input_aux * input_aux)
    match_array(wrapper_result, expected_wrapper)

    check_ir_num('graph_before_compile', 1)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_capture_bound_method_from_custom_class():
    """
    Feature: ms.jit bound method capture.
    Description: Capture a class bound method with ms.jit capture_mode='bytecode'.
    Expectation: JIT result matches pynative result and generates one graph.
    Migrated from: test_pijit_catch.py::test_pijit_catch_decorate_custom_class
    """

    class MyClass:
        def func(self, x):
            return x + x

    instance = MyClass()

    input_np = np.random.rand(2, 3).astype(np.float32)
    tensor = Tensor(input_np)

    pynative_result = instance.func(tensor)

    jit_func = ms.jit(instance.func, capture_mode="bytecode")
    jit_result = jit_func(tensor)

    match_array(pynative_result, jit_result)
    check_ir_num('graph_before_compile', 1)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_capture_function_with_graph_break():
    """
    Feature: ms.jit function containing graph break.
    Description: Use ms.jit capture_mode='bytecode' on a function calling Tensor.asnumpy within its body.
    Expectation: JIT result matches pynative result and generates one graph.
    Migrated from: test_pijit_catch.py::test_pijit_catch_multi_func
    """

    def func1(x: Tensor):
        a = x.asnumpy() * 2
        y = Tensor(a)
        return x + y

    def func(x):
        return func1(x)

    input_np = np.random.rand(2, 3).astype(np.float32)
    tensor = Tensor(input_np)

    pynative_result = func(tensor)

    jit_func = ms.jit(func, capture_mode="bytecode")
    jit_result = jit_func(tensor)

    match_array(pynative_result, jit_result)
    check_ir_num('graph_before_compile', 1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_nested_function_call_chains():
    """
    Feature: ms.jit nested function usage.
    Description: Call a jitted function through multiple Python call layers and support different tensor dtypes.
    Expectation: Nested calls return expected tensor values.
    Migrated from: test_pijit_catch.py::test_pijit_catch_func_nested
    """

    @ms.jit(capture_mode="bytecode")
    def func1(x: Tensor):
        return x * 2

    def func2(x: Tensor):
        return func1(x)

    def func3(x: Tensor):
        return func2(x)

    input_fp32 = np.random.rand(2, 3).astype(np.float32)
    tensor_fp32 = Tensor(input_fp32)
    out_fp32 = func2(tensor_fp32)
    expected_fp32 = Tensor(2 * input_fp32)
    match_array(out_fp32, expected_fp32)

    input_fp64 = np.random.rand(3, 2).astype(np.float64)
    tensor_fp64 = Tensor(input_fp64)
    out_fp64 = func3(tensor_fp64)
    expected_fp64 = Tensor(2 * input_fp64)
    match_array(out_fp64, expected_fp64)
