# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
""" test whileloop, scan, foriloop """
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, jit, context, ops, nn, Parameter
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark


context.set_context(mode=context.GRAPH_MODE)
context.set_context(jit_config={"jit_level": "O0"})


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=2048.0)
def test_while_loop():
    """
    Feature: control flow
    Description: Using WhileLoopEvaluator to handle ops.WhileLoop operation
    Expectation: No exception.
    """

    def complex_pure_function(init_value):
        input_tensor = init_value
        activation = ops.ReLU()
        fc = activation(input_tensor)
        return  fc + 1

    def cond_func(init_value):
        return init_value.value() < 100

    @jit(backend="ms_backend")
    def test_while_loop_inner(init_val):
        whileop = ops.WhileLoop()
        result = whileop(cond_func, complex_pure_function, init_val)
        return result

    dtype = mstype.float32
    input_tensor = ms.Parameter(Tensor([1], dtype))
    init_state = input_tensor
    result = test_while_loop_inner(init_state)
    assert result == 100


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_while_loop2():
    """
    Feature: control flow
    Description: Using ScanEvaluator to handle ops.Scan operation
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.whileop = ops.WhileLoop()
            self.activation = ops.ReLU()

        def construct(self, result_init):
            def complex_pure_function(init_value):
                input_tensor, init = init_value
                fc = self.activation(input_tensor)
                init = init + 1
                return [fc, init]

            def cond_func(init_value):
                return init_value[-1] < 100

            result = self.whileop(cond_func, complex_pure_function, result_init)
            return result

    dtype = mstype.float32
    input_tensor = Tensor(np.random.rand(6, 6), dtype)
    init = 0
    init_state = [input_tensor, init]
    net = Net()
    result = net(init_state)
    assert result[-1] == 100


@arg_mark(plat_marks=['platform_gpu',], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_while_loop3():
    """
    Feature: control flow
    Description: Using WhileLoopEvaluator to handle ops.WhileLoop operation
    Expectation: No exception.
    """

    def cond_func(init_value):
        return init_value[1] > 1

    def while_function(init_value):
        input_tensor, init, add = init_value
        out = add(input_tensor, init)
        init = init - 1
        return [out, init, add]

    class WhileLoopNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = ops.Add()
            self.whileop = ops.WhileLoop()

        def construct(self, inputs):
            out = inputs
            res = self.whileop(cond_func, while_function, [out, 3, self.add])
            out = res[0]
            return out

    net = WhileLoopNet()
    out = net(Tensor([2]))
    assert out == 7


def generator_dataset(size, x_shape=(32, 1), y_shape=(32, 1)):
    for _ in range(size):
        x = np.full(x_shape, 0.1, dtype=np.float32)
        y = np.full(y_shape, 0.2, dtype=np.float32)
        yield (x, y)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_while_loop_with_nested_for():
    """
    Feature: control flow
    Description: Using WhileLoopEvaluator to handle ops.WhileLoop operation
    Expectation: No exception.
    """
    def cond_func(init_value):
        return init_value[1] > 1

    def for_in_while_function(init_value):
        input_tensor, init = init_value
        add = ops.Add()
        out = add(input_tensor, init)
        for _ in range(3):
            out = add(out, init)
        init = init - 1
        return [out, init]

    class WhileForNet(nn.Cell):
        def __init__(self, loop_op=False, size_1=(1, 1)):
            super().__init__()
            self.loop_op = loop_op
            self.add = ops.Add()
            self.whileop = ops.WhileLoop()

            self.weight_1 = Parameter(Tensor(np.full(size_1, 0.5, dtype=np.float32)), name="weight_1")

        def construct(self, inputs, loop_times=3):
            out = inputs
            if self.loop_op:
                res = self.whileop(cond_func, for_in_while_function, [out, loop_times])
                out = res[0]
            else:
                while loop_times > 1:
                    out = self.add(out, loop_times)
                    for _ in range(3):
                        out = self.add(out, loop_times)
                    loop_times = loop_times - 1
            out = self.add(out, self.weight_1)
            return out

    input_x = np.random.rand(2)
    net_1 = WhileForNet(loop_op=False)
    out1 = net_1(Tensor(input_x))

    net_2 = WhileForNet(loop_op=True)
    out2 = net_2(Tensor(input_x))
    np.allclose(out1.asnumpy(), out2.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=2048.00)
def test_scan_unroll():
    """
    Feature: control flow
    Description: Using ScanEvaluator to handle ops.Scan operation
    Expectation: No exception.
    """

    def complex_pure_function(input_tensor, el):
        activation = ops.ReLU()
        fc = activation(input_tensor)
        return fc, el

    @jit(backend="ms_backend")
    def test_scan_inner(result_init, array):
        scan_op = ops.Scan()
        return scan_op(complex_pure_function, result_init, array, len(array), True)

    array = []
    result_init = Tensor(np.random.rand(6, 6), mstype.float32)
    i = 0
    while i < 10:
        input_tensor = Tensor(np.random.rand(6, 6), mstype.float32)
        array.append(input_tensor)
        i = i + 1
    result = test_scan_inner(result_init, array)
    assert len(result[-1]) == 10


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.0)
def test_scan_not_unroll():
    """
    Feature: control flow
    Description: Using ScanEvaluator to handle ops.Scan operation
    Expectation: No exception.
    """

    def complex_pure_function(input_tensor, el):
        activation = ops.ReLU()
        fc = activation(input_tensor)
        return fc, el

    @jit(backend="ms_backend")
    def test_scan_inner(result_init, array):
        scan_op = ops.Scan()
        return scan_op(complex_pure_function, result_init, array, len(array), False)

    array = []
    result_init = Tensor(np.random.rand(6, 6), mstype.float32)
    i = 0
    while i < 10:
        input_tensor = Tensor(np.random.rand(6, 6), mstype.float32)
        array.append(input_tensor)
        i = i + 1
    result = test_scan_inner(result_init, array)
    assert len(result[-1]) == 10


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
def test_scan_simple_loop():
    """
    Feature: control flow
    Description: Using ScanEvaluator to handle ops.Scan operation
    Expectation: No exception.
    """

    def simple_loop_func(res, el):
        res = res + el
        return res, res

    @jit(backend="ms_backend")
    def test_simple_scan_inner(result_init):
        array = [1, 2, 3, 4]
        scan_op = ops.Scan()
        return scan_op(simple_loop_func, result_init, array, len(array), False)

    result_init = ms.Tensor(0)
    result = test_simple_scan_inner(result_init)
    assert result == (10, [1, 3, 6, 10])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scan_with_nested_foriloop():
    """
    Feature: control flow
    Description: Using ScanEvaluator to handle ops.Scan operation
    Expectation: No exception.
    """
    def fori_loop_func(_, init_value):
        input_tensor, init = init_value
        add = ops.Add()
        out = add(input_tensor, init)
        return [out, init]

    def foriloop_in_scan_function(input_tensor, el):
        add = ops.Add()
        out = add(input_tensor, 1)
        fori_loop = ops.ForiLoop()
        res = fori_loop(0, 3, fori_loop_func, [out, el])
        return res[0], el

    class ScanForiLoopNet(nn.Cell):
        def __init__(self, loop_op=False, size_1=(1, 1)):
            super().__init__()
            self.loop_op = loop_op
            self.add = ops.Add()
            self.scanop = ops.Scan()

            self.weight_1 = Parameter(Tensor(np.full(size_1, 0.5, dtype=np.float32)), name="weight_1")

        def construct(self, inputs, x=3, y=10):
            out = inputs
            if self.loop_op:
                res = self.scanop(foriloop_in_scan_function, inputs, [x, y], 2, False)
                out = res[0]
            else:
                for i in [x, y]:
                    out = self.add(out, 1)
                    for _ in range(3):
                        out = self.add(out, i)
            out = self.add(out, self.weight_1)
            return out

    input_x = np.random.rand(2)
    net_1 = ScanForiLoopNet(loop_op=False)
    out1 = net_1(Tensor(input_x))

    net_2 = ScanForiLoopNet(loop_op=True)
    out2 = net_2(Tensor(input_x))
    np.allclose(out1.asnumpy(), out2.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=2048.0)
def test_foriloop_unroll():
    """
    Feature: control flow
    Description: Using ForiLoopEvaluator to handle ops.ForiLoop operation
    Expectation: No exception.
    """

    def complex_pure_function(index, val):
        add = ops.Add()
        return add(val, index)

    @jit(backend="ms_backend")
    def test_fori_loop_inner(result_init):
        fori_loop = ops.ForiLoop()
        return fori_loop(0, 10, complex_pure_function, result_init)

    result_init = Tensor(0, mstype.float32)
    result = test_fori_loop_inner(result_init)
    print(result)
    assert result == 45


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=2048.00)
def test_foriloop_with_nested_foriloop():
    """
    Feature: control flow
    Description: Using ForiLoopEvaluator to handle ops.ForiLoop operation
    Expectation: No exception.
    """
    def fori_loop_func(_, init_value):
        input_tensor, init = init_value
        add = ops.Add()
        out = add(input_tensor, init)
        return [out, init]

    def foriloop_in_foriloop_function(index, input_tensor):
        add = ops.Add()
        out = add(input_tensor, 1)
        fori_loop = ops.ForiLoop()
        res = fori_loop(0, 3, fori_loop_func, [out, index])
        return res[0]

    class ForiLoopForiLoopNet(nn.Cell):
        def __init__(self, loop_op=False, size_1=(32, 1)):
            super().__init__()
            self.loop_op = loop_op
            self.add = ops.Add()
            self.fori_loop = ops.ForiLoop()

            self.weight_1 = Parameter(Tensor(np.full(size_1, 0.5, dtype=np.float32)), name="weight_1")

        def construct(self, inputs, x=3, y=10):
            out = inputs
            if self.loop_op:
                out = self.fori_loop(x, y, foriloop_in_foriloop_function, out, False)
            else:
                for i in range(x, y):
                    out = self.add(out, 1)
                    for _ in range(3):
                        out = self.add(out, i)
            out = self.add(out, self.weight_1)
            return out

    input_x = np.random.rand(2)
    net_1 = ForiLoopForiLoopNet(loop_op=False)
    out1 = net_1(Tensor(input_x))

    net_2 = ForiLoopForiLoopNet(loop_op=True)
    out2 = net_2(Tensor(input_x))
    np.allclose(out1.asnumpy(), out2.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
def test_high_order_with_unroll_as_false():
    """
    Feature: control flow
    Description: test higher order grad with unroll as false
    Expectation: Raise error with unsupported reason.
    """
    def for_in_foriloop_function(index, input_tensor):
        add = ops.Add()
        out = add(input_tensor, 1)
        for _ in range(3):
            out = add(out, index)
        return out

    class ForiLoopForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.fori_loop = ops.ForiLoop()

        def construct(self, inputs):
            out = inputs
            out = self.fori_loop(0, 7, for_in_foriloop_function, out, False)
            return out

    @jit
    def get_grad(x):
        net_2 = ForiLoopForNet()
        grad_net_2_f = ops.grad(net_2) # pylint: disable=E1102
        grad_net_2_s = ops.grad(grad_net_2_f) # pylint: disable=E1102
        return grad_net_2_s(x)

    with pytest.raises(RuntimeError, match="Loop op with unroll set as false is not allow do higher order grad"):
        x = Tensor(np.random.randn(32, 1).astype(np.float32))
        get_grad(x)


@pytest.mark.skip(reason="Cannot process ops.xx both in loop_func and whileloop declaration in construct")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_while_loop_unsupport1():
    """
    Feature: control flow
    Description: Using ScanEvaluator to handle ops.Scan operation
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, result_init):

            def complex_pure_function(init_value):
                input_tensor, init = init_value
                activation = ops.ReLU()
                fc = activation(input_tensor)
                init = init + 1
                return [fc, init]

            def cond_func(init_value):
                return init_value[-1] < 100

            whileop = ops.WhileLoop()
            result = whileop(cond_func, complex_pure_function, result_init)
            return result

    dtype = mstype.float32
    input_tensor = Tensor(np.random.rand(6, 6), dtype)
    init = 0
    init_state = [input_tensor, init]
    net = Net()
    result = net(init_state)
    assert result[-1] == 100


@pytest.mark.skip(reason="Unsupported loop func with side effect")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
def test_scan_unsupport1():
    """
    Feature: control flow
    Description: Using ScanEvaluator to handle ops.Scan operation
    Expectation: No exception.
    """

    def complex_pure_function(input_tensor, el):
        activation = ops.ReLU()
        assign_op = ops.Assign()
        assign_op(input_tensor, activation(input_tensor))
        return input_tensor, input_tensor

    @jit(backend="ms_backend")
    def test_scan_inner(result_init, array):
        scan_op = ops.Scan()
        return scan_op(complex_pure_function, result_init, array, len(array), True)

    array = []
    result_init = ms.Parameter(Tensor(np.random.rand(6, 6), mstype.float32))
    i = 0
    while i < 10:
        input_tensor = Tensor(np.random.rand(6, 6), mstype.float32)
        array.append(input_tensor)
        i = i + 1
    result = test_scan_inner(result_init, array)
    assert len(result[-1]) == 10
