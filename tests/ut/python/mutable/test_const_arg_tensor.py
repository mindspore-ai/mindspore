# Copyright 2022-2024 Huawei Technologies Co., Ltd
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
"""test const tensor for network arg"""
import os
import subprocess
import numpy as np
import mindspore as ms
from mindspore.ops.composite import GradOperation
from mindspore.common import mutable
from mindspore import ops
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, context, jit


def test_grad_constant_tensor():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get gradient with respect to the constant tensor input.
    Expectation: Get an empty gradient.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, x, y):
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    grad_net = GradNetWrtX(Net())
    output = grad_net(x, y)
    assert isinstance(output, tuple)
    assert output == ()


def test_grad_constant_tensor_mixed_call():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get gradient with respect to the constant tensor input for mixed call of mutable and const_arg.
    Expectation: Get an empty gradient.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, x, y):
            out = self.matmul(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = GradOperation()

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    x = mutable(x)
    x.set_const_arg(True)
    grad_net = GradNetWrtX(Net())
    output = grad_net(x, y)
    assert isinstance(output, tuple)
    assert output == ()
    grad_net = GradOperation()(Net())
    output = grad_net(x, y)
    assert isinstance(output, tuple)
    assert output == ()


def test_ms_function_grad_constant_tensor():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get gradient with respect to the constant tensor input of ms_function.
    Expectation: Get an empty gradient.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, x, y):
            out = self.matmul(x, y)
            return out

    @jit
    def fn(x, y):
        net = Net()
        grad_op = GradOperation()
        return grad_op(net)(x, y)

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    output = fn(x, y)
    assert isinstance(output, tuple)
    assert output == ()


def test_tensor_constant_folding():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get result of add operator for two constant tensor by constant folding in frontend.
    Expectation: Get a correct result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.add = P.Add()

        def construct(self, x, y):
            out = self.add(x, y)
            return out

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32, const_arg=True)
    net = Net()
    output = net(x, y)
    expect_output = np.array([[0.51, 0.9, 1.5],
                              [1.3, 1.5, 2.4]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_output)


def test_ms_function_tensor_constant_folding():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get result of add operator of ms_function for two constant tensor by constant folding in frontend.
    Expectation: Get a correct result.
    """

    @jit
    def fn(x, y):
        return P.Add()(x, y)

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32, const_arg=True)
    output = fn(x, y)
    expect_output = np.array([[0.51, 0.9, 1.5],
                              [1.3, 1.5, 2.4]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_output)


def test_constant_tensor_if():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get result of control flow with if for constant tensor.
    Expectation: Get the correct result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.z = Tensor([3], dtype=mstype.int32)

        def construct(self, x, y):
            out = y
            if x < self.z:
                out = out + y
            return out

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([0], dtype=mstype.int32, const_arg=True)
    y = Tensor([1], dtype=mstype.int32, const_arg=True)
    net = Net()
    output = net(x, y)
    expect_output = np.array([2]).astype(np.int32)
    assert np.allclose(output.asnumpy(), expect_output)


def test_ms_function_constant_tensor_if():
    """
    Feature: Set mutable tensor input to constant.
    Description: Get result of control flow with if of ms_function for constant tensor.
    Expectation: Get the correct result.
    """

    @jit
    def fn(x, y):
        z = Tensor([3], dtype=mstype.int32)
        out = y
        if x < z:
            out = out + y
        return out

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([0], dtype=mstype.int32, const_arg=True)
    y = Tensor([1], dtype=mstype.int32, const_arg=True)
    output = fn(x, y)
    expect_output = np.array([2]).astype(np.int32)
    assert np.allclose(output.asnumpy(), expect_output)


def test_check_mutable_value():
    """
    Feature: Set mutable tensor input to constant.
    Description: Check the illegal arg.
    Expectation: Raise the correct error log.
    """
    context.set_context(mode=context.GRAPH_MODE)
    try:
        x = Tensor([0], dtype=mstype.int32, const_arg=1)
    except TypeError as e:
        assert str(e) == "For 'Tensor', the type of 'const_arg' should be 'bool', but got type 'int'."

    try:
        x = Tensor([0], dtype=mstype.int32)
        x.set_const_arg(1)
    except TypeError as e:
        assert str(e) == "For 'set_const_arg', the type of 'const_arg' should be 'bool', but got type 'int'."


def test_run_same_value_const_tensor_run_twice():
    """
    Feature: Check const tensor in tuple.
    Description: Check warning raising when const tensor arg in tuple.
    Expectation: No warning raised when const arg cause recompiling.
    """

    @ms.jit
    def net_func_tuple_input(tuple_input):
        return ops.add(tuple_input[0], tuple_input[1])

    const_tensor1 = ms.Tensor([1])
    const_tensor2 = ms.Tensor([1])
    net_func_tuple_input((const_tensor1, const_tensor2))
    net_func_tuple_input((const_tensor1, const_tensor2))


def test_run_different_value_const_tensor_run_twice():
    """
    Feature: Check const tensor in tuple.
    Description: Check warning raising when const tensor arg in tuple.
    Expectation: Warning raised when const arg cause recompiling.
    """

    @ms.jit
    def net_func_tuple_input(tuple_input):
        return ops.add(tuple_input[0], tuple_input[1])

    const_tensor1 = ms.Tensor([1])
    const_tensor2 = ms.Tensor([2])
    net_func_tuple_input((const_tensor1, const_tensor1))
    net_func_tuple_input((const_tensor2, const_tensor2))


def test_run_cell_different_value_const_tensor_run_twice():
    """
    Feature: Check const tensor in tuple.
    Description: Check warning raising when const tensor arg in tuple.
    Expectation: Warning raised when const arg cause recompiling.
    """

    class Net(nn.Cell):
        def construct(self, tuple_input):
            return ops.add(tuple_input[0], tuple_input[1])

    const_tensor1 = ms.Tensor([1])
    const_tensor2 = ms.Tensor([2])
    context.set_context(mode=context.GRAPH_MODE)
    net_func_tuple_input = Net()
    net_func_tuple_input((const_tensor1, const_tensor1))
    net_func_tuple_input((const_tensor2, const_tensor2))


def test_run_different_value_mutable_tensor_run_twice():
    """
    Feature: Check const tensor in tuple.
    Description: Check warning raising when const tensor arg in tuple.
    Expectation: No warning raised.
    """

    @ms.jit
    def net_func_tuple_input(tuple_input):
        return ops.add(tuple_input[0], tuple_input[1])

    const_tensor1 = ms.Tensor([1])
    const_tensor2 = ms.Tensor([2])
    net_func_tuple_input(mutable((const_tensor1, const_tensor1)))
    net_func_tuple_input(mutable((const_tensor2, const_tensor2)))


def test_run_different_value_const_tensor_run_twice2():
    """
    Feature: Check const tensor in tuple.
    Description: Check warning raising when const tensor arg in tuple-tuple.
    Expectation: Warning raised when const arg cause recompiling.
    """

    @ms.jit
    def net_func_tuple_tuple_input(tuple_tuple_input):
        return ops.add(tuple_tuple_input[0][0], tuple_tuple_input[0][1])

    const_tensor1 = ms.Tensor([1])
    const_tensor2 = ms.Tensor([2])
    net_func_tuple_tuple_input(((const_tensor1, const_tensor1),))
    net_func_tuple_tuple_input(((const_tensor2, const_tensor2),))


def test_create_cell_instance_twice():
    """
    Feature: Check create same cell instance twice in function.
    Description: Check warning raising when same cell create twice.
    Expectation: Warning raised when cell create twice.
    """

    class Net(nn.Cell):
        def construct(self, input_arg):
            return ops.add(input_arg, input_arg)

    def exec_cell_obj(arg):
        return Net()(arg)

    const_tensor1 = ms.Tensor([1])
    context.set_context(mode=context.GRAPH_MODE)
    exec_cell_obj(const_tensor1)
    exec_cell_obj(const_tensor1)


def test_create_cell_instance_twice_with_differ_args():
    """
    Feature: Check create same cell instance twice in function.
    Description: Check warning raising when same cell create twice.
    Expectation: No warning raised when cell create twice with different init args.
    """

    class Net(nn.Cell):
        def __init__(self, arg0, arg1):
            super(Net, self).__init__()
            self.arg0 = arg0
            self.arg1 = arg1

        def construct(self, input_arg):
            return ops.add(input_arg, input_arg)

    def exec_cell_obj(arg, init_arg0, init_ar1):
        return Net(init_arg0, init_ar1)(arg)

    const_tensor1 = ms.Tensor([1])
    context.set_context(mode=context.GRAPH_MODE)
    exec_cell_obj(const_tensor1, 1, [2, 3])
    exec_cell_obj(const_tensor1, 1, [4, 5])

def test_create_cell_instance_twice_no_warning():
    """
    Feature: Check create same cell instance twice.
    Description: Check warning raising when same cell create twice.
    Expectation: No warning raised when cell create twice.
    """

    class Net(nn.Cell):
        def construct(self, input_arg):
            return ops.add(input_arg, input_arg)


    const_tensor1 = ms.Tensor([1])
    context.set_context(mode=context.GRAPH_MODE)
    net1 = Net()
    net2 = Net()
    out1 = net1(const_tensor1)
    out2 = net2(const_tensor1)
    print("out1:", out1)
    print("out2:", out2)

def test_check_const_tensor_in_tuple_warning():
    """
    Feature: Check const tensor in tuple.
    Description: Check warning raising when const tensor arg in tuple.
    Expectation: Warning raised when const arg cause recompiling.
    """

    def run_test_case_and_save_log(file_name, case_name, filter_str, is_in):
        _cur_dir = os.path.dirname(os.path.realpath(__file__))
        file_name = os.path.join(_cur_dir, file_name)
        assert os.path.exists(file_name)

        log_file_name = case_name + "_tmp_log"
        log_file_name = os.path.join(_cur_dir, log_file_name)
        if os.path.exists(log_file_name):
            os.remove(log_file_name)
        assert not os.path.exists(log_file_name)
        cmd_first = f"GLOG_v=2 pytest -s " + file_name + "::" + case_name + " > " + log_file_name + " 2>&1"
        subprocess.check_output(cmd_first, shell=True)
        assert os.path.exists(log_file_name)
        with open(log_file_name, "r") as f_first:
            data_first = f_first.read()
        if is_in:
            assert filter_str in data_first
        else:
            assert filter_str not in data_first

        # Clean files
        os.remove(log_file_name)

    # Same value no warning log.
    run_test_case_and_save_log("test_const_arg_tensor.py", "test_run_same_value_const_tensor_run_twice",
                               "Constant value tensor are detected in tuple or list", False)

    # Different value of tuple print warning log.
    run_test_case_and_save_log("test_const_arg_tensor.py", "test_run_different_value_const_tensor_run_twice",
                               "Constant value tensor are detected in tuple or list", True)

    # Different value of tuple print warning log with cell.
    run_test_case_and_save_log("test_const_arg_tensor.py", "test_run_cell_different_value_const_tensor_run_twice",
                               "Constant value tensor are detected in tuple or list", True)

    # Different mutable value of tuple no warning log.
    run_test_case_and_save_log("test_const_arg_tensor.py", "test_run_different_value_mutable_tensor_run_twice",
                               "Constant value tensor are detected in tuple or list", False)

    # Different value of tuple-tuple print warning log.
    run_test_case_and_save_log("test_const_arg_tensor.py", "test_run_different_value_const_tensor_run_twice2",
                               "Constant value tensor are detected in tuple or list", True)

    # Create instance twice print warning log
    # run_test_case_and_save_log("test_const_arg_tensor.py", "test_create_cell_instance_twice",
    #                            "only once to avoid recompiling", True)

    # Create instance twice no warning log
    run_test_case_and_save_log("test_const_arg_tensor.py", "test_create_cell_instance_twice_no_warning",
                               "only once to avoid recompiling", False)
