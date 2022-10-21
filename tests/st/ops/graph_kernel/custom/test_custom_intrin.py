import pytest
import numpy as np
from mindspore import context, Tensor
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops import kernel

#########################
# test cases for serial #
#########################


@kernel
def add_serial_1(a, b):
    out = output_tensor(a.shape, a.dtype)
    row = a.shape[0]
    col = a.shape[1]
    for i in serial(row):
        for j in range(col):
            out[i, j] = a[i, j] + b[0, j]
    return out


@kernel
def add_serial_2(a, b):
    out = output_tensor(a.shape, a.dtype)
    row = a.shape[0]
    col = a.shape[1]
    for i in range(row):
        for j in serial(col):
            out[i, j] = a[i, j] + b[0, j]
    return out

###########################
# test cases for parallel #
###########################


@kernel
def add_parallel_1(a, b):
    out = output_tensor(a.shape, a.dtype)
    row = a.shape[0]
    col = a.shape[1]
    for i in range(row):
        for j in parallel(col):
            out[i, j] = a[i, j] + b[0, j]
    return out


@kernel
def add_parallel_2(a, b):
    out = output_tensor(a.shape, a.dtype)
    row = a.shape[0]
    col = a.shape[1]
    for i in parallel(row):
        for j in range(col):
            out[i, j] = a[i, j] + b[0, j]
    return out


@kernel
def add_parallel_3(a, b):
    l0 = b.shape[1]
    l1 = a.shape[0]
    l2 = a.shape[1]
    out = output_tensor((l0, l1, l2), a.dtype)
    for i in range(l0):
        for j in parallel(l1):
            for k in range(l2):
                out[i, j, k] = a[j, k] + b[j, i]
    return out


############################
# test cases for vectorize #
############################


@kernel
def add_vectorize_1(a, b):
    out = output_tensor(a.shape, a.dtype)
    row = a.shape[0]
    col = a.shape[1]
    for j in vectorize(col):
        for i in range(row):
            out[i, j] = a[i, j] + b[0, j]
    return out


@kernel
def add_vectorize_2(a, b):
    out = output_tensor(a.shape, a.dtype)
    row = a.shape[0]
    col = a.shape[1]
    for i in range(row):
        for j in vectorize(col):
            out[i, j] = a[i, j] + b[0, j]
    return out


@kernel
def add_vectorize_3(a, b):
    l0 = b.shape[1]
    l1 = a.shape[0]
    l2 = a.shape[1]
    out = output_tensor((l0, l1, l2), a.dtype)
    for i in vectorize(l0):
        for j in range(l1):
            for k in vectorize(l2):
                out[i, j, k] = a[j, k] + b[j, i]
    return out


#########################
# test cases for reduce #
#########################


@kernel
def add_reduce_1(a):
    out = output_tensor((a.shape[0],), a.dtype)
    row = a.shape[0]
    col = a.shape[1]
    for i in range(row):
        out[i] = 0.0
        for k in reduce(col):
            out[i] = out[i] + a[i, k]
    return out


class TestMsHybridDSLSingle(Cell):
    """Net for single input"""

    def __init__(self, func, func_type):
        super(TestMsHybridDSLSingle, self).__init__()

        self.program = ops.Custom(func, func_type=func_type)

    def construct(self, x):
        return self.program(x)


class TestMsHybridDSLBin(Cell):
    """Net for binary inputs"""

    def __init__(self, func, func_type):
        super(TestMsHybridDSLBin, self).__init__()

        self.program = ops.Custom(func, func_type=func_type)

    def construct(self, x, y):
        return self.program(x, y)


def ms_kernel_single_input_test(dtype, num, kernel_fn):
    """
    test case Custom Op with functions written in Hybrid DSL with single input
    """
    support_list = {"float16": np.float16, "float32": np.float32}

    input1 = np.ones((num, num * 2)).astype(support_list.get(dtype))

    test = TestMsHybridDSLSingle(kernel_fn, "hybrid")
    output = test(Tensor(input1))
    expect = kernel_fn(input1)
    compare_res = np.allclose(expect, output.asnumpy(), 0.001, 0.001)
    if not compare_res:
        raise ValueError("Precision error, compare result: {}".format(compare_res))


def ms_kernel_bin_inputs_test(dtype, kernel_fn):
    """
    test case Custom Op with functions written in Hybrid DSL with two inputs
    """
    support_list = {"float16": np.float16, "float32": np.float32}

    input1 = np.ones((1024, 32)).astype(support_list.get(dtype))
    input2 = np.ones((1024, 64)).astype(support_list.get(dtype))

    test = TestMsHybridDSLBin(kernel_fn, "hybrid")
    output = test(Tensor(input1), Tensor(input2))
    expect = kernel_fn(input1, input2)
    compare_res = np.allclose(expect, output.asnumpy(), 0.001, 0.001)
    if not compare_res:
        raise ValueError("Precision error, compare result: {}".format(compare_res))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ms_kernel_ascend_scheduling_intrin():
    """
    Feature: test case for Custom op with new scheduling intrin
    Description: ascend test case, Python DSL with kernel decorator in GRAPH_MODE.
    Expectation: the result match with numpy result
    """

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    ms_kernel_bin_inputs_test(dtype="float32", kernel_fn=add_serial_1)
    ms_kernel_bin_inputs_test(dtype="float32", kernel_fn=add_serial_2)

    ms_kernel_bin_inputs_test(dtype="float32", kernel_fn=add_vectorize_1)
    ms_kernel_bin_inputs_test(dtype="float32", kernel_fn=add_vectorize_2)
    ms_kernel_bin_inputs_test(dtype="float32", kernel_fn=add_vectorize_3)

    ms_kernel_bin_inputs_test(dtype="float32", kernel_fn=add_parallel_1)
    ms_kernel_bin_inputs_test(dtype="float32", kernel_fn=add_parallel_2)
    ms_kernel_bin_inputs_test(dtype="float32", kernel_fn=add_parallel_3)

    ms_kernel_single_input_test(dtype="float32", num=1024, kernel_fn=add_reduce_1)
