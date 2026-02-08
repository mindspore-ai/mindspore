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
# ==============================================================================
import pytest 
import glob
import os
import shutil
import threading
import numpy as np
import mindspore as ms
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import assert_graph_compile_status, match_array


def count_file_key(file, key):
    """Count key string in file"""
    count = 0
    with open(file, 'r') as fp:
        for line in fp:
            if key in line:
                count += 1
    return count


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_enable_dynamic_input_signature_with_jit():
    """
    Feature: ms.enable_dynamic input signature.
    Description: Use enable_dynamic with a dynamic shape tensor and wrap the function by ms.jit.
    Expectation: JIT execution matches pynative execution for different input shapes.
    Migrated from: test_pijit_use.py::test_pijit_input_signature
    """
    input_signature = ms.Tensor(shape=[3, None], dtype=ms.float32)

    @ms.enable_dynamic(x=input_signature)
    def mul_dynamic(x):
        return x * x

    jit_mul = ms.jit(mul_dynamic, capture_mode="bytecode", fullgraph=True)

    tensor_x = ms.Tensor(np.random.rand(3, 4).astype(np.float32), dtype=ms.float32)
    tensor_y = ms.Tensor(np.random.rand(3, 5).astype(np.float32), dtype=ms.float32)

    pynative_out1 = mul_dynamic(tensor_x)
    pynative_out2 = mul_dynamic(tensor_y)

    jit_out1 = jit_mul(tensor_x)
    jit_out2 = jit_mul(tensor_y)

    match_array(pynative_out1, jit_out1)
    match_array(pynative_out2, jit_out2)
    assert_graph_compile_status(jit_mul, 0, 2, 1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape():
    """
    Feature: Support enable_dynamic
    Description: Test dynamic shape
    Expectation: No exception
    """
    @ms.jit(capture_mode="bytecode", fullgraph=True)
    @ms.enable_dynamic(y=ms.Tensor(shape=[2, None], dtype=ms.float32))
    def fn(x, y):
        return x + 1, y + 1

    x1 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x2 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    y2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    y3 = ms.Tensor(np.random.randn(2, 4), ms.float32)

    fn(x1, y1)
    fn(x2, y2)
    fn(x3, y3)
    assert_graph_compile_status(fn, 0, 3, 1) # break_count: 0, call_count: 3, compile_count: 1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_cell():
    """
    Feature: Support enable_dynamic
    Description: Test dynamic shape with nn.Cell
    Expectation: No exception
    """
    class Net(ms.nn.Cell):
        @ms.enable_dynamic(y=ms.Tensor(shape=[2, None], dtype=ms.float32))
        @ms.jit(capture_mode="bytecode", fullgraph=True)
        def construct(self, x, y):
            return x + 1, y + 1

    x1 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x2 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    y2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    y3 = ms.Tensor(np.random.randn(2, 4), ms.float32)

    net = Net()
    net(x1, y1)
    net(x2, y2)
    net(x3, y3)
    assert_graph_compile_status(net.construct, 0, 3, 1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_rank():
    """
    Feature: Support enable_dynamic
    Description: Test dynamic rank and co_freevars.
    Expectation: No exception
    """
    t = ms.Tensor([1], ms.float32)

    @ms.jit(capture_mode="bytecode", fullgraph=True)
    @ms.enable_dynamic(y=ms.Tensor(shape=None, dtype=ms.float32))
    def fn(x, y, z):
        return x + t, y + t

    x1 = ms.Tensor(np.random.randn(4, 3), ms.float32)
    x2 = ms.Tensor(np.random.randn(4, 3), ms.float32)
    x3 = ms.Tensor(np.random.randn(4, 3), ms.float32)

    y1 = ms.Tensor(np.random.randn(1, 3), ms.float32)
    y2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    y3 = ms.Tensor(np.random.randn(3, 4), ms.float32)

    fn(x1, y1, 1)
    fn(x2, y2, 1)
    fn(x3, y3, 1)
    assert_graph_compile_status(fn, 0, 3, 1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_expression():
    """
    Feature: Support enable_dynamic
    Description: Test list_expression and co_cellvars
    Expectation: No exception
    """
    @ms.jit(capture_mode="bytecode", fullgraph=True)
    @ms.enable_dynamic(y=ms.Tensor(shape=None, dtype=ms.float32))
    def fn(x, y, z):
        out = [y + 1 for i in range(3)]
        return out

    x1 = ms.Tensor(np.random.randn(4, 3), ms.float32)
    x2 = ms.Tensor(np.random.randn(4, 3), ms.float32)
    y1 = ms.Tensor(np.random.randn(1, 3), ms.float32)
    y2 = ms.Tensor(np.random.randn(2, 2), ms.float32)

    fn(x1, y1, 1)
    fn(x2, y2, 1)
    assert_graph_compile_status(fn, 0, 2, 1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_multiple_type_args():
    """
    Feature: Support enable_dynamic
    Description: Test *args and **kwargs
    Expectation: No exception
    """
    @ms.jit(capture_mode="bytecode", fullgraph=True)
    @ms.enable_dynamic(x=ms.Tensor(shape=None, dtype=ms.float32),
                       a=ms.Tensor(shape=[2, None], dtype=ms.float32),
                       y=ms.Tensor(shape=None, dtype=ms.float32))
    def fn(x, y, a, b, *args, **kwargs):
        return x + 1, y + 1, a + b, args[0] + args[1]

    x1 = ms.Tensor(np.random.randn(1, 1), ms.float32)
    x2 = ms.Tensor(np.random.randn(1, 2), ms.float32)
    x3 = ms.Tensor(np.random.randn(1, 2, 3), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    y2 = ms.Tensor(np.random.randn(1), ms.float32)
    y3 = ms.Tensor(np.random.randn(1, 2, 3, 4), ms.float32)

    a1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    a2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    a3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    t1 = ms.Tensor(np.random.randn(3, 3), ms.float32)
    t2 = ms.Tensor(np.random.randn(3, 3), ms.float32)
    t3 = ms.Tensor(np.random.randn(3, 3), ms.float32)

    fn(x1, y1, a1, 2, t1, t1)
    fn(x2, y2, a2, 2, t2, t2)
    fn(x3, y3, a3, 2, t3, t3)
    assert_graph_compile_status(fn, 0, 3, 1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fn_callable():
    """
    Feature: Support enable_dynamic
    Description: Test callable
    Expectation: No exception
    """
    def inner_fn(a, b, c):
        return a + 1, b + 1, c + 1

    jit_fn = ms.jit(inner_fn, capture_mode="bytecode", fullgraph=True)
    fn = ms.enable_dynamic(a=ms.Tensor(shape=None, dtype=ms.int32),
                           b=ms.Tensor(shape=None, dtype=ms.int32),
                           c=ms.Tensor(shape=None, dtype=ms.int32))(jit_fn)

    x1 = ms.Tensor(np.random.randn(1, 1), ms.int32)
    x2 = ms.Tensor(np.random.randn(1, 2), ms.int32)
    x3 = ms.Tensor(np.random.randn(1, 2, 3), ms.int32)

    y1 = ms.Tensor(np.random.randn(2, 2), ms.int32)
    y2 = ms.Tensor(np.random.randn(1), ms.int32)
    y3 = ms.Tensor(np.random.randn(1, 2, 3, 4), ms.int32)

    z1 = ms.Tensor(np.random.randn(2), ms.int32)
    z2 = ms.Tensor(np.random.randn(3), ms.int32)
    z3 = ms.Tensor(np.random.randn(2, 3), ms.int32)

    fn(x1, y1, z1)
    fn(x2, y2, z2)
    fn(x3, y3, z3)
    assert_graph_compile_status(inner_fn, 0, 3, 1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mutable_sequence():
    """
    Feature: Support enable_dynamic
    Description: Test mutable tuple/list
    Expectation: No exception
    """
    @ms.jit(capture_mode="bytecode", fullgraph=True)
    @ms.enable_dynamic(y=[ms.Tensor(shape=[None, 1], dtype=ms.float32), ms.Tensor(shape=[2, None], dtype=ms.float32)])
    def fn(x, y):
        return x + 1, y[0] + 1, y[1] + 1

    x1 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x2 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    y2 = ms.Tensor(np.random.randn(3, 1), ms.float32)
    y3 = ms.Tensor(np.random.randn(4, 1), ms.float32)

    z1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    z2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    z3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    fn(x1, ms.mutable([y1, z1]))
    fn(x2, ms.mutable([y2, z2]))
    fn(x3, ms.mutable([y3, z3]))
    assert_graph_compile_status(fn, 0, 3, 1)


@pytest.mark.skip(reason="Symbolic shape for jit is not supported yet")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_symbolic_shape():
    """
    Feature: Support enable_dynamic
    Description: Test symbolic shape
    Expectation: No exception
    """
    target_dir = './pijit_enable_dynamic_symbolic_shape'
    expect_file = '_validate'
    os.environ['MS_DEV_SAVE_GRAPHS'] = '1'
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = target_dir
    os.environ['MS_DEV_DUMP_IR_PASSES'] = expect_file

    @ms.jit(capture_mode="bytecode", fullgraph=True)
    @ms.enable_dynamic(x=ms.Tensor(shape=[2, ms.Symbol(max=4, unique=True)], dtype=ms.float32))
    def fn(x, y):
        return x + 1, y + 1

    x1 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    x3 = ms.Tensor(np.random.randn(2, 4), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    y2 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    y3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    try:
        fn(x1, y1)
        fn(x2, y2)
        fn(x3, y3)
        assert_graph_compile_status(fn, 0, 3, 1)

        ir_files = glob.glob(os.path.join(target_dir, "*" + expect_file + "*.ir"))
        assert len(ir_files) == 1
        assert count_file_key(ir_files[0], '[2, s1<[1,inf]>]') > 1
    finally:
        os.unsetenv('MS_DEV_SAVE_GRAPHS')
        os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
        os.unsetenv('MS_DEV_DUMP_IR_PASSES')
        shutil.rmtree(target_dir)
