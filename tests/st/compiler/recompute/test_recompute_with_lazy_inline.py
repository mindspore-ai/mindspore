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

import os
import re
import subprocess
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from mindspore.common import Tensor, Parameter
from mindspore import context, lazy_inline, nn, ops
import mindspore.common.dtype as dtype

match_dyn_mem = re.compile(r'Used peak memory usage \(without fragments\)\: (.*?)M', re.S)


def get_max(mem_uses):
    max_mem = 0
    for i in mem_uses:
        max_mem = max(max_mem, int(i))
    return max_mem


def run_testcase(testcase_name, expect_memory_usage):
    # Clear log file
    log_filename = testcase_name + ".log"
    if os.path.exists(log_filename):
        os.remove(log_filename)
    assert not os.path.exists(log_filename)

    cmd = (f"export GLOG_v=1; export MS_ALLOC_CONF=\"memory_recycle:False\"; "
           f"export MS_DEV_RUNTIME_CONF=\"ge_kernel:False\"; pytest -s test_recompute.py::") + \
          testcase_name + " > " + log_filename + " 2>&1"
    subprocess.check_output(cmd, shell=True)
    assert os.path.exists(log_filename)
    with open(log_filename, "r") as f:
        data = f.read()
    mem_uses = re.findall(match_dyn_mem, data)
    assert len(mem_uses) == 2
    max_mem = get_max(mem_uses)
    assert max_mem == expect_memory_usage
    # Clear log file
    os.remove(log_filename)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1.00)
def test_recompute_cell_recompute():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by the cell recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_block_recompute", 43)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_recompute_cell_recompute_func_api():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by the recomputed func api.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_block_recompute_func_api", 43)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_recompute_cell_recompute_func_api_with_kwargs():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by the recomputed func api with kwargs.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_block_recompute_func_api_with_kwargs", 42)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
@test_utils.run_test_with_On
def test_recompute_op_recompute1():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by the primitive recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_op_recompute1", 63)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_recompute_op_recompute2():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by the primitive recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_op_recompute2", 50)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_recompute_op_recompute3():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by the primitive recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_op_recompute3", 136)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1.00)
def test_recompute_cell_and_op_recompute1():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by both the primitive and cell recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_cell_and_op_recompute1", 63)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_recompute_cell_and_op_recompute_with_func_api():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by both the primitive api and the recomputed func api.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_cell_and_op_recompute_with_func_api", 63)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_recompute_cell_and_op_recompute2():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by both the primitive and cell recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_cell_and_op_recompute2", 67)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@test_utils.run_test_with_On
def test_recompute_cell_and_op_recompute_with_tuple_outputs1():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by both the primitive and cell recompute api and return a tuple.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_cell_and_op_recompute_with_tuple_outputs1", 65)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_recompute_cell_and_op_recompute_with_tuple_outputs2():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by both the primitive and cell recompute api and return a tuple.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_cell_and_op_recompute_with_tuple_outputs2", 65)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
def test_recompute_origin_inputs_umonad_fv():
    """
    Feature: Recompute with lazy inline.
    Description: Recomputed cell used the umonad fv from the origin inputs.
    Expectation: Run successfully.
    """

    context.set_context(mode=context.GRAPH_MODE, jit_level='O0')

    class TestIfBlock(nn.Cell):
        def __init__(self):
            super(TestIfBlock, self).__init__()
            self.y = Parameter(Tensor([5], dtype.float32))

        def construct(self, x):
            x = x + self.y
            x = x - 9
            return x

    class MyBlock(nn.Cell):
        @lazy_inline
        def __init__(self):
            super(MyBlock, self).__init__()
            self.block = TestIfBlock()
            self.block.recompute()

        def construct(self, x):
            x = self.block(x)
            return x

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.SequentialCell()
            for _ in range(3):
                b = MyBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            out = self.blocks(out)
            return out

    class Grad(nn.Cell):
        def __init__(self, net):
            super(Grad, self).__init__()
            self.grad = ops.GradOperation()
            self.net = net

        def construct(self, x):
            grad_net = self.grad(self.net)
            return grad_net(x)

    x = Tensor([10.0], dtype.float32)
    net = Net()
    grad_net = Grad(net)
    grad = grad_net(x)
    assert grad == 1
