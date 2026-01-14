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
"""test kernel select in ascend"""
import os
import pytest
import numpy as np
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import Tensor, jit
from mindspore import ops, mint
from mindspore.nn import Cell

class Net(Cell):
    def __init__(self):
        super().__init__()
        self.op1 = mint.sin
        self.op2 = mint.cos
        self.op3 = ops.auto_generate.matmul_add_
    @jit
    def construct(self, x, weight, c):
        x = self.op1(x)
        x = self.op2(x)
        res = self.op3(x, weight, c)
        return res

def generate_inputs(m, k, n, batch=None, dtype=ms.float16):
    if batch is not None:
        x_shape = (batch, k, m)
        w_shape = (batch, k, n)
        c_shape = (batch, m, n)
    else:
        x_shape = (k, m)
        w_shape = (k, n)
        c_shape = (m, n)
    x = Tensor(np.random.randn(*x_shape), dtype=dtype)
    w = Tensor(np.random.randn(*w_shape), dtype=dtype)
    c = Tensor(np.random.randn(*c_shape), dtype=ms.float32)
    return x, w, c


def test_kernel_select_num():
    """
    Feature: Ascend
    Description: test select kernel
    Expectation: expect correct result.
    """
    x, weight, c = generate_inputs(10, 20, 8)
    net = Net()
    net(x, weight, c)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_kernel_select():
    """
    Feature: Ascend
    Description: test select kernel
    Expectation: expect correct result.
    """
    os.environ["VLOG_v"] = "13000"
    os.system("pytest -sv test_kernel_select.py::test_kernel_select_num > log_kernel_select.txt 2>&1")
    res_aclnn = os.popen("grep 'select aclnn' log_kernel_select.txt | wc -l").read()
    assert int(res_aclnn) == 2
    res_atb = os.popen("grep 'select atb' log_kernel_select.txt | wc -l").read()
    assert int(res_atb) == 1
    os.system("rm -rf log_kernel_select.txt")
    del os.environ["VLOG_v"]

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_kernel_select_failed():
    """
    Feature: kernel select
    Description: test select kernel
    Expectation: expect correct result
    """
    @jit
    def kernel_select_failed(images, size):
        resize_op = ops.ResizeBicubic(False, False)
        return resize_op(images, size)
    images = Tensor(shape=[1, 1, 2, 2], dtype=ms.int32)
    size = Tensor([1, 4], ms.int32)
    with pytest.raises(TypeError):
        kernel_select_failed(images, size)
