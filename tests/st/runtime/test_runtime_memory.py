# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
test custom memory pool
"""
import pytest
import numpy as np
from tests.mark_utils import arg_mark
from tests.device_utils import set_device
from mindspore import context, nn, Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype
import mindspore.runtime as rt
import mindspore as ms


class SparseApplyFtrlNet(nn.Cell):
    """simple net"""
    def __init__(self, var, accum, linear, lr=0.001, l1=0.0, l2=0.0, lr_power=-0.5):
        super().__init__()
        self.sparse_apply_ftrl = P.SparseApplyFtrl(lr=lr, l1=l1, l2=l2, lr_power=lr_power)
        self.var = Parameter(var, name="var")
        self.accum = Parameter(accum, name="accum")
        self.linear = Parameter(linear, name="linear")

    def construct(self, grad, indices):
        out = self.sparse_apply_ftrl(self.var, self.accum, self.linear, grad, indices)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_sparse_apply_ftrl_with_memory_optimize():
    """
    Feature: Integration of dynamic and static memory.
    Description: Test the scene of output ref node.
    Expectation: The result meet expectation.
    """
    context.set_context(mode=context.GRAPH_MODE)
    rt.set_memory(optimize_level="O1")

    grad_np = np.ones([3, 3, 3])
    indice_np = [0, 1, 2]
    var_np = np.ones([3, 3, 3])
    accum_np = np.ones([3, 3, 3])
    linear_np = np.ones([3, 3, 3])

    # test1: var/accum/linear/gradient are float32 and indices is int32.
    gradient = Tensor(grad_np, dtype=mstype.float32)
    indices = Tensor(indice_np, dtype=mstype.int32)
    var = Tensor(var_np, dtype=mstype.float32)
    accum = Tensor(accum_np, dtype=mstype.float32)
    linear = Tensor(linear_np, dtype=mstype.float32)
    sparse_apply_ftrl = SparseApplyFtrlNet(var, accum, linear)
    out = sparse_apply_ftrl(gradient, indices)
    expect_var = np.array([[[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]],
                           [[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]],
                           [[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]]]).astype(np.float32)
    assert np.all(out[0].asnumpy() == expect_var)

    # test2: var/accum/linear/gradient are float16 and indices is int32.
    gradient = Tensor(grad_np, dtype=mstype.float16)
    indices = Tensor(indice_np, dtype=mstype.int32)
    var = Tensor(var_np, dtype=mstype.float16)
    accum = Tensor(accum_np, dtype=mstype.float16)
    linear = Tensor(linear_np, dtype=mstype.float16)
    sparse_apply_ftrl = SparseApplyFtrlNet(var, accum, linear)
    out = sparse_apply_ftrl(gradient, indices)
    expect_var = np.array([[[0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915]],
                           [[0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915]],
                           [[0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915]]]).astype(np.float16)
    assert np.all(out[0].asnumpy() == expect_var)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_runtime_use_mem_pool_error():
    """
    Feature: runtime memory api use_mem_pool.
    Description: Test runtime.use_mem_pool api when so is error.
    Expectation: runtime.use_mem_pool api performs as expected.
    """
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)
    so_path = "/data/libfake_custom_allocator.so"
    with pytest.raises(OSError):
        ms.runtime.PluggableAllocator(so_path, "Alloc", "Free")

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_runtime_use_mem_pool():
    """
    Feature: runtime memory api use_mem_pool.
    Description: Test runtime.use_mem_pool api.
    Expectation: runtime.use_mem_pool api performs as expected.
    """
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)
    so_path = "/home/workspace/mindspore_dataset/custom_so/libascend_custom_allocator.so"
    shape = (1024, 1024)
    allocator = ms.runtime.PluggableAllocator(so_path, "Alloc", "Free")
    mem_pool = ms.runtime.MemPool(allocator)
    x = Tensor(np.zeros(shape), ms.float32)
    with ms.runtime.use_mem_pool(mem_pool):
        y = Tensor(np.zeros(shape), ms.float32)
        y += 1

        z = Tensor(np.zeros(shape), ms.float32)
        z += 2

    output = x + y + z
    expected = Tensor(np.ones(shape) * 3, ms.float32)
    assert np.allclose(output.asnumpy(), expected)
