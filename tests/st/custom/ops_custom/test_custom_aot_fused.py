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
import platform
from tests.mark_utils import arg_mark
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp


class ReduceDynNet(Cell):
    def __init__(self, func_path, out_types, axis, keep_dim):
        super(ReduceDynNet, self).__init__()
        reduce_cpu_info = CustomRegOp("reduce_kernel_cpu") \
            .input(0, "x1") \
            .input(0, "x2") \
            .output(0, "y") \
            .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None) \
            .attr("axis", "required", "all", value=axis) \
            .attr("keep_dim", "required", "all", value=keep_dim) \
            .target("CPU") \
            .get_op_info()
        self.program = ops.Custom(func_path + "./kernel.cc:CustomKernel", None,
                                  out_types, "aot", reg_info=reduce_cpu_info)

    def construct(self, x, y):
        return self.program(x, y)


def aot_fused_kernel():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    func_path = dir_path + "/aot_test_files/"

    shape = (4, 5)
    axis = 1
    keep_dim = False

    input_x = np.ones(shape).astype(np.float32)
    input_y = np.ones(shape).astype(np.float32)
    expected = np.ones((4,)).astype(np.float32) * 10

    test = ReduceDynNet(func_path, mstype.float32, axis, keep_dim)
    dyn_x = Tensor(shape=[4, None], dtype=mstype.float32)
    # set the net to dynamic shape
    test.set_inputs(dyn_x, dyn_x)
    output = test(Tensor(input_x), Tensor(input_y))
    assert np.allclose(expected, output.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_aot_fused_kernel():
    """
    Feature: custom aot operator, multiple inputs, single output, CPU, GRAPH_MODE
    Description: pre-compile xxx.cc to xxx.so, custom operator launches xxx.so
    Expectation: nn result matches numpy result
    """
    sys = platform.system()
    if sys.lower() in {"windows", "darwin"}:
        pass
    else:
        aot_fused_kernel()
