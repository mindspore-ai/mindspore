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

import pytest
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore import context, Tensor
from mindspore.ops import DataType, CustomRegOp
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
def test_custom_arg_pool_2d_aclnn(context_mode):
    """
    Feature: Custom op testcase
    Description: test case for AvgPool2D by custom
    Expectation: the result match with numpy result
    """

    class CustomNet(Cell):
        def __init__(self, func):
            super(CustomNet, self).__init__()
            reg_info = CustomRegOp("aclnnAvgPool2d") \
                .input(0, "x", "required") \
                .attr("kernel_size", "required", "listInt") \
                .attr("stride", "required", "listInt") \
                .attr("padding", "required", "listInt") \
                .attr("ceil_mode", "required", "bool") \
                .attr("count_include_pad", "required", "bool") \
                .attr("divisor_override", "required", "int") \
                .attr("cube_math_type", "required", "bool") \
                .output(0, "output", "required") \
                .dtype_format(DataType.F16_Default, DataType.F16_Default) \
                .target("Ascend") \
                .get_op_info()
            self.kernel_size = (2, 2)
            self.stride = (2, 2)
            self.padding = (1, 1)
            self.ceil_mode = False
            self.count_include_pad = True
            self.divisor_override = 0
            self.cube_math_type = False
            self.custom_avg_pool = ops.Custom(func, None,
                                              lambda x, kernel_size, stride, padding, ceil_mode, count_include_pad,
                                                     divisor_override, cube_math_type: x,
                                              func_type="aot",
                                              bprop=None, reg_info=reg_info)
            self.custom_avg_pool.add_prim_attr("kernel_size", self.kernel_size)
            self.custom_avg_pool.add_prim_attr("stride", self.stride)
            self.custom_avg_pool.add_prim_attr("padding", self.padding)

        def construct(self, x):
            res = self.custom_avg_pool(x, self.kernel_size, self.stride, self.padding, self.ceil_mode,
                                       self.count_include_pad, self.divisor_override, self.cube_math_type)
            return res

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    image = Tensor(np.array([[[4.1702e-1, 7.2032e-1, 1.1437e-4, 3.0223e-1],
                              [1.4676e-1, 9.2339e-2, 1.8626e-1, 3.4556e-1],
                              [3.9677e-1, 5.3882e-1, 4.1919e-1, 6.8522e-1],
                              [2.0445e-1, 8.7812e-1, 2.7338e-2, 6.7047e-1]]]).astype(np.float32))
    net = CustomNet("./infer_file/custom_cpp_infer.cc:aclnnAvgPool2d")
    output = net(image)

    expected = np.array([[[0.1043, 0.1801, 0.0756],
                          [0.1359, 0.3092, 0.2577],
                          [0.0511, 0.2264, 0.1676]]]).astype(np.float32)

    assert np.allclose(output.asnumpy(), expected, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
def test_custom_arg_pool_2d_py_infer_aclnn(context_mode):
    """
    Feature: Custom op testcase
    Description: test case for AvgPool2D by custom
    Expectation: the result match with numpy result
    """

    class CustomNet(Cell):
        def __init__(self, func):
            super(CustomNet, self).__init__()
            reg_info = CustomRegOp("aclnnAvgPool2d") \
                .input(0, "x", "required") \
                .attr("kernel_size", "required", "listInt") \
                .attr("stride", "required", "listInt") \
                .attr("padding", "required", "listInt") \
                .attr("ceil_mode", "required", "bool") \
                .attr("count_include_pad", "required", "bool") \
                .attr("divisor_override", "required", "int") \
                .attr("cube_math_type", "required", "bool") \
                .output(0, "output", "required") \
                .dtype_format(DataType.F16_Default, DataType.F16_Default) \
                .target("Ascend") \
                .get_op_info()
            self.kernel_size = (2, 2)
            self.stride = (2, 2)
            self.padding = (1, 1)
            self.ceil_mode = False
            self.count_include_pad = True
            self.divisor_override = 0
            self.cube_math_type = False

            def infer_shape(x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override,
                            cube_math_type):
                out = []
                out.append(x[0])
                h_out = (x[1] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
                w_out = (x[2] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
                out.append(h_out)
                out.append(w_out)
                return out

            self.custom_avg_pool = ops.Custom(func, infer_shape,
                                              lambda x, kernel_size, stride, padding, ceil_mode, count_include_pad,
                                                     divisor_override, cube_math_type: x,
                                              func_type="aot",
                                              bprop=None, reg_info=reg_info)


        def construct(self, x):
            res = self.custom_avg_pool(x, self.kernel_size, self.stride, self.padding, self.ceil_mode,
                                       self.count_include_pad, self.divisor_override, self.cube_math_type)
            return res

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    image = Tensor(np.array([[[4.1702e-1, 7.2032e-1, 1.1437e-4, 3.0223e-1],
                              [1.4676e-1, 9.2339e-2, 1.8626e-1, 3.4556e-1],
                              [3.9677e-1, 5.3882e-1, 4.1919e-1, 6.8522e-1],
                              [2.0445e-1, 8.7812e-1, 2.7338e-2, 6.7047e-1]]]).astype(np.float32))
    net = CustomNet("aclnnAvgPool2d")
    output = net(image)

    expected = np.array([[[0.1043, 0.1801, 0.0756],
                          [0.1359, 0.3092, 0.2577],
                          [0.0511, 0.2264, 0.1676]]]).astype(np.float32)

    assert np.allclose(output.asnumpy(), expected, rtol=1e-4, atol=1e-4)
