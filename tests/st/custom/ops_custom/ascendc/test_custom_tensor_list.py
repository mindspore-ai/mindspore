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

import os
import shutil
import tempfile
import pytest
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore import context
from mindspore.ops import DataType, CustomRegOp
import mindspore.common.dtype as mstype
from tests.mark_utils import arg_mark


def GenerateConcatInput(shape_param):
    np_inputs = []
    tensor_inputs = []
    for shape in shape_param:
        np_input = np.random.rand(*shape).astype(np.float32)
        np_inputs.append(np_input)
        t = ms.Tensor(np_input)
        tensor_inputs.append(t)
    return np_inputs, tensor_inputs


def concat_out_shape(x, y):
    out = x[0].copy()
    out[1] = x[0][1] + x[1][1]
    return out


class CustomNetConcat(Cell):
    def __init__(self, func, out_shape, out_dtype):
        super(CustomNetConcat, self).__init__()
        reg_info = CustomRegOp("aclnnCat") \
            .input(0, "x", "dynamic") \
            .attr("dim", "required", "int") \
            .output(0, "output", "required") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default) \
            .target("Ascend") \
            .get_op_info()

        self.concat = ops.Custom(func, out_shape, out_dtype, func_type="aot", reg_info=reg_info)

    def bprop(self, x1, x2, axis, out, dout):
        dx1 = ops.slice(dout, (0, 0), (2, 2))
        dx2 = ops.slice(dout, (0, 2), (2, 3))
        return dx1, dx2, 0

    def construct(self, x1, x2, axis):
        res = self.concat((x1, x2), axis)
        return res


class CustomNetSplitTensor(Cell):
    def __init__(self, func, out_shape, out_dtype):
        super(CustomNetSplitTensor, self).__init__()
        reg_info = CustomRegOp("aclnnSplitTensor") \
            .input(0, "x", "required") \
            .attr("split_size", "required", "int") \
            .attr("dim", "required", "int") \
            .output(0, "output", "dynamic") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default) \
            .target("Ascend") \
            .get_op_info()

        self.split = ops.Custom(func, out_shape, out_dtype, func_type="aot", reg_info=reg_info)

    def construct(self, x1, split_size, dim):
        res = self.split(x1, split_size, dim)
        return res


@pytest.fixture(scope="function", autouse=True)
def compiler_cache():
    temp_dir = tempfile.mkdtemp(prefix="ms_compiler_cache_")
    os.environ["MS_COMPILER_CACHE_PATH"] = temp_dir
    print(f"[SETUP] Set MS_COMPILER_CACHE_PATH to {temp_dir}")
    yield
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.environ.pop("MS_COMPILER_CACHE_PATH", None)
    print(f"[TEARDOWN] Removed temp dir and unset MS_COMPILER_CACHE_PATH")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
@pytest.mark.parametrize("params", [(((2, 2), (2, 3)), 1), (((3, 2, 3), (3, 3, 3)), -2)])
def test_custom_concat_py_infer(context_mode, params):
    """
    Feature: Custom op testcase
    Description: test case for Concat by custom
    Expectation: the result match with numpy result
    """

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    shape_param, axis = params
    np_inputs, tensor_inputs = GenerateConcatInput(shape_param)
    expected_out = np.concatenate(np_inputs, axis)

    net = CustomNetConcat("aclnnCat", concat_out_shape, lambda x, _: x[0])
    out = net(tensor_inputs[0], tensor_inputs[1], axis)
    assert np.allclose(out.asnumpy(), expected_out)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
@pytest.mark.parametrize("params", [(((2, 2), (2, 3)), 1), (((3, 2, 3), (3, 3, 3)), -2)])
def test_custom_concat_aclnn_cpp_infer(context_mode, params):
    """
    Feature: Custom op testcase
    Description: test case for Concat by custom
    Expectation: the result match with numpy result
    """

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    shape_param, axis = params
    np_inputs, tensor_inputs = GenerateConcatInput(shape_param)
    expected_out = np.concatenate(np_inputs, axis)

    net = CustomNetConcat("./infer_file/custom_cpp_infer.cc:aclnnCat", None, None)
    out = net(tensor_inputs[0], tensor_inputs[1], axis)
    assert np.allclose(out.asnumpy(), expected_out)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
@pytest.mark.parametrize("params", [(((2, 2), (2, 3)), 1), (((3, 2, 3), (3, 3, 3)), -2)])
def test_custom_concat_aclnn_cpp_infer_shape_py_infer_type(context_mode, params):
    """
    Feature: Custom op testcase
    Description: test case for Concat by custom
    Expectation: the result match with numpy result
    """

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    shape_param, axis = params
    np_inputs, tensor_inputs = GenerateConcatInput(shape_param)
    expected_out = np.concatenate(np_inputs, axis)

    net = CustomNetConcat("./infer_file/custom_cpp_infer.cc:aclnnCat", None, lambda x, _: x[0])
    out = net(tensor_inputs[0], tensor_inputs[1], axis)
    assert np.allclose(out.asnumpy(), expected_out)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
@pytest.mark.parametrize("params", [(((2, 2), (2, 3)), 1), (((3, 2, 3), (3, 3, 3)), -2)])
def test_custom_concat_aclnn_cpp_infer_type_py_infer_shape(context_mode, params):
    """
    Feature: Custom op testcase
    Description: test case for Concat by custom
    Expectation: the result match with numpy result
    """

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    shape_param, axis = params
    np_inputs, tensor_inputs = GenerateConcatInput(shape_param)
    expected_out = np.concatenate(np_inputs, axis)

    net = CustomNetConcat("./infer_file/custom_cpp_infer.cc:aclnnCat", concat_out_shape, None)
    out = net(tensor_inputs[0], tensor_inputs[1], axis)
    assert np.allclose(out.asnumpy(), expected_out)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
@pytest.mark.parametrize("params", [(((2, 2), (2, 3)), 1)])
def test_custom_concat_aclnn_bprop(context_mode, params):
    """
    Feature: Custom op testcase
    Description: test case for Concat by custom
    Expectation: the result match with numpy result
    """

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    shape_param, axis = params
    _, tensor_inputs = GenerateConcatInput(shape_param)

    net = CustomNetConcat("./infer_file/custom_cpp_infer.cc:aclnnCat", None, None)
    grads = ops.GradOperation(get_all=True, sens_param=False)(net)(tensor_inputs[0], tensor_inputs[1], axis)
    expect_grad1 = np.ones(shape_param[0]).astype(np.float32)
    expect_grad2 = np.ones(shape_param[1]).astype(np.float32)
    assert np.allclose(grads[0].asnumpy(), expect_grad1)
    assert np.allclose(grads[1].asnumpy(), expect_grad2)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
@pytest.mark.parametrize("params", [(((2, 2), (2, 3)), 1)])
def test_custom_concat_aclnn_dynamic_shape(context_mode, params):
    """
    Feature: Custom op testcase
    Description: test case for Concat by custom
    Expectation: the result match with numpy result
    """

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    shape_param, axis = params
    np_inputs, tensor_inputs = GenerateConcatInput(shape_param)

    net = CustomNetConcat("./infer_file/custom_cpp_infer.cc:aclnnCat", None, None)
    dyn_x = ms.Tensor(shape=(2, None), dtype=mstype.float32)
    net.set_inputs(dyn_x, tensor_inputs[1], axis)

    expected_out = np.concatenate(np_inputs, axis)
    out = net(tensor_inputs[0], tensor_inputs[1], axis)
    assert np.allclose(out.asnumpy(), expected_out)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
def test_custom_split_aclnn(context_mode):
    """
    Feature: Custom op testcase
    Description: test case for SplitTensor by custom
    Expectation: the result match with numpy result
    """

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    net = CustomNetSplitTensor("aclnnSplitTensor", lambda x, split_size, dim: [[x[0] // 2, x[1]], [x[0] // 2, x[1]]],
                               lambda x, split_size, dim: [x, x])
    a = np.array(np.arange(20).reshape((10, 2)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    split_size_or_sections = 5
    out = net(x, split_size_or_sections, 0)
    expect = [np.array(np.arange(10).reshape((5, 2)), dtype=np.float32),
              np.array(np.arange(10, 20).reshape((5, 2)), dtype=np.float32)]
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)
