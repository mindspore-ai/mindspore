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
""" test_custom_multi_type.py """
import pytest
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
from mindspore import ops
from mindspore import context, Tensor
from mindspore.ops import DataType, CustomRegOp
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, dim=None, keepdim=False):
    return np.argmin(x, axis=dim)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
def test_custom_add(context_mode):
    """
    Feature: custom op multi type.
    Description: test function add forward.
    Expectation: expect correct result.
    """

    class CustomNet(Cell):
        def __init__(self):
            super(CustomNet, self).__init__()
            aclnn_reg_info = CustomRegOp("aclnnAdd") \
                .input(0, "self", "required") \
                .input(1, "other", "required") \
                .attr("alpha", "required", "int") \
                .output(0, "out", "required") \
                .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
                .target("Ascend") \
                .get_op_info()

            self.custom_add = ops.Custom("aclnnAdd",
                                         lambda self, other, alpha: self, lambda self, other, alpha: self,
                                         func_type="aot", bprop=None,
                                         reg_info=aclnn_reg_info)

        def construct(self, x, y):
            res = self.custom_add(x, y, 1)
            return res

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})
    x = generate_random_input((2, 3, 4), np.float32)
    y = generate_random_input((2, 3, 4), np.float32)
    net = CustomNet()
    output = net(ms.Tensor(x), ms.Tensor(y))
    expect = x + y
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
def test_custom_argmin(context_mode):
    """
    Feature: custom op multi type.
    Description: test function argmin forward.
    Expectation: expect correct result.
    """

    class CustomNet(Cell):
        def __init__(self, dim, keep_dim):
            super(CustomNet, self).__init__()
            aclnn_reg_info = CustomRegOp("aclnnArgMin") \
                .input(0, "x", "required") \
                .attr("dim", "required", "int") \
                .attr("keep_dim", "required", "bool") \
                .output(0, "z", "required") \
                .dtype_format(DataType.F16_Default, DataType.F16_Default) \
                .target("Ascend") \
                .get_op_info()
            self.dim = dim
            self.keep_dim = keep_dim
            self.custom_argmin = ops.Custom("aclnnArgMin",
                                            lambda x, dim, keep_dim: (x[1], x[2], x[3]), ms.dtype.int64,
                                            func_type="aot", bprop=None,
                                            reg_info=aclnn_reg_info)

        def construct(self, x):
            res = self.custom_argmin(x, self.dim, self.keep_dim)
            return res

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})
    x = generate_random_input((2, 3, 4, 5), np.float32)
    dim = 0
    keep_dim = False

    net = CustomNet(dim, keep_dim)
    output = net(ms.Tensor(x))
    expect = generate_expect_forward_output(x, dim, keep_dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
def test_custom_mul_aclnn(context_mode):
    """
    Feature: Custom op testcase
    Description: test case for mul by custom
    Expectation: the result match with numpy result
    """

    class CustomNet(Cell):
        def __init__(self, func, out_shape, bprop):
            super(CustomNet, self).__init__()
            self.custom_mul = ops.Custom(func, out_shape, lambda x, _: x, func_type="aot", bprop=bprop,
                                         reg_info=None)
            self.add = P.Add()
            self.sub = P.Sub()

        def construct(self, x, y, z):
            res = self.add(x, y)
            res = self.custom_mul(res, y)
            res = self.sub(res, z)
            return res

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    x = np.ones([8, 2048]).astype(np.float16)
    y = np.ones([8, 2048]).astype(np.float16)
    z = np.random.rand(8, 2048).astype(np.float16)
    net = CustomNet("./infer_file/custom_callback.cc:aclnnMul", lambda x, _: x, None)
    expect_out = (x + y) * y - z
    out = net(Tensor(x), Tensor(y), Tensor(z))
    assert np.allclose(out.asnumpy(), expect_out, 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
def test_custom_batch_norm_aclnn(context_mode):
    """
    Feature: Custom op testcase
    Description: test case for batch norm by custom
    Expectation: the result match with numpy result
    """

    class CustomNet(Cell):
        def __init__(self, func):
            super(CustomNet, self).__init__()
            aclnn_reg_info = CustomRegOp("aclnnBatchNorm") \
                .input(0, "x", "required") \
                .input(1, "scale", "required") \
                .input(2, "bias", "required") \
                .input(3, "mean", "required") \
                .input(4, "var", "required") \
                .attr("training", "required", "bool") \
                .attr("momentum", "required", "float") \
                .attr("eps", "required", "float") \
                .output(0, "output", "required") \
                .output(1, "saved_mean", "required") \
                .output(2, "saved_variance", "required") \
                .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                              DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
                .target("Ascend") \
                .get_op_info()

            self.custom_batch_norm = ops.Custom(func, lambda x, scale, bias, mean, var, training, momentum, eps: (
                x, scale, scale), lambda x, scale, bias, mean, var, training, momentum, eps: (x, x, x), func_type="aot",
                                                bprop=None, reg_info=aclnn_reg_info)
            # pylint: disable=protected-access
            self.custom_batch_norm._generate_get_worspace_size_func_by_types(
                "const aclTensor* input, const aclTensor* weight,const aclTensor * bias,"
                "aclTensor * runningMean,aclTensor * runningVar, bool training, float momentum, "
                "float eps,aclTensor * output, aclTensor * saveMean, aclTensor * saveInvstd,"
                "uint64_t * workspaceSize, aclOpExecutor ** executor")
            self.training = False
            self.momentum = 0.1
            self.eps = 1e-5

        def construct(self, x, scale, bias, mean, var):
            res = self.custom_batch_norm(x, scale, bias, mean, var, self.training, self.momentum, self.eps)
            return res

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    x = Tensor((3 * np.ones(16)).reshape(2, 2, 1, 4).astype(np.float32))
    scale = Tensor(np.ones(2).astype(np.float32))
    bias = Tensor(np.ones(2).astype(np.float32))
    mean = Tensor(np.ones(2).astype(np.float32))
    variance = Tensor(np.ones(2).astype(np.float32))

    expect = np.array([2.99999]).repeat(16, axis=0).astype(np.float32).reshape((2, 2, 1, 4))
    net = CustomNet("aclnnBatchNorm")
    output = net(x, scale, bias, mean, variance)[0]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
def test_custom_batch_norm_double_aclnn(context_mode):
    """
    Feature: Custom op testcase
    Description: test case for batch norm by custom
    Expectation: the result match with numpy result
    """

    class CustomNet(Cell):
        def __init__(self, func):
            super(CustomNet, self).__init__()
            aclnn_reg_info = CustomRegOp("aclnnBatchNorm") \
                .input(0, "x", "required") \
                .input(1, "scale", "required") \
                .input(2, "bias", "required") \
                .input(3, "mean", "required") \
                .input(4, "var", "required") \
                .attr("training", "required", "bool") \
                .attr("momentum", "required", "double") \
                .attr("eps", "required", "double") \
                .output(0, "output", "required") \
                .output(1, "saved_mean", "required") \
                .output(2, "saved_variance", "required") \
                .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                              DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
                .target("Ascend") \
                .get_op_info()

            self.custom_batch_norm = ops.Custom(func, lambda x, scale, bias, mean, var, training, momentum, eps: (
                x, scale, scale), lambda x, scale, bias, mean, var, training, momentum, eps: (x, x, x), func_type="aot",
                                                bprop=None, reg_info=aclnn_reg_info)
            self.training = False
            self.momentum = 0.1
            self.eps = 1e-5

        def construct(self, x, scale, bias, mean, var):
            res = self.custom_batch_norm(x, scale, bias, mean, var, self.training, self.momentum, self.eps)
            return res

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    x = Tensor((3 * np.ones(16)).reshape(2, 2, 1, 4).astype(np.float32))
    scale = Tensor(np.ones(2).astype(np.float32))
    bias = Tensor(np.ones(2).astype(np.float32))
    mean = Tensor(np.ones(2).astype(np.float32))
    variance = Tensor(np.ones(2).astype(np.float32))

    expect = np.array([2.99999]).repeat(16, axis=0).astype(np.float32).reshape((2, 2, 1, 4))
    net = CustomNet("aclnnBatchNorm")
    output = net(x, scale, bias, mean, variance)[0]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=12.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
def test_custom_cast_aclnn(context_mode):
    """
    Feature: Custom op testcase
    Description: test case for Cast by custom
    Expectation: the result match with numpy result
    """

    class CustomNet(Cell):
        def __init__(self, func, out_dtype):
            super(CustomNet, self).__init__()

            self.custom_cast = ops.Custom(func, lambda x, dst_type: x,
                                          out_dtype,
                                          func_type="aot",
                                          bprop=None, reg_info=None)

        def construct(self, x, dst_type):
            res = self.custom_cast(x, dst_type)
            return res

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    x = np.random.randn(1280, 1280).astype(np.float16)
    dtype = mstype.float32
    net = CustomNet("aclnnCast", dtype)
    output = net(ms.Tensor(x), dtype)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (1280, 1280)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
def test_custom_all_finite_aclnn(context_mode):
    """
    Feature: Custom op testcase
    Description: test case for AllFinite by custom
    Expectation: the result match with numpy result
    """

    class CustomNet(Cell):
        def __init__(self, func):
            super(CustomNet, self).__init__()
            self.custom_all_finite = ops.Custom(func, lambda x: [1],
                                                mstype.bool_,
                                                func_type="aot",
                                                bprop=None, reg_info=None)

        def construct(self, x):
            res = self.custom_all_finite(x)
            return res

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    x1 = Tensor(np.full([128, 128], -np.inf, np.float16))
    x2 = Tensor(np.full([12960, 65], 10, np.float16))
    net = CustomNet("aclnnAllFinite")
    out1 = net(x1)
    out2 = net(x2)
    assert out1.asnumpy()
    assert not out2.asnumpy()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level4', card_mark='onecard', essential_mark='essential')
def test_custom_topk():
    """
  Feature: Custom op testcase
  Description: test case for topk by custom
  Expectation: the result match with numpy result
  """

    class MoeSoftMaxTopkNet(Cell):
        def __init__(self, func, out_shape):
            super(MoeSoftMaxTopkNet, self).__init__()
            self.k = 2
            self.moe_softmax_topk_custom = ops.Custom(func=func, out_shape=out_shape,
                                                      out_dtype=[mstype.float32, mstype.int32], func_type="aot",
                                                      bprop=None, reg_info=None)
            # Used for infer shape
            self.moe_softmax_topk_custom.add_prim_attr("attr_k", self.k)

        def construct(self, x):
            res = self.moe_softmax_topk_custom(x, self.k)
            return res

    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graph",
                   pynative_synchronize=False)
    ms.set_context(jit_config={"jit_level": "O1"})

    input_x = ms.Tensor(np.ones([1024, 16]), ms.float32)
    net = MoeSoftMaxTopkNet("./infer_file/custom_topk.cc:MoeSoftMaxTopk", None)
    output = net(input_x)
    print(output)
