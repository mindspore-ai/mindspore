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
''' test ops expander with scalar '''
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
from mindspore.common import Tensor, dtype
import mindspore.ops.operations as P
from mindspore.ops.operations import _grad_ops as G
from mindspore import ops, nn
from mindspore.ops.auto_generate.gen_ops_prim import AsinExt, MeanExt
from tests.mark_utils import arg_mark


class OpsCell(Cell):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def construct(self, *args):
        return self.op(*args)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2048.01)
def test_unsortedsegmentsum_grad():
    """
    Feature: UnsortedSegmentSum
    Description: Verify the result of UnsortedSegmentSum with grad.
    Expectation: success
    """
    input_x = Tensor([1, 2, 3, 4], dtype.float32)
    segment_ids = Tensor([0, 0, 1, 2], dtype.int32)
    num_segments = 4

    net = OpsCell(ops.UnsortedSegmentSum())
    net_me = ops.GradOperation(sens_param=False)(net)
    output = net_me(input_x, segment_ids, num_segments)
    expect = [1, 1, 1, 1]
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2048.0)
def test_unsortedsegmentmin_grad():
    """
    Feature: UnsortedSegmentMin
    Description: Verify the result of UnsortedSegmentMin with grad.
    Expectation: success
    """
    input_x = Tensor([1, 2, 3, 4], dtype.int32)
    segment_ids = Tensor([0, 0, 1, 2], dtype.int32)
    num_segments = 4
    net = OpsCell(P.UnsortedSegmentMin())
    net_me = ops.GradOperation(sens_param=False)(net)
    output = net_me(input_x, segment_ids, num_segments)
    expect = [1, 0, 1, 1]
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_unsortedsegmentmax_grad():
    """
    Feature: UnsortedSegmentMax
    Description: Verify the result of UnsortedSegmentMax with grad.
    Expectation: success
    """
    input_x = Tensor([1, 2, 3, 4], dtype.int32)
    segment_ids = Tensor([0, 0, 1, 2], dtype.int32)
    num_segments = 4
    net = OpsCell(P.UnsortedSegmentMax())
    net_me = ops.GradOperation(sens_param=False)(net)
    output = net_me(input_x, segment_ids, num_segments)
    expect = [0, 1, 1, 1]
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=18.00)
def test_Gatherdgrad_grad():
    """
    Feature: GatherD
    Description: Verify the result of GatherD with grad.
    Expectation: success
    """
    x = Tensor(np.array([1, 2, 3, 4]), dtype.float32)
    dim = 0
    index = Tensor(np.array([0, 0, 1, 2]), dtype.int32)
    expect = np.array([2, 1, 1, 0], np.float32)
    net = OpsCell(P.GatherD())
    grad_net = ops.GradOperation(get_all=True)(net)
    output = grad_net(x, dim, index)
    assert (output[0].asnumpy() == expect).all()


@arg_mark(plat_marks=['cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_embeddinglookup_grad():
    """
    Feature: EmbeddingLookup
    Description: Verify the result of EmbeddingLookup with grad.
    Expectation: success
    """
    net = OpsCell(nn.EmbeddingLookup(4, 2, dtype=dtype.float32))
    x = Tensor(np.array([[1, 0], [3, 2]]), dtype.int32)
    grad_net = ops.GradOperation(get_all=True)(net)
    output = grad_net(x)
    expect = np.array([[0, 0], [0, 0]], np.float32)
    assert (output[0].asnumpy() == expect).all()


@arg_mark(plat_marks=['cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_affine_grid_grad():
    """
    Feature: gpu backend of operator AffineGrid with grad
    Description: special case when h or w = 1 and align_corners = True
    Expectation: success
    """
    n, c, h, w = 2, 2, 2, 1
    net = OpsCell(ops.AffineGrid(align_corners=True))
    t = Tensor(np.ones((2, 2, 3)), dtype.float32)
    output_size = (n, c, h, w)
    grad_net = ops.GradOperation(get_all=True)(net)
    output = grad_net(t, output_size)
    expected = np.array([[[[0, 0, 2]], [[0, 0, 2]]],
                         [[[0, 0, 2]], [[0, 0, 2]]]]).astype(np.float32)
    assert np.allclose(output[0].asnumpy(), expected, atol=0.001, rtol=0.001)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_ops_clip_by_norm_with_grad():
    """
    Feature: ops.clip_by_norm with grad
    Description: Verify the result of clip_by_norm with grad
    Expectation: success
    """
    x = ms.Tensor([[0.8, 0.1, 0.0],
                   [0.7, 0.4, 0.0],
                   [4.6, 0.2, 2.1]])
    net = OpsCell(ops.clip_by_norm)
    grad_net = ops.GradOperation(get_all=True)(net)
    out = grad_net(x, 1)
    expect_out = np.array([[0.141767010, 0.186396033, 0.192771614],
                           [0.148142591, 0.167269319, 0.192771614],
                           [-0.100504845, 0.180020466, 0.0588845462]])
    assert np.allclose(out[0].asnumpy(), expect_out)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_rgbtohsv_with_grad():
    """
    Feature: Rgbtohsv_with_grad
    Description: test rgbtohsv_with_grad
    Expectation: success
    """

    x = np.array([0.25, 0.5, 0.5]).astype(np.float16).reshape([1, 1, 1, 3])
    net = OpsCell(P.RGBToHSV())
    grad_net = ops.GradOperation(get_all=True)(net)
    output = grad_net(Tensor(x))
    expected = np.array([-2.0, 2.0, 0.66666687]).astype(
        np.float16).reshape([1, 1, 1, 3])
    assert np.allclose(output[0].asnumpy(), expected, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_silu_with_grad():
    """
    Feature: nn.SiLU with grad
    Description: test nn.SiLU with grad
    Expectation: success
    """
    np_x = np.ones(shape=[2, 3]).astype(np.float32)
    x = Tensor(np_x)
    net = OpsCell(nn.SiLU())
    grad_net = ops.GradOperation(get_all=True)(net)
    output = grad_net(Tensor(x))
    expect_out = np.ones(shape=[2, 3]).astype(np.float32) * 0.927670717
    assert np.allclose(output[0].asnumpy(), expect_out)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=34.00)
def test_igamma_grad():
    """
    Feature: Igamma with grad.
    Description: Verify igamma operator with grad.
    Expectation: No exception.
    """
    class IGammaTest(nn.Cell):
        def __init__(self):
            super().__init__()
            self.igamma = nn.IGamma()

        def construct(self, x, a):
            return self.igamma(a=a, x=x)

    x = 4.22
    a = 2.29
    net = IGammaTest()
    grad_net = ops.GradOperation(get_all=True)(net)
    output = grad_net(Tensor(x, dtype.float32), Tensor(a, dtype.float32))
    assert np.allclose(output[0].asnumpy(), 0.08120076)
    assert np.allclose(output[1].asnumpy(), -0.11669071)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_minimum_grad():
    """
    Feature: minimum with grad
    Description: test minimum with grad
    Expectation: success
    """
    input_x = Tensor([1, 2, 3, 4], dtype.int32)
    input_y = Tensor([0, 0, 5, 6], dtype.int32)
    net = OpsCell(P.Minimum())
    net_me = ops.GradOperation(get_all=True)(net)
    output = net_me(input_x, input_y)
    expect = [0, 0, 1, 1]
    assert (output[0].asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_maximum_grad():
    """
    Feature: maximum with grad
    Description: test maximum with grad
    Expectation: success
    """
    input_x = Tensor([1, 2, 3, 4], dtype.int32)
    input_y = Tensor([0, 0, 5, 6], dtype.int32)
    net = OpsCell(P.Maximum())
    net_me = ops.GradOperation(get_all=True)(net)
    output = net_me(input_x, input_y)
    expect = [1, 1, 0, 0]
    assert (output[0].asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_addcdiv_float32_with_grad():
    """
    Feature: Addcdiv with grad
    Description: Test addcdiv with grad
    Expectation: The results are as expected
    """
    input_data = Tensor(np.array([12]).astype(np.float32))
    x1 = Tensor(np.array([7]).astype(np.float32))
    x2 = Tensor(np.array([3]).astype(np.float32))
    value = Tensor(np.array([37]).astype(np.float32))
    net = OpsCell(P.Addcdiv())
    net_me = ops.GradOperation()(net)
    output = net_me(input_data, x1, x2, value)
    assert output.asnumpy() == 1


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_addcmul_float32_with_grad():
    """
    Feature: Addcmul with grad
    Description: Test Addcmul with grad
    Expectation: The results are as expected
    """
    input_data = Tensor(np.array([12]).astype(np.float32))
    x1 = Tensor(np.array([7]).astype(np.float32))
    x2 = Tensor(np.array([3]).astype(np.float32))
    value = Tensor(np.array([37]).astype(np.float32))
    net = OpsCell(P.Addcmul())
    net_me = ops.GradOperation()(net)
    output = net_me(input_data, x1, x2, value)
    assert output.asnumpy() == 1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=1024.01)
def test_asin_with_grad():
    """
    Feature: Asin_with_grad
    Description: test asin_with_grad
    Expectation: the result match expectation
    """
    np_array = np.array([-0.5, 0, 0.5], dtype=np.float32)
    input_x = Tensor(np_array)
    net = OpsCell(P.Asin())
    net_me = ops.GradOperation(get_all=True)(net)
    output = net_me(input_x)
    expect = np.array([1.15470052, 1.0, 1.15470052], dtype=np.float32)
    assert (output[0].asnumpy() == expect).all()
    net2 = OpsCell(AsinExt())
    net_me2 = ops.GradOperation(get_all=True)(net2)
    output2 = net_me2(input_x)
    assert (output2[0].asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_asinh_grad_with_grad():
    """
    Feature: Asinhgrad_with_grad
    Description: test Asinhgrad_with_grad
    Expectation: the result match expectation
    """
    out = np.array([-0.5, 0, 0.5]).astype('float32')
    dy = np.array([1, 0, -1]).astype('float32')
    asinh_grad = OpsCell(G.AsinhGrad())
    net_me = ops.GradOperation(get_all=True)(asinh_grad)
    output = net_me(Tensor(out), Tensor(dy))
    expect1 = np.array([0.409814239, -0.0, 0.409814239], dtype=np.float32)
    expect2 = np.array([0.886818886, 1.0, 0.886818886], dtype=np.float32)
    assert np.allclose(output[0].asnumpy(), expect1)
    assert np.allclose(output[1].asnumpy(), expect2)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_acos_grad():
    """
    Feature: Acosgrad_with_grad
    Description: test Acosgrad_with_grad
    Expectation: the result match expectation
    """
    x = np.array([-0.5, 0, 0.5]).astype('float32')
    dy = np.array([1, 0, -1]).astype('float32')
    acos_grad = OpsCell(G.ACosGrad())
    net_me = ops.GradOperation(get_all=True)(acos_grad)
    output = net_me(Tensor(x), Tensor(dy))
    expect1 = np.array([0.769800365, -0.0, 0.769800365], dtype=np.float32)
    expect2 = np.array([-1.15470052, -1.0, -1.15470052], dtype=np.float32)
    assert np.allclose(output[0].asnumpy(), expect1)
    assert np.allclose(output[1].asnumpy(), expect2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_acosh_grad():
    """
    Feature: Acoshgrad_with_grad
    Description: test Acoshgrad_with_grad
    Expectation: the result match expectation
    """
    out = np.array([5, 4, 3]).astype('float32')
    dy = np.array([1, 1, -1]).astype('float32')
    acosh_grad = OpsCell(G.AcoshGrad())
    net_me = ops.GradOperation(get_all=True)(acosh_grad)
    output = net_me(Tensor(out), Tensor(dy))
    expect1 = np.array(
        [-0.01347773, -0.03666817, 0.10031768], dtype=np.float32)
    expect2 = np.array([0.01347637, 0.03666762, 0.10031758], dtype=np.float32)
    assert np.allclose(output[0].asnumpy(), expect1, 1e-3, 1e-3)
    assert np.allclose(output[1].asnumpy(), expect2, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_atan_grad_float():
    """
    Feature: Atangrad_with_grad
    Description: test Atangrad_with_grad
    Expectation: the result match expectation
    """
    x = np.array([-0.5, 0, 0.5]).astype(np.float32)
    dy = np.array([1, 0, -1]).astype(np.float32)
    atan_grad = OpsCell(G.AtanGrad())
    net_me = ops.GradOperation(get_all=True)(atan_grad)
    output = net_me(Tensor(x), Tensor(dy))
    expect1 = np.array([0.640000045, -0.0, 0.640000045], dtype=np.float32)
    expect2 = np.array([0.800000012, 1.0, 0.800000012], dtype=np.float32)
    assert np.allclose(output[0].asnumpy(), expect1)
    assert np.allclose(output[1].asnumpy(), expect2)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_log1p_with_grad():
    """
    Feature: log1p
    Description: test log1p with grad
    Expectation: the result match expectation
    """
    log1p = OpsCell(P.Log1p())
    x = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    x = Tensor(x, dtype=dtype.float32)
    net_me = ops.GradOperation(get_all=True)(log1p)
    output = net_me(x)
    expect = np.array([0.3333, 0.25, 0.2, 0.1666], dtype=np.float32)
    assert np.allclose(output[0].asnumpy(), expect, 1e-3, 1e-3)


class LogAddExpNet(nn.Cell):
    def construct(self, x1, x2):
        return x1.logaddexp(x2)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_tensor_logaddexp_with_grad():
    """
    Feature: tensor.logaddexp with grad
    Description: Verify the result of logaddexp with grad
    Expectation: success
    """
    x1 = ms.Tensor([-100, 1, 30], ms.float32)
    x2 = ms.Tensor([-1, -1, 3], ms.float32)
    net = LogAddExpNet()
    net_me = ops.GradOperation()(net)
    output = net_me(x1, x2)
    expect_output = np.array([0.0, 0.88079715, 1.0])
    assert np.allclose(output.asnumpy(), expect_output)


class LogAddExp2Net(nn.Cell):
    def construct(self, x1, x2):
        return x1.logaddexp2(x2)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_tensor_logaddexp2_with_grad():
    """
    Feature: tensor.logaddexp2 with grad
    Description: Verify the result of logaddexp2 with grad
    Expectation: success
    """
    x1 = ms.Tensor([-100, 1, 30], ms.float32)
    x2 = ms.Tensor([-1, -1, 3], ms.float32)
    net = LogAddExp2Net()
    net_me = ops.GradOperation()(net)
    output = net_me(x1, x2)
    expect_output = np.array([0.0, 0.8, 1.0])
    assert np.allclose(output.asnumpy(), expect_output, 1e-3, 1e-3)


class LogSumExpNet(nn.Cell):
    def construct(self, x, dim, keepdim=False):
        return x.logsumexp(dim, keepdim)


@arg_mark(plat_marks=['cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=1024.01)
def test_tensor_logsumexp_with_grad():
    """
    Feature: Tensor.logsumexp with grad
    Description: Verify the result of Tensor.logsumexp with grad
    Expectation: success
    """
    net = LogSumExpNet()
    x = Tensor(np.array([[1, 2], [3, 4]]), ms.float32)
    dim = 1
    keepdim = False
    net_me = ops.GradOperation()(net)
    output = net_me(x, dim, keepdim)
    expect_output = np.array(
        [[0.26894107, 0.7310589], [0.26894107, 0.7310589]])
    assert np.allclose(output.asnumpy(), expect_output, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_expm1_with_grad():
    """
    Feature: Test expm1 with grad.
    Description: Test expm1 with grad.
    Expectation: Success.
    """
    net = OpsCell(P.Expm1())
    x = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    x = Tensor(x, dtype=dtype.float32)
    net_me = ops.GradOperation(get_all=True)(net)
    output = net_me(x)
    expect_output = np.array([7.3890, 20.0855, 54.5981, 148.4131])
    assert np.allclose(output[0].asnumpy(), expect_output, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_exp2_with_grad():
    """
    Feature: Test exp2 with grad.
    Description: Test exp2 with grad.
    Expectation: Success.
    """
    net = OpsCell(ops.exp2)
    x = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    x = Tensor(x, dtype=dtype.float32)
    net_me = ops.GradOperation(get_all=True)(net)
    output = net_me(x)
    expect_output = np.array([2.7725, 5.5451, 11.0903, 22.1807])
    assert np.allclose(output[0].asnumpy(), expect_output, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_tan_grad():
    """
    Feature: Test tan with grad.
    Description: Test tan with grad.
    Expectation: Success.
    """
    np_array = np.array([-1, -0.5, 0, 0.5, 1]).astype('float32')
    input_x = Tensor(np_array)
    net = OpsCell(P.Tan())
    net_me = ops.GradOperation(get_all=True)(net)
    output = net_me(input_x)
    expect_output = np.array([3.4255, 1.2984, 1.0, 1.2984, 3.4255])
    assert np.allclose(output[0].asnumpy(), expect_output, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_sinc_grad():
    """
    Feature: Test sinc with grad.
    Description: Test sinc with grad.
    Expectation: Success.
    """
    np_array = np.array([-1, -0.5, 0, 0.5, 1]).astype('float32')
    input_x = Tensor(np_array)
    net = OpsCell(P.Sinc())
    net_me = ops.GradOperation(get_all=True)(net)
    output = net_me(input_x)
    expect_output = np.array([1.0, 1.2732, 0.0, -1.2732, -1.0])
    assert np.allclose(output[0].asnumpy(), expect_output, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_squared_difference_grad():
    """
    Feature: Test squared difference with grad.
    Description: Test squared difference with grad.
    Expectation: Success.
    """
    input_x = Tensor(np.array([1, 2]).astype('float32'))
    input_y = Tensor(np.array([3, 4]).astype('float32'))
    net = OpsCell(P.SquaredDifference())
    net_me = ops.GradOperation(get_all=True)(net)
    output = net_me(input_x, input_y)
    expect_output = np.array([-4.0, -4.0])
    assert np.allclose(output[0].asnumpy(), expect_output, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_square_sum_all_grad():
    """
    Feature: Test squared sum all with grad.
    Description: Test squared sum all with grad.
    Expectation: Success.
    """
    input_x = Tensor(np.array([1, 2]).astype('float32'))
    input_y = Tensor(np.array([3, 4]).astype('float32'))
    net = OpsCell(P.SquareSumAll())
    net_me = ops.GradOperation(get_all=True)(net)
    output = net_me(input_x, input_y)
    expect_output = np.array([2.0, 4.0])
    assert np.allclose(output[0].asnumpy(), expect_output, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_erfinv_grad():
    """
    Feature: Test erfinv with grad.
    Description: Test erfinv with grad.
    Expectation: Success.
    """
    input_x = Tensor(
        np.array([[0.5, 0.1, 0.2], [-0.5, 0.0, -0.9]]).astype('float32'))
    net = OpsCell(P.Erfinv())
    net_me = ops.GradOperation(get_all=True)(net)
    output = net_me(input_x)
    expect_output = np.array(
        [[1.1125, 0.8932, 0.9151], [1.1125, 0.8862, 3.4280]])
    assert np.allclose(output[0].asnumpy(), expect_output, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_betainc_grad():
    """
    Feature: Test betainc with grad.
    Description: Test betainc with grad.
    Expectation: Success.
    """
    a_np = np.array([[1, 2], [3, 4]]).astype(np.float32)
    b_np = np.array([[2, 3], [4, 5]]).astype(np.float32)
    x_np = np.array([[0.5, 0.5], [0.4, 0.3]]).astype(np.float32)
    a = Tensor(a_np)
    b = Tensor(b_np)
    x = Tensor(x_np)
    net = OpsCell(P.Betainc())
    net_me = ops.GradOperation(get_all=True)(net)
    output = net_me(a, b, x)
    expect_output = np.array([[1.0, 1.4999], [2.0736, 1.8151]])
    assert np.allclose(output[2].asnumpy(), expect_output, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_lp_norm_grad():
    """
    Feature: Test LpNorm with grad.
    Description: test LpNorm with grad.
    Expectation: Success.
    """
    axis = [0, 1]
    p = 2
    keep_dims = False
    lp_norm_net = OpsCell(P.LpNorm(axis, p, keep_dims))
    input_x_np = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32)
    expect_output = np.array([[[1.09108940e-01,  1.82574183e-01], [3.27326834e-01,  3.65148365e-01]],
                             [[5.45544684e-01,  5.47722578e-01], [7.63762593e-01,  7.30296731e-01]]]
                             ).astype(np.float32)
    net_me = ops.GradOperation(get_all=True)(lp_norm_net)
    output = net_me(Tensor(input_x_np))
    assert np.allclose(output[0].asnumpy(), expect_output, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_renorm_grad():
    """
    Feature: Test renorm with grad.
    Description: test renorm with grad.
    Expectation: Success.
    """
    renorm_net = OpsCell(ops.Renorm(p=1, dim=0, maxnorm=5.))
    x = Tensor(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), ms.float32)
    expect_output = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [
                             0.0, 0.0, 0.0]]).astype(np.float32)
    net_me = ops.GradOperation(get_all=True)(renorm_net)
    output = net_me(x)
    assert np.allclose(output[0].asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_polygamma_grad():
    """
    Feature: Test polygamma with grad.
    Description: test polygamma with grad.
    Expectation: Success.
    """
    polygamma_net = OpsCell(P.math_ops.Polygamma())
    a = np.array(1).astype(np.int64)
    x_ms = np.array([1, 0.4273, 9, -3.12, 12246.345]).astype(np.float16)
    expect_output = np.array(
        [-2.4043, -26.578, -0.013794, 1181, -0.0]).astype(np.float32)
    net_me = ops.GradOperation(get_all=True)(polygamma_net)
    output = net_me(Tensor(a), Tensor(x_ms))
    assert np.allclose(output[1].asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=1024.01)
def test_zeta_grad():
    """
    Feature: Test zeta with grad.
    Description: test zeta with grad.
    Expectation: Success.
    """
    zeta_net = OpsCell(ops.zeta)
    x = Tensor(np.array([2., 4.]), ms.float32)
    expect_output = np.array([0.0, 0.0]).astype(np.float32)
    net_me = ops.GradOperation(get_all=True)(zeta_net)
    output = net_me(x, 1)
    assert np.allclose(output[0].asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_meanext_grad():
    """
    Feature: Test meanext with grad.
    Description: test meanext with grad.
    Expectation: Success.
    """
    meanext_net = OpsCell(MeanExt())
    input_data = Tensor(np.ones([64, 128]), dtype=ms.float32)
    dim = 0
    keepdim = False
    expect_output = np.ones(shape=[64, 128]).astype(np.float32) * 0.015625
    net_me = ops.GradOperation(get_all=True)(meanext_net)
    output = net_me(input_data, dim, keepdim)
    assert np.allclose(output[0].asnumpy(), expect_output, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_selu_grad():
    """
    Feature: Test selu with grad.
    Description: test selu with grad.
    Expectation: Success.
    """
    selu_net = OpsCell(P.SeLU())
    x = Tensor(np.array([2., 4.]), ms.float32)
    expect_output = np.array([1.05070102, 1.05070102]).astype(np.float32)
    net_me = ops.GradOperation(get_all=True)(selu_net)
    output = net_me(x)
    assert np.allclose(output[0].asnumpy(), expect_output, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_sigmoidgrad_grad():
    """
    Feature: Test sigmoidgrad with grad.
    Description: test sigmoidgrad with grad.
    Expectation: Success.
    """
    sigmoidgrad_net = OpsCell(G.SigmoidGrad())
    y = Tensor(np.array([[[[-1, 1, 2],
                           [1, -1, 1],
                           [2, 1, -1]]]]).astype(np.float32))
    dy = Tensor(np.array([[[[-11, 2, 4],
                            [-1, 1, -1],
                            [-4, 4, -4]]]]).astype(np.float32))
    expect_output = np.array(
        [[[[-33, -2, -12], [1, 3, 1], [12, -4, -12]]]]).astype(np.float32)
    net_me = ops.GradOperation(get_all=True)(sigmoidgrad_net)
    output = net_me(y, dy)
    assert np.allclose(output[0].asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_softplusgrad_grad():
    """
    Feature: Test softplusGrad with grad.
    Description: test softplusGrad with grad.
    Expectation: Success.
    """
    softplusGrad_net = OpsCell(G.SoftplusGrad())
    y = Tensor(np.array([-1, 1, 2]).astype(np.float32))
    dy = Tensor(np.array([-11, 2, 4]).astype(np.float32))
    expect_output = np.array(
        [0.0000167, 0.8807970, 0.9820137]).astype(np.float32)
    net_me = ops.GradOperation(get_all=True)(softplusGrad_net)
    output = net_me(y, dy)
    assert np.allclose(output[0].asnumpy(), expect_output, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_softsign_grad():
    """
    Feature: Test softsign with grad.
    Description: test softsign with grad.
    Expectation: Success.
    """
    softsign_net = OpsCell(P.Softsign())
    y = Tensor(np.array([-1, 1, 2]).astype(np.float32))
    expect_output = np.array([0.25, 0.25, 0.111111]).astype(np.float32)
    net_me = ops.GradOperation(get_all=True)(softsign_net)
    output = net_me(y)
    assert np.allclose(output[0].asnumpy(), expect_output, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_tanhgrad_grad():
    """
    Feature: tanhgrad_with_grad
    Description: test tanhgrad_with_grad
    Expectation: the result match expectation
    """
    x = np.array([-0.5, 0, 0.5]).astype(np.float32)
    dy = np.array([1, 0, -1]).astype(np.float32)
    tanhgrad = OpsCell(G.TanhGrad())
    net_me = ops.GradOperation(get_all=True)(tanhgrad)
    output = net_me(Tensor(x), Tensor(dy))
    expect1 = np.array([1, 0, 1], dtype=np.float32)
    expect2 = np.array([0.75, 1.0, 0.75], dtype=np.float32)
    assert np.allclose(output[0].asnumpy(), expect1)
    assert np.allclose(output[1].asnumpy(), expect2)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_bcewithlogitsloss_grad():
    """
    Feature: bcewithlogitsloss_with_grad
    Description: test bcewithlogitsloss_with_grad
    Expectation: the result match expectation
    """
    logits = Tensor(
        np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]).astype(np.float32))
    label = Tensor(
        np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]).astype(np.float32))
    weight = Tensor(np.array([1.0, 1.0, 1.0]).astype(np.float32))
    pos_weight = Tensor(np.array([1.0, 1.0, 1.0]).astype(np.float32))
    bcewithlogitsloss = OpsCell(P.BCEWithLogitsLoss(reduction="mean"))
    net_me = ops.GradOperation(get_all=True)(bcewithlogitsloss)
    output = net_me(logits, label, weight, pos_weight)
    expect1 = np.array([[0.0016709, -0.0052458, -0.0886353],
                       [0.1791701, 0.0502187, -0.2553020]], dtype=np.float32)
    assert np.allclose(output[0].asnumpy(), expect1, 1e-6, 1e-6)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_celu_grad():
    """
    Feature: celu withgrad
    Description: test celu_with_grad
    Expectation: the result match expectation
    """
    x = np.array([-2.0, -1.0, 1.0, 2.0]).astype(np.float32)
    input_x = ms.Tensor(x, ms.float32)
    celugrad = OpsCell(ops.CeLU(1.0))
    net_me = ops.GradOperation(get_all=True)(celugrad)
    output = net_me(input_x)
    expect1 = np.array([0.135335, 0.367879, 1, 1], dtype=np.float32)
    print(output)
    assert np.allclose(output[0].asnumpy(), expect1, 1e-4, 1e-4)
