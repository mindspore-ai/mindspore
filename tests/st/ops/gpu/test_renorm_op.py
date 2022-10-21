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
# ============================================================================

import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.functional import vmap
from mindspore.common.api import jit


class ReNormNet(nn.Cell):
    def __init__(self, p=1, axis=0, maxnorm=10.0):
        super(ReNormNet, self).__init__()
        self.renorm = P.Renorm(p, axis, maxnorm)

    def construct(self, input_x):
        output = self.renorm(input_x)
        return output


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_renorm_op_float32(data_type=np.float32):
    """
    Feature: test Renorm with using float32.
    Description: inputs with batch.
    Expectation: the result match with expect.
    """
    error = 1e-6
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_x = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                        [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]).astype(data_type)
    benchmark_output = np.array([[[0.27777779, 0.55555558, 0.83333337, 1.11111116],
                                  [1.38888896, 1.66666675, 1.94444454, 2.22222233]],
                                 [[0.90000004, 1.00000000, 1.10000002, 1.20000005],
                                  [1.30000007, 1.39999998, 1.50000000, 1.60000002]]]).astype(data_type)

    re_norm = ReNormNet()
    output = re_norm(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output = re_norm(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_renorm_op_float16(data_type=np.float16):
    """
    Feature: test Renorm using float16.
    Description: inputs with batch.
    Expectation: the result match with expect.
    """
    error = 1e-3
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_x = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                        [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]).astype(data_type)
    benchmark_output = np.array([[[0.27783203, 0.55566406, 0.83349609, 1.11132812],
                                  [1.38867188, 1.66699219, 1.94531250, 2.22265625]],
                                 [[0.89990234, 1.00000000, 1.09960938, 1.19921875],
                                  [1.29980469, 1.39941406, 1.50000000, 1.59960938]]]).astype(data_type)

    re_norm = ReNormNet()
    output = re_norm(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output = re_norm(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_renorm_op1_float32(data_type=np.float32):
    """
    Feature: test Renorm using float32.
    Description: inputs with batch.
    Expectation: the result match with expect.
    """
    error = 1e-6
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_x = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                        [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]).astype(data_type)
    benchmark_output = np.array([[[0.45834923, 0.91669846, 1.37504768, 1.83339691],
                                  [1.56556070, 1.87867284, 2.19178486, 2.50489712]],
                                 [[4.12514305, 4.58349228, 5.04184151, 5.50019073],
                                  [4.07045794, 4.38356972, 4.69668198, 5.00979424]]]).astype(data_type)

    re_norm = ReNormNet(p=2, axis=1, maxnorm=10.0)
    output = re_norm(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output = re_norm(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_renorm_op2_float16(data_type=np.float16):
    """
    Feature: test Renorm using float16.
    Description: inputs with batch.
    Expectation: the result match with expect.
    """
    error = 1e-3
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_x = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                        [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]).astype(data_type)
    benchmark_output = np.array([[[0.60192931, 1.09108937, 1.49255586, 1.82574177],
                                  [3.00964642, 3.27326822, 3.48263025, 3.65148354]],
                                 [[5.41736364, 5.45544672, 5.47270441, 5.47722530],
                                  [7.82508087, 7.63762569, 7.46277905, 7.30296707]]]).astype(data_type)

    re_norm = ReNormNet(p=2, axis=2, maxnorm=10.0)
    output = re_norm(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output = re_norm(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)


def vmap_case():
    class Net(nn.Cell):
        def __init__(self, p, axis, maxnorm):
            super(Net, self).__init__()
            self.renorm = P.Renorm(p, axis, maxnorm)

        def construct(self, x):
            return self.renorm(x)

    class VmapNet(nn.Cell):
        def __init__(self, net, in_axes, out_axes):
            super(VmapNet, self).__init__()
            self.net = net
            self.in_axes = in_axes
            self.out_axes = out_axes

        def construct(self, x):
            return vmap(self.net, self.in_axes, self.out_axes)(x)

    @jit
    def for_net(input_x, p, axis, maxnorm):
        # split and concat along dimension 0
        output = []
        for i in range(x.shape[0]):
            out = P.Renorm(p, axis, maxnorm)(input_x[i])
            output.append(out)
        return F.stack(output)

    x = Tensor(np.array([[[[4, 3, 2, 1], [8, 7, 6, 5], [12, 11, 10, 9]],
                          [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]],
                         [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                          [[4, 3, 2, 1], [8, 7, 6, 5], [12, 11, 10, 9]]]], dtype=np.float32))
    output = VmapNet(Net(1, 0, 10.0), 0, 0)(x)
    fornet_output = for_net(x, 1, 0, 10.0)
    np.testing.assert_allclose(output.asnumpy(), fornet_output.asnumpy(), rtol=1e-6)


def vmap_nested_case():
    class Net(nn.Cell):
        def __init__(self, p, axis, maxnorm):
            super(Net, self).__init__()
            self.renorm = P.Renorm(p, axis, maxnorm)

        def construct(self, x):
            return self.renorm(x)

    class WrapNet(nn.Cell):
        def __init__(self, net, inin_axes, inout_axes, outin_axes, outout_axes):
            super(WrapNet, self).__init__()
            self.net = net
            self.ii = inin_axes
            self.io = inout_axes
            self.oi = outin_axes
            self.oo = outout_axes

        def construct(self, x):
            return vmap(vmap(self.net, self.ii, self.io), self.oi, self.oo)(x)

    @jit
    def for_net(input_x, p, axis, maxnorm):
        # split and concat along dimension 0 and 1
        output = []
        for i in range(x.shape[0]):
            inner_output = []
            for j in range(x.shape[1]):
                out = P.Renorm(p, axis, maxnorm)(input_x[i][j])
                inner_output.append(out)
            output.append(F.stack(inner_output))
        return F.stack(output)

    x = Tensor(np.array([[[[4, 3, 2, 1], [8, 7, 6, 5], [12, 11, 10, 9]],
                          [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]],
                         [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                          [[4, 3, 2, 1], [8, 7, 6, 5], [12, 11, 10, 9]]]], dtype=np.float32))

    output = WrapNet(Net(1, 0, 10.0), 0, 0, 1, 1)(x)
    fornet_output = for_net(x, 1, 0, 10.0)
    np.testing.assert_allclose(output.asnumpy(), fornet_output.asnumpy(), rtol=1e-6)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_renorm_vmap_cpu():
    """
    Feature: test Renorm vmap on GPU.
    Description: inputs with batch.
    Expectation: the result match with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    vmap_case()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_renorm_vmap_cpu_nested():
    """
    Feature: test nested Renorm vmap on GPU.
    Description: inputs with batch.
    Expectation: the result match with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    vmap_nested_case()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_renorm_op2_complex64(data_type=np.complex64):
    """
    Feature: test Renorm using complex64.
    Description: inputs with batch.
    Expectation: the result match with expect.
    """
    error = 1e-6
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_x = np.array([[1+2j, 2+3j], [3+4j, 4+5j]]).astype(data_type)
    benchmark_output = np.array([[0.91287088+1.82574177j, 1.36082768+2.04124165j],
                                 [2.73861265+3.65148354j, 2.72165537+3.40206909j]]).astype(data_type)

    re_norm = ReNormNet(p=2, axis=1, maxnorm=5.0)
    output = re_norm(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output = re_norm(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)
