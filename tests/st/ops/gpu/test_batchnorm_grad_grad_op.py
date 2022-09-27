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

"""test batchNormGradGrad op"""

import numpy as np
import pytest

from mindspore import context
from mindspore import Tensor
from mindspore import nn
from mindspore.ops import functional as F
from mindspore.ops.operations._grad_ops import BatchNormGradGrad

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class BatchnormGradGradNet(nn.Cell):
    def __init__(self, is_training=False, epsilon=1e-5, data_format='NHWC'):
        super(BatchnormGradGradNet, self).__init__()
        self.bn_grad_gad = BatchNormGradGrad(is_training=is_training, epsilon=epsilon, data_format=data_format)

    def construct(self, dy, x, scale, mean, variance, dout_dx, dout_dscale, dout_dbias):
        ddy, dx, dscale = self.bn_grad_gad(x, dy, scale, mean, variance, dout_dx, dout_dscale, dout_dbias)
        return dx, ddy, dscale


np_type_fp32 = np.float32
np_types = (np.float16, np.float32)


def get_nhwc_inputs(np_type):
    x = Tensor(np.array([[[[1., 6., 10.]],
                          [[2., 11., 4.]]],
                         [[[9., 16., 4.]],
                          [[5., 3., 8.]]]]).astype(np_type))
    scale = Tensor(np.array([5., 10., 8.]).astype(np_type_fp32))
    dy = Tensor(np.array([[[[1., 3., 32.]],
                           [[1., 5., 3.]]],
                          [[[1., 3., 32.]],
                           [[1., 5., 5.]]]]).astype(np_type))
    dout_dx = Tensor(np.array([[[[10., 5., 20.]],
                                [[1., 3., 2.]]],
                               [[[10., 5., 20.]],
                                [[1., 3., 5.]]]]).astype(np_type))
    dout_dscale = Tensor(np.array([10., 20., 13.]).astype(np_type_fp32))
    dout_dbias = Tensor(np.array([10., 15., 13.]).astype(np_type_fp32))
    return x, scale, dy, dout_dx, dout_dscale, dout_dbias


def get_nchw_inputs(np_type):
    x = Tensor(np.array([[[[1., 6., 10.]],
                          [[2., 11., 4.]]],
                         [[[9., 16., 4.]],
                          [[5., 3., 8.]]]]).astype(np_type))
    scale = Tensor(np.array([5., 10.]).astype(np_type_fp32))
    dy = Tensor(np.array([[[[1., 3., 32.]],
                           [[1., 5., 3.]]],
                          [[[1., 3., 32.]],
                           [[1., 5., 5.]]]]).astype(np_type))
    dout_dx = Tensor(np.array([[[[10., 5., 20.]],
                                [[1., 3., 2.]]],
                               [[[10., 5., 20.]],
                                [[1., 3., 5.]]]]).astype(np_type))
    dout_dscale = Tensor(np.array([10., 20.]).astype(np_type_fp32))
    dout_dbias = Tensor(np.array([10., 15.]).astype(np_type_fp32))
    return x, scale, dy, dout_dx, dout_dscale, dout_dbias


def get_nc_inputs(np_type):
    x = Tensor(np.array([[1., 6.],
                         [2., 11.]]).astype(np_type))
    scale = Tensor(np.array([5., 10.]).astype(np_type_fp32))
    dy = Tensor(np.array([[1., 3.],
                          [1., 5.]]).astype(np_type))
    dout_dx = Tensor(np.array([[10., 5.],
                               [1., 3.]]).astype(np_type))
    dout_dscale = Tensor(np.array([10., 20.]).astype(np_type_fp32))
    dout_dbias = Tensor(np.array([10., 15.]).astype(np_type_fp32))
    return x, scale, dy, dout_dx, dout_dscale, dout_dbias


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchnorm_grad_grad_training_nhwc():
    """
    Feature: test BatchNormGradGrad
    Description: test BatchNormGradGrad in training mode and NHWC format
    Expectation: The outputs are same as exception
    """
    for np_type in np_types:
        x, scale, dy, dout_dx, dout_dscale, dout_dbias = get_nhwc_inputs(np_type)
        mean = Tensor(np.array([4.25, 9., 6.5]).astype(np_type_fp32))
        variance = Tensor(np.array([9.6875, 24.5, 6.75]).astype(np_type_fp32))
        batchnorf_grad_grad = BatchnormGradGradNet(is_training=True, epsilon=1e-10, data_format="NHWC")
        ddy, dx, dscale = batchnorf_grad_grad(dy, x, scale, mean, variance, dout_dx, dout_dscale, dout_dbias)
        expect_ddy = np.array([[[[8.60602665e+00, 5.39324379e+00, 4.69353676e+01]],
                                [[-3.19870663e+00, 2.07310696e+01, -2.31164665e+01]]],
                               [[[2.98317432e+01, 4.41501160e+01, 3.23091583e+01]],
                                [[4.76093674e+00, -1.02744274e+01, -4.12805939e+00]]]]).astype(np_type)
        expect_dx = np.array([[[[0.00000000e+00, -4.82651806e+00, -1.70410995e+02]],
                               [[0.00000000e+00, 4.45460033e+00, 1.23609985e+02]]],
                              [[[0.00000000e+00, -1.10733974e+00, 1.28613693e+02]],
                               [[0.00000000e+00, 1.47925758e+00, -8.18126678e+01]]]]).astype(np_type)
        expect_dscale = np.array([0.00000000e+00, -6.76183939e-01, 1.68714584e+02]).astype(np_type_fp32)
        assert np.allclose(dx.asnumpy(), expect_dx)
        assert np.allclose(ddy.asnumpy(), expect_ddy)
        assert np.allclose(dscale.asnumpy(), expect_dscale)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchnorm_grad_grad_training_nchw():
    """
    Feature: test BatchNormGradGrad
    Description: test BatchNormGradGrad in training mode and NCHW format
    Expectation: The outputs are same as exception
    """
    for np_type in np_types:
        x, scale, dy, dout_dx, dout_dscale, dout_dbias = get_nchw_inputs(np_type)
        mean = Tensor(np.array([7.666667, 5.5]).astype(np_type_fp32))
        variance = Tensor(np.array([22.888891, 9.583334]).astype(np_type_fp32))
        batchnorf_grad_grad = BatchnormGradGradNet(is_training=True, epsilon=1e-10, data_format="NCHW")
        ddy, dx, dscale = batchnorf_grad_grad(dy, x, scale, mean, variance, dout_dx, dout_dscale, dout_dbias)
        expect_ddy = np.array([[[[-8.04403496e+00, -1.04288101e+00, 2.44149323e+01]],
                                [[-9.60639381e+00, 4.76680756e+01, 4.91587257e+00]]],
                               [[[1.15186062e+01, 2.34104195e+01, 9.74295139e+00]],
                                [[7.33156919e+00, 2.50017643e+00, 3.71906967e+01]]]]).astype(np_type)
        expect_dx = np.array([[[[-6.68375254e+00, -1.82859936e+01, 4.34827538e+01]],
                               [[-4.37121344e+00, -2.44060922e+00, 1.73812890e+00]]],
                              [[[-3.13677616e+01, -4.91410027e+01, 6.19957581e+01]],
                               [[-1.05440617e+01, 1.40203199e+01, 1.59743786e+00]]]]).astype(np_type)
        expect_dscale = np.array([1.00431015e+02, 2.65164828e+00]).astype(np_type_fp32)
        assert np.allclose(dx.asnumpy(), expect_dx)
        assert np.allclose(ddy.asnumpy(), expect_ddy)
        assert np.allclose(dscale.asnumpy(), expect_dscale)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchnorm_grad_grad_inference_nhwc():
    """
    Feature: test BatchNormGradGrad
    Description: test BatchNormGradGrad in inference mode and NHWC format
    Expectation: The outputs are same as exception
    """
    for np_type in np_types:
        x, scale, dy, dout_dx, dout_dscale, dout_dbias = get_nhwc_inputs(np_type)
        mean = Tensor(np.array([10., 4., 2.]).astype(np_type_fp32))
        variance = Tensor(np.array([0.1, 0.4, 0.3]).astype(np_type_fp32))
        batchnorf_grad_grad = BatchnormGradGradNet(is_training=False, epsilon=1e-10, data_format="NHWC")
        ddy, dx, dscale = batchnorf_grad_grad(dy, x, scale, mean, variance, dout_dx, dout_dscale, dout_dbias)
        expect_ddy = np.array([[[[-1.16491104e+02, 1.57302490e+02, 4.94995819e+02]],
                                [[-2.27170822e+02, 2.83793610e+02, 8.96811523e+01]]],
                               [[[1.36491104e+02, 4.73530273e+02, 3.52587952e+02]],
                                [[-1.32302490e+02, 3.08113918e+01, 2.28437531e+02]]]]).astype(np_type)
        expect_dx = np.array([[[[3.16227760e+01, 9.48683319e+01, 7.59508545e+02]],
                               [[3.16227760e+01, 1.58113876e+02, 7.12039261e+01]]],
                              [[[3.16227760e+01, 9.48683319e+01, 7.59508545e+02]],
                               [[3.16227760e+01, 1.58113876e+02, 1.18673210e+02]]]]).astype(np_type)
        expect_dscale = np.array([6.95701065e+01, 9.48683319e+01, 2.39354736e+03]).astype(np_type_fp32)
        assert np.allclose(dx.asnumpy(), expect_dx)
        assert np.allclose(ddy.asnumpy(), expect_ddy)
        assert np.allclose(dscale.asnumpy(), expect_dscale)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchnorm_grad_grad_inference_nchw():
    """
    Feature: test BatchNormGradGrad
    Description: test BatchNormGradGrad in inference mode and NCHW format
    Expectation: The outputs are same as exception
    """
    for np_type in np_types:
        x, scale, dy, dout_dx, dout_dscale, dout_dbias = get_nchw_inputs(np_type)
        mean = Tensor(np.array([10., 4.]).astype(np_type_fp32))
        variance = Tensor(np.array([0.1, 0.4]).astype(np_type_fp32))
        batchnorf_grad_grad = BatchnormGradGradNet(is_training=False, epsilon=1e-10, data_format="NCHW")
        ddy, dx, dscale = batchnorf_grad_grad(dy, x, scale, mean, variance, dout_dx, dout_dscale, dout_dbias)
        expect_ddy = np.array([[[[-1.16491104e+02, -3.74341660e+01, 3.26227753e+02]],
                                [[-3.24341621e+01, 2.83793610e+02, 4.66227760e+01]]],
                               [[[1.36491104e+02, 2.78793610e+02, 1.36491089e+02]],
                                [[6.24341660e+01, 3.08113918e+01, 2.20548050e+02]]]]).astype(np_type)
        expect_dx = np.array([[[[3.16227760e+01, 9.48683319e+01, 1.01192883e+03]],
                               [[3.16227760e+01, 1.58113876e+02, 9.48683319e+01]]],
                              [[[3.16227760e+01, 9.48683319e+01, 1.01192883e+03]],
                               [[3.16227760e+01, 1.58113876e+02, 1.58113876e+02]]]]).astype(np_type)
        expect_dscale = np.array([4.20582910e+03, 9.96117477e+01]).astype(np_type_fp32)
        assert np.allclose(dx.asnumpy(), expect_dx)
        assert np.allclose(ddy.asnumpy(), expect_ddy)
        assert np.allclose(dscale.asnumpy(), expect_dscale)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchnorm_grad_grad_training_nc():
    """
    Feature: test BatchNormGradGrad
    Description: test BatchNormGradGrad in training mode and NC format
    Expectation: The outputs are same as exception
    """
    for np_type in np_types:
        x, scale, dy, dout_dx, dout_dscale, dout_dbias = get_nc_inputs(np_type)
        mean = Tensor(np.array([1.5, 8.5]).astype(np_type_fp32))
        variance = Tensor(np.array([0.25, 6.25]).astype(np_type_fp32))
        batchnorf_grad_grad = BatchnormGradGradNet(is_training=True, epsilon=1e-10, data_format="NCHW")
        ddy, dx, dscale = batchnorf_grad_grad(dy, x, scale, mean, variance, dout_dx, dout_dscale, dout_dbias)
        expect_ddy = np.array([[0.00000000e+00, -5.00000000e+00],
                               [2.00000000e+01, 3.50000000e+01]]).astype(np_type)
        expect_dx = np.array([[0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00]]).astype(np_type)
        expect_dscale = np.array([0.00000000e+00, 0.00000000e+00]).astype(np_type_fp32)
        assert np.allclose(dx.asnumpy(), expect_dx)
        assert np.allclose(ddy.asnumpy(), expect_ddy)
        assert np.allclose(dscale.asnumpy(), expect_dscale)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchnorm_grad_grad_inference_nc():
    """
    Feature: test BatchNormGradGrad
    Description: test BatchNormGradGrad in inference mode and NC format
    Expectation: The outputs are same as exception
    """
    for np_type in np_types:
        x, scale, dy, dout_dx, dout_dscale, dout_dbias = get_nc_inputs(np_type)
        mean = Tensor(np.array([10., 4.]).astype(np_type_fp32))
        variance = Tensor(np.array([0.1, 0.4]).astype(np_type_fp32))
        batchnorf_grad_grad = BatchnormGradGradNet(is_training=False, epsilon=1e-10, data_format="NCHW")
        ddy, dx, dscale = batchnorf_grad_grad(dy, x, scale, mean, variance, dout_dx, dout_dscale, dout_dbias)
        expect_ddy = np.array([[-1.16491104e+02, 1.57302490e+02],
                               [-2.27170822e+02, 2.83793610e+02]]).astype(np_type)
        expect_dx = np.array([[3.16227760e+01, 9.48683319e+01],
                              [3.16227760e+01, 1.58113876e+02]]).astype(np_type)
        expect_dscale = np.array([3.47850533e+01, 4.74341660e+01]).astype(np_type_fp32)
        assert np.allclose(dx.asnumpy(), expect_dx)
        assert np.allclose(ddy.asnumpy(), expect_ddy)
        assert np.allclose(dscale.asnumpy(), expect_dscale)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchnorm_grad_grad_training_nchw_2():
    """
    Feature: test BatchNormGradGrad
    Description: test BatchNormGradGrad in training mode and NCHW format
    Expectation: The outputs are same as exception
    """
    for np_type in np_types:
        x = Tensor(np.array([1, 6, 2, 11, 10, 16, 22, 11, 10, 6, 23, 11, 18, 36, 24, 11, 3, 5, 9, 2, 7, 4, 5, 13])
                   .reshape((2, 3, 2, 2)).astype(np_type))
        scale = Tensor(np.array([5, 10, 13]).astype(np_type_fp32))
        dy = Tensor(np.array([4, 3, 11, 6, 7, 3, 2, 15, 21, 8, 10, 5, 12, 6, 8, 5, 21, 30, 8, 5, 4, 20, 12, 26])
                    .reshape((2, 3, 2, 2)).astype(np_type))
        dout_dx = Tensor(np.array([10, 45, 1, 31, 10, 5, 13, 3, 10, 25, 7, 23, 10, 5, 21,
                                   3, 10, 5, 19, 6, 10, 5, 2, 33]).reshape((2, 3, 2, 2)).astype(np_type))
        dout_dscale = Tensor(np.array([10, 20, 8]).astype(np_type_fp32))
        dout_dbias = Tensor(np.array([10, 15, 2]).astype(np_type_fp32))
        mean = Tensor(np.array([13.625, 9.75, 9.875]).astype(np_type_fp32))
        variance = Tensor(np.array([124.234375, 39.9375, 33.109375]).astype(np_type_fp32))
        batchnorf_grad_grad = BatchnormGradGradNet(is_training=True, epsilon=1e-10, data_format="NCHW")
        ddy, dx, dscale = batchnorf_grad_grad(dy, x, scale, mean, variance, dout_dx, dout_dscale, dout_dbias)
        expect_ddy = np.array([[[[-5.0274234, 15.603123],
                                 [-8.078747, 14.252785]],
                                [[17.515953, 27.26279],
                                 [57.58051, 9.382442]],
                                [[-7.7716827, 22.513603],
                                 [-2.8373299, 22.499744]]],
                               [[[11.734285, 27.239033],
                                 [22.584667, 1.6922784]],
                                [[-3.0858822, -5.1115246],
                                 [28.814217, -12.358505]],
                                [[-10.474489, -24.473639],
                                 [-30.350508, 46.8943]]]]).astype(np_type)
        expect_dx = np.array([[[[-3.0650284, -4.545036],
                                [4.0242057, -1.2047515]],
                               [[-10.74366, -13.374659],
                                [9.901716, 6.22377]],
                               [[6.9668703, -1.8542786],
                                [-13.217213, -8.492305]]],
                              [[[5.2714543, -0.08000898],
                                [1.2891254, -1.6899571]],
                               [[12.131741, 37.288],
                                [1.404377, -42.831272]],
                               [[-5.9648676, 10.930217],
                                [2.98559, 8.645981]]]]).astype(np_type)
        expect_dscale = np.array([-13.763241, -11.468014, 16.121202]).astype(np_type_fp32)
        assert np.allclose(dx.asnumpy(), expect_dx)
        assert np.allclose(ddy.asnumpy(), expect_ddy)
        assert np.allclose(dscale.asnumpy(), expect_dscale)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchnorm_grad_grad_training_nhwc_2():
    """
    Feature: test BatchNormGradGrad
    Description: test BatchNormGradGrad in training mode and NHWC format
    Expectation: The outputs are same as exception
    """
    for np_type in np_types:
        x = Tensor(np.array([1, 6, 2, 11, 10, 16, 22, 11, 10, 6, 23, 11, 18, 36, 24, 11, 3, 5, 9, 2, 7, 4, 5, 13])
                   .reshape((2, 2, 2, 3)).astype(np_type))
        scale = Tensor(np.array([5, 10, 13]).astype(np_type_fp32))
        dy = Tensor(np.array([4, 3, 11, 6, 7, 3, 2, 15, 21, 8, 10, 5, 12, 6, 8, 5, 21, 30, 8, 5, 4, 20, 12, 26])
                    .reshape((2, 2, 2, 3)).astype(np_type))
        dout_dx = Tensor(np.array([10, 45, 1, 31, 10, 5, 13, 3, 10, 25, 7, 23, 10, 5, 21,
                                   3, 10, 5, 19, 6, 10, 5, 2, 33]).reshape((2, 2, 2, 3)).astype(np_type))
        dout_dscale = Tensor(np.array([10, 20, 8]).astype(np_type_fp32))
        dout_dbias = Tensor(np.array([10, 15, 2]).astype(np_type_fp32))
        mean = Tensor(np.array([10.25, 12., 11.]).astype(np_type_fp32))
        variance = Tensor(np.array([42.9375, 121., 41.5]).astype(np_type_fp32))
        batchnorf_grad_grad = BatchnormGradGradNet(is_training=True, epsilon=1e-10, data_format="NHWC")
        ddy, dx, dscale = batchnorf_grad_grad(dy, x, scale, mean, variance, dout_dx, dout_dscale, dout_dbias)
        expect_ddy = np.array([[[[-7.508998, 33.534935, -19.357697],
                                 [23.731525, 9.96619, -17.301367]],
                                [[26.734844, 5.664913, -4.6332817],
                                 [11.544975, 34.049587, 21.170918]]],
                               [[[18.359093, 59.042076, 11.548992],
                                 [2.3661919, -4.4703217, -12.574798]],
                                [[11.531647, -10.169046, -3.3442173],
                                 [-6.7592793, -7.618332, 40.491455]]]]).astype(np_type)
        expect_dx = np.array([[[[-12.525581, -13.745609, 10.849744],
                                [0.18534184, -7.1285133, -14.531107]],
                               [[-2.6849413, 9.152922, -3.9971244],
                                [-0.79296684, 5.818636, 12.025189]]],
                              [[[9.639168, 4.4382114, -21.06007],
                                [-6.3899727, 17.928495, -0.7197275]],
                               [[-0.11268234, -16.05016, 10.23838],
                                [12.681629, -0.41397095, 7.1947174]]]]).astype(np_type)
        expect_dscale = np.array([-15.722887, -26.014277, 35.56406]).astype(np_type_fp32)
        assert np.allclose(dx.asnumpy(), expect_dx)
        assert np.allclose(ddy.asnumpy(), expect_ddy)
        assert np.allclose(dscale.asnumpy(), expect_dscale)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchnorm_grad_grad_training_nchw_dynamic_shape():
    """
    Feature: test BatchNormGradGrad
    Description: test BatchNormGradGrad dynamic shape in training mode and NCHW format
    Expectation: The outputs are same as exception
    """
    for np_type in np_types:
        x = Tensor(np.array([1, 6, 2, 11, 10, 16, 22, 11, 10, 6, 23, 11, 18, 36, 24, 11, 3, 5, 9, 2, 7, 4, 5, 13])
                   .reshape((2, 3, 2, 2)).astype(np_type))
        scale = Tensor(np.array([5, 10, 13]).astype(np_type_fp32))
        dy = Tensor(np.array([4, 3, 11, 6, 7, 3, 2, 15, 21, 8, 10, 5, 12, 6, 8, 5, 21, 30, 8, 5, 4, 20, 12, 26])
                    .reshape((2, 3, 2, 2)).astype(np_type))
        dout_dx = Tensor(np.array([10, 45, 1, 31, 10, 5, 13, 3, 10, 25, 7, 23, 10, 5, 21,
                                   3, 10, 5, 19, 6, 10, 5, 2, 33]).reshape((2, 3, 2, 2)).astype(np_type))
        dout_dscale = Tensor(np.array([10, 20, 8]).astype(np_type_fp32))
        dout_dbias = Tensor(np.array([10, 15, 2]).astype(np_type_fp32))
        mean = Tensor(np.array([13.625, 9.75, 9.875]).astype(np_type_fp32))
        variance = Tensor(np.array([124.234375, 39.9375, 33.109375]).astype(np_type_fp32))
        batchnorf_grad_grad = BatchnormGradGradNet(is_training=True, epsilon=1e-10, data_format="NCHW")
        x_dyn = Tensor(shape=(2, 3, 2, None), dtype=x.dtype)
        scale_dyn = Tensor(shape=(None,), dtype=scale.dtype)
        dout_dx_dyn = Tensor(shape=(2, None, 2, 2), dtype=dout_dx.dtype)
        dout_dbias_dyn = Tensor(shape=(None,), dtype=dout_dbias.dtype)
        batchnorf_grad_grad.set_inputs(dy, x_dyn, scale_dyn, mean, variance,
                                       dout_dx_dyn, dout_dscale, dout_dbias_dyn)
        ddy, dx, dscale = batchnorf_grad_grad(dy, x, scale, mean, variance, dout_dx, dout_dscale, dout_dbias)
        expect_ddy = np.array([[[[-5.0274234, 15.603123],
                                 [-8.078747, 14.252785]],
                                [[17.515953, 27.26279],
                                 [57.58051, 9.382442]],
                                [[-7.7716827, 22.513603],
                                 [-2.8373299, 22.499744]]],
                               [[[11.734285, 27.239033],
                                 [22.584667, 1.6922784]],
                                [[-3.0858822, -5.1115246],
                                 [28.814217, -12.358505]],
                                [[-10.474489, -24.473639],
                                 [-30.350508, 46.8943]]]]).astype(np_type)
        expect_dx = np.array([[[[-3.0650284, -4.545036],
                                [4.0242057, -1.2047515]],
                               [[-10.74366, -13.374659],
                                [9.901716, 6.22377]],
                               [[6.9668703, -1.8542786],
                                [-13.217213, -8.492305]]],
                              [[[5.2714543, -0.08000898],
                                [1.2891254, -1.6899571]],
                               [[12.131741, 37.288],
                                [1.404377, -42.831272]],
                               [[-5.9648676, 10.930217],
                                [2.98559, 8.645981]]]]).astype(np_type)
        expect_dscale = np.array([-13.763241, -11.468014, 16.121202]).astype(np_type_fp32)
        assert np.allclose(dx.asnumpy(), expect_dx)
        assert np.allclose(ddy.asnumpy(), expect_ddy)
        assert np.allclose(dscale.asnumpy(), expect_dscale)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchnorm_grad_grad_training_nhwc_dynamic_shape():
    """
    Feature: test BatchNormGradGrad
    Description: test BatchNormGradGrad dynamic shape in training mode NHWC format
    Expectation: The outputs are same as exception
    """
    for np_type in np_types:
        x = Tensor(np.array([1, 6, 2, 11, 10, 16, 22, 11, 10, 6, 23, 11, 18, 36, 24, 11, 3, 5, 9, 2, 7, 4, 5, 13])
                   .reshape((2, 2, 2, 3)).astype(np_type))
        scale = Tensor(np.array([5, 10, 13]).astype(np_type_fp32))
        dy = Tensor(np.array([4, 3, 11, 6, 7, 3, 2, 15, 21, 8, 10, 5, 12, 6, 8, 5, 21, 30, 8, 5, 4, 20, 12, 26])
                    .reshape((2, 2, 2, 3)).astype(np_type))
        dout_dx = Tensor(np.array([10, 45, 1, 31, 10, 5, 13, 3, 10, 25, 7, 23, 10, 5, 21,
                                   3, 10, 5, 19, 6, 10, 5, 2, 33]).reshape((2, 2, 2, 3)).astype(np_type))
        dout_dscale = Tensor(np.array([10, 20, 8]).astype(np_type_fp32))
        dout_dbias = Tensor(np.array([10, 15, 2]).astype(np_type_fp32))
        mean = Tensor(np.array([10.25, 12., 11.]).astype(np_type_fp32))
        variance = Tensor(np.array([42.9375, 121., 41.5]).astype(np_type_fp32))
        batchnorf_grad_grad = BatchnormGradGradNet(is_training=True, epsilon=1e-10, data_format="NHWC")
        x_dyn = Tensor(shape=(2, 2, 2, None), dtype=x.dtype)
        scale_dyn = Tensor(shape=(None,), dtype=scale.dtype)
        dout_dx_dyn = Tensor(shape=(2, None, 2, 3), dtype=dout_dx.dtype)
        dout_dbias_dyn = Tensor(shape=(None,), dtype=dout_dbias.dtype)
        batchnorf_grad_grad.set_inputs(dy, x_dyn, scale_dyn, mean, variance,
                                       dout_dx_dyn, dout_dscale, dout_dbias_dyn)
        ddy, dx, dscale = batchnorf_grad_grad(dy, x, scale, mean, variance, dout_dx, dout_dscale, dout_dbias)
        expect_ddy = np.array([[[[-7.508998, 33.534935, -19.357697],
                                 [23.731525, 9.96619, -17.301367]],
                                [[26.734844, 5.664913, -4.6332817],
                                 [11.544975, 34.049587, 21.170918]]],
                               [[[18.359093, 59.042076, 11.548992],
                                 [2.3661919, -4.4703217, -12.574798]],
                                [[11.531647, -10.169046, -3.3442173],
                                 [-6.7592793, -7.618332, 40.491455]]]]).astype(np_type)
        expect_dx = np.array([[[[-12.525581, -13.745609, 10.849744],
                                [0.18534184, -7.1285133, -14.531107]],
                               [[-2.6849413, 9.152922, -3.9971244],
                                [-0.79296684, 5.818636, 12.025189]]],
                              [[[9.639168, 4.4382114, -21.06007],
                                [-6.3899727, 17.928495, -0.7197275]],
                               [[-0.11268234, -16.05016, 10.23838],
                                [12.681629, -0.41397095, 7.1947174]]]]).astype(np_type)
        expect_dscale = np.array([-15.722887, -26.014277, 35.56406]).astype(np_type_fp32)
        assert np.allclose(dx.asnumpy(), expect_dx)
        assert np.allclose(ddy.asnumpy(), expect_ddy)
        assert np.allclose(dscale.asnumpy(), expect_dscale)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchnorm_grad_grad_training_nchw_vmap():
    """
    Feature: test BatchNormGradGrad
    Description: test BatchNormGradGrad vmap function in training mode and NCHW format
    Expectation: The outputs are same as exception
    """
    for np_type in np_types:
        x, scale, dy, dout_dx, dout_dscale, dout_dbias = get_nchw_inputs(np_type)
        x_new_shape = (2, 2, 1, 1, 3)
        scale_new_shape = (2, 1)
        x, dy, dout_dx = x.reshape(x_new_shape), dy.reshape(x_new_shape), dout_dx.reshape(x_new_shape)
        scale, dout_dscale, dout_dbias = scale.reshape(scale_new_shape), dout_dscale.reshape(scale_new_shape), \
                                         dout_dbias.reshape(scale_new_shape)
        mean = Tensor(np.array([7.666667, 5.5]).astype(np_type_fp32).reshape(scale_new_shape))
        variance = Tensor(np.array([22.888891, 9.583334]).astype(np_type_fp32).reshape(scale_new_shape))
        batchnorf_grad_grad = BatchnormGradGradNet(is_training=True, epsilon=1e-10, data_format="NCHW")
        ddy, dx, dscale = F.vmap(batchnorf_grad_grad, (1, 1, 0, 0, 0, 1, 0, 0), (1, 1, 0))(dy, x, scale, mean, variance,
                                                                                           dout_dx, dout_dscale,
                                                                                           dout_dbias)
        expect_ddy = np.array([[[[-8.04403496e+00, -1.04288101e+00, 2.44149323e+01]],
                                [[-9.60639381e+00, 4.76680756e+01, 4.91587257e+00]]],
                               [[[1.15186062e+01, 2.34104195e+01, 9.74295139e+00]],
                                [[7.33156919e+00, 2.50017643e+00, 3.71906967e+01]]]]).astype(np_type) \
            .reshape(x_new_shape)
        expect_dx = np.array([[[[-6.68375254e+00, -1.82859936e+01, 4.34827538e+01]],
                               [[-4.37121344e+00, -2.44060922e+00, 1.73812890e+00]]],
                              [[[-3.13677616e+01, -4.91410027e+01, 6.19957581e+01]],
                               [[-1.05440617e+01, 1.40203199e+01, 1.59743786e+00]]]]).astype(np_type) \
            .reshape(x_new_shape)
        expect_dscale = np.array([1.00431015e+02, 2.65164828e+00]).astype(np_type_fp32) \
            .reshape(scale_new_shape)
        assert np.allclose(dx.asnumpy(), expect_dx)
        assert np.allclose(ddy.asnumpy(), expect_ddy)
        assert np.allclose(dscale.asnumpy(), expect_dscale)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchnorm_grad_grad_inference_nhwc_vmap():
    """
    Feature: test BatchNormGradGrad
    Description: test BatchNormGradGrad vmap function in inference mode and NHWC format
    Expectation: The outputs are same as exception
    """
    for np_type in np_types:
        x, scale, dy, dout_dx, dout_dscale, dout_dbias = get_nhwc_inputs(np_type)
        mean = Tensor(np.array([10., 4., 2.]).astype(np_type_fp32))
        variance = Tensor(np.array([0.1, 0.4, 0.3]).astype(np_type_fp32))
        x_new_shape = (2, 2, 1, 3, 1)
        scale_new_shape = (3, 1)
        batchnorf_grad_grad = BatchnormGradGradNet(is_training=False, epsilon=1e-10, data_format="NHWC")
        ddy, dx, dscale = F.vmap(batchnorf_grad_grad, (3, 3, 0, 0, 0, 3, 0, 0), (3, 3, 0))(
            dy.reshape(x_new_shape), x.reshape(x_new_shape), scale.reshape(scale_new_shape),
            mean.reshape(scale_new_shape), variance.reshape(scale_new_shape), dout_dx.reshape(x_new_shape),
            dout_dscale.reshape(scale_new_shape), dout_dbias.reshape(scale_new_shape))
        expect_ddy = np.array([[[[-1.16491104e+02, 1.57302490e+02, 4.94995819e+02]],
                                [[-2.27170822e+02, 2.83793610e+02, 8.96811523e+01]]],
                               [[[1.36491104e+02, 4.73530273e+02, 3.52587952e+02]],
                                [[-1.32302490e+02, 3.08113918e+01, 2.28437531e+02]]]]).astype(np_type)\
            .reshape(x_new_shape)
        expect_dx = np.array([[[[3.16227760e+01, 9.48683319e+01, 7.59508545e+02]],
                               [[3.16227760e+01, 1.58113876e+02, 7.12039261e+01]]],
                              [[[3.16227760e+01, 9.48683319e+01, 7.59508545e+02]],
                               [[3.16227760e+01, 1.58113876e+02, 1.18673210e+02]]]]).astype(np_type)\
            .reshape(x_new_shape)
        expect_dscale = np.array([6.95701065e+01, 9.48683319e+01, 2.39354736e+03]).astype(np_type_fp32)\
            .reshape(scale_new_shape)
        assert np.allclose(dx.asnumpy(), expect_dx)
        assert np.allclose(ddy.asnumpy(), expect_ddy)
        assert np.allclose(dscale.asnumpy(), expect_dscale)
