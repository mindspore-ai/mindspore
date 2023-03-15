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
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops


class Net(nn.Cell):
    def __init__(self, mode):
        super(Net, self).__init__()
        self.mode = mode

    def construct(self, x, size=None, scale_factor=None, align_corners=None, recompute_scale_factor=None):
        return ops.interpolate(x, size, scale_factor, self.mode, align_corners, recompute_scale_factor)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_interpolate_area_3d(mode):
    """
    Feature: interpolate
    Description: Verify the result of interpolate area mode
                1. 3D size
                2. 3D scale_factor
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net("area")

    # 1. 3D(1, 3, 5)
    input_3d = np.array([[[0.816117, 0.004037, 0.746452, 0.137449, 0.337593],
                          [0.970709, 0.558792, 0.053919, 0.734102, 0.432973],
                          [0.830186, 0.77753, 0.384094, 0.905231, 0.76362]]], dtype=np.float32)
    except_3d_1 = np.array([[[0.410077, 0.375244, 0.441951, 0.237521],
                             [0.76475, 0.306356, 0.394011, 0.583538],
                             [0.803858, 0.580812, 0.644662, 0.834426]]], dtype=np.float32)
    size = 4
    output_3d_1 = net(Tensor(input_3d), size=size)
    assert np.allclose(output_3d_1.asnumpy(), except_3d_1, atol=1e-3, rtol=1e-3)

    # 2. 3D(1, 3, 5) scale_factor=0.2
    except_3d_2 = np.array([[[0.40833],
                             [0.550099],
                             [0.732132]]], dtype=np.float32)
    scale_factor = 0.2
    output_3d_2 = net(Tensor(input_3d), scale_factor=scale_factor)
    assert np.allclose(output_3d_2.asnumpy(), except_3d_2, atol=1e-3, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_interpolate_area_4d(mode):
    """
    Feature: interpolate
    Description: Verify the result of interpolate area mode
                1. 4D size
                2. 4D scale_factor
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net("area")

    # 1. 4D(1, 3, 3, 5) size=(5, 8)
    input_4d = np.array([[[[0.4992, 0.743079, 0.570383, 0.942855, 0.833395],
                           [0.645754, 0.485126, 0.957497, 0.933144, 0.556276],
                           [0.298626, 0.594928, 0.150964, 0.447654, 0.267512],
                           [0.46943, 0.419558, 0.665492, 0.414906, 0.708427]]]], dtype=np.float32)
    except_4d_1 = np.array([[[[0.4992, 0.62114, 0.743079, 0.656731, 0.756619, 0.942855,
                               0.888125, 0.833395],
                              [0.572477, 0.59329, 0.614102, 0.689021, 0.85097, 0.937999,
                               0.816418, 0.694836],
                              [0.47219, 0.506109, 0.540027, 0.547129, 0.622315, 0.690399,
                               0.551147, 0.411894],
                              [0.384028, 0.445635, 0.507243, 0.457736, 0.419754, 0.43128,
                               0.459625, 0.48797],
                              [0.46943, 0.444494, 0.419558, 0.542525, 0.540199, 0.414906,
                               0.561666, 0.708427]]]], dtype=np.float32)
    size = (5, 8)
    output_4d_1 = net(Tensor(input_4d), size=size)
    assert np.allclose(output_4d_1.asnumpy(), except_4d_1, atol=1e-5, rtol=1e-5)

    # 2. 4D(1, 3, 3, 5) scale_factor=(1.5, 0.4)
    except_4d_2 = np.array([[[[0.604221, 0.782211],
                              [0.650173, 0.798925],
                              [0.696126, 0.815639],
                              [0.348173, 0.28871],
                              [0.433166, 0.442492],
                              [0.51816, 0.596275]]]], dtype=np.float32)
    scale_factor = (1.5, 0.4)
    output_4d_2 = net(Tensor(input_4d), scale_factor=scale_factor)
    assert np.allclose(output_4d_2.asnumpy(), except_4d_2, atol=1e-5, rtol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_interpolate_area_5d(mode):
    """
    Feature: interpolate
    Description: Verify the result of interpolate area mode
                1. 5D size
                2. 5D scale_factor
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net("area")

    # 1. 5D(1, 1, 1, 4, 5) size=(2, 4, 3)
    input_5d = np.array([[[[[0.4992, 0.743079, 0.570383, 0.942855, 0.833395],
                            [0.645754, 0.485126, 0.957497, 0.933144, 0.556276],
                            [0.298626, 0.594928, 0.150964, 0.447654, 0.267512],
                            [0.46943, 0.419558, 0.665492, 0.414906, 0.708427]]]]], dtype=np.float32)
    except_5d_1 = np.array([[[[[0.62114, 0.752106, 0.888125],
                               [0.56544, 0.791922, 0.74471],
                               [0.446777, 0.397849, 0.357583],
                               [0.444494, 0.499985, 0.561666]],

                              [[0.62114, 0.752106, 0.888125],
                               [0.56544, 0.791922, 0.74471],
                               [0.446777, 0.397849, 0.357583],
                               [0.444494, 0.499985, 0.561666]]]]], dtype=np.float32)
    size = (2, 4, 3)
    output_5d_1 = net(Tensor(input_5d), size=size)
    assert np.allclose(output_5d_1.asnumpy(),
                       except_5d_1, atol=1e-5, rtol=1e-5)

    # 2. 5D(1, 1, 1, 4, 5) scale_factor=(3, 0.4, 0.7)
    except_5d_2 = np.array([[[[[0.519463, 0.610466, 0.638021]],
                              [[0.519463, 0.610466, 0.638021]],
                              [[0.519463, 0.610466, 0.638021]]]]], dtype=np.float32)
    scale_factor = (3., 0.4, 0.7)
    output_5d_2 = net(Tensor(input_5d), scale_factor=scale_factor)
    assert np.allclose(output_5d_2.asnumpy(), except_5d_2, atol=1e-5, rtol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_interpolate_bicubic(mode):
    """
    Feature: interpolate
    Description: Verify the result of interpolate bicubic mode
                1. 4D size tf
                2. 4D size align_corners=True
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net("bicubic")
    input_4d = np.array([[[[0.003088, 0.313131, 0.481231, 0.326219, 0.190293],
                           [0.711616, 0.583990, 0.718121, 0.258823, 0.121847],
                           [0.781316, 0.591508, 0.858185, 0.091935, 0.444639]],
                          [[0.389884, 0.894497, 0.471427, 0.188708, 0.557449],
                           [0.998047, 0.380719, 0.570574, 0.722258, 0.997173],
                           [0.195751, 0.050744, 0.002008, 0.482685, 0.708559]]]],
                        dtype=np.float32)
    size = (5, 3)
    # 1. 4D size=(5, 3)
    except_4d_1 = np.array([[[[0.038226, 0.46334, 0.228343],
                              [0.287464, 0.558147, 0.186838],
                              [0.671836, 0.718121, 0.14377],
                              [0.729394, 0.819616, 0.255285],
                              [0.723466, 0.868763, 0.334474]],

                             [[0.522484, 0.463939, 0.409831],
                              [0.670862, 0.531739, 0.626711],
                              [0.821431, 0.570574, 0.926654],
                              [0.40312, 0.206133, 0.777062],
                              [0.107333, -0.040933, 0.642958]]]], dtype=np.float32)
    output_4d_1 = net(Tensor(input_4d), size=size)
    assert np.allclose(output_4d_1.asnumpy(),
                       except_4d_1, atol=1e-5, rtol=1e-5)
    # 2. 4D size=(5, 3), align_corners=True
    except_4d_2 = np.array([[[[0.003088, 0.481231, 0.190293],
                              [0.350818, 0.586545, 0.125808],
                              [0.711616, 0.718121, 0.121847],
                              [0.81289, 0.810361, 0.276826],
                              [0.781316, 0.858185, 0.444639]],

                             [[0.389884, 0.471427, 0.557449],
                              [0.769181, 0.574304, 0.804369],
                              [0.998047, 0.570574, 0.997173],
                              [0.653914, 0.295586, 0.89409],
                              [0.195751, 0.002008, 0.708559]]]], dtype=np.float32)
    output_4d_2 = net(Tensor(input_4d), size=size, align_corners=True)

    assert np.allclose(output_4d_2.asnumpy(), except_4d_2, atol=1e-5, rtol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_interpolate_linear(mode):
    """
    Feature: interpolate
    Description: Verify the result of interpolate linear mode
                1. 3D size
                2. 3D size align_corners=True
                3. 3D scale_factor recompute_scale_factor=True
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net("linear")

    # 1. 3D(1, 3, 5) size=8
    input_3d = np.array([[[0.784811, 0.008794, 0.52707, 0.821989, 0.88349],
                          [0.00115, 0.556748, 0.8774, 0.761826, 0.23384],
                          [0.842441, 0.853507, 0.294813, 0.375813, 0.399932]]], dtype=np.float32)
    except_3d_1 = np.array([[[0.784811, 0.445304, 0.041186, 0.365109, 0.619232, 0.803557, 0.856583, 0.88349],
                             [0.00115, 0.244224, 0.576789, 0.777196, 0.841283, 0.769049, 0.464834, 0.23384],
                             [0.842441, 0.847282, 0.818589, 0.469405, 0.320126, 0.370751, 0.38938, 0.399932]]],
                           dtype=np.float32)
    size = 8
    output_3d_1 = net(Tensor(input_3d), size=size)
    assert np.allclose(output_3d_1.asnumpy(), except_3d_1, atol=1e-5, rtol=1e-5)

    # 2. 3D(1, 3, 5) size=8 align_corners=True
    except_3d_2 = np.array([[[0.784811, 0.341373, 0.082833, 0.378991, 0.611333, 0.779858,
                              0.848347, 0.88349],
                             [0.00115, 0.318635, 0.602555, 0.785785, 0.844379, 0.778337,
                              0.535546, 0.23384],
                             [0.842441, 0.848764, 0.773694, 0.45444, 0.317956, 0.364242,
                              0.38615, 0.399932]]], dtype=np.float32)
    output_3d_2 = net(Tensor(input_3d), size=size, align_corners=True)
    assert np.allclose(output_3d_2.asnumpy(), except_3d_2, atol=1e-5, rtol=1e-5)

    # 3. scale_factor=0.7, recompute_scale_factor=True
    except_3d_3 = np.array([[[0.526139, 0.52707, 0.86299],
                             [0.186349, 0.8774, 0.409835],
                             [0.84613, 0.294813, 0.391892]]], dtype=np.float32)
    scale_factor = 0.7
    output_3d_3 = net(Tensor(input_3d), scale_factor=scale_factor, recompute_scale_factor=True)
    assert np.allclose(output_3d_3.asnumpy(), except_3d_3, atol=1e-5, rtol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_interpolate_bilinear(mode):
    """
    Feature: interpolate
    Description: Verify the result of interpolate bilinear mode
                1. 4D size
                2. 4D size align_corners=True
                3. 4D scale_factor recompute_scale_factor=True
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net("bilinear")
    input_4d = np.array([[[[0.003088, 0.313131, 0.481231, 0.326219, 0.190293],
                           [0.711616, 0.58399, 0.718121, 0.258823, 0.121847],
                           [0.781316, 0.591508, 0.858185, 0.091935, 0.444639]],
                          [[0.389884, 0.894497, 0.471427, 0.188708, 0.557449],
                           [0.998047, 0.380719, 0.570574, 0.722258, 0.997173],
                           [0.195751, 0.050744, 0.002008, 0.482685, 0.708559]]]],
                        dtype=np.float32)
    size = (5, 3)

    # 1. 4D size=(5, 3)
    except_4d_1 = np.array([[[[0.106436, 0.481231, 0.235602],
                              [0.331491, 0.575987, 0.208363],
                              [0.669074, 0.718121, 0.167506],
                              [0.698458, 0.802159, 0.263245],
                              [0.718047, 0.858185, 0.327071]],
                             [[0.558088, 0.471427, 0.434535],
                              [0.651761, 0.511086, 0.622935],
                              [0.792271, 0.570574, 0.905535],
                              [0.405358, 0.229434, 0.742174],
                              [0.147415, 0.002008, 0.633268]]]], dtype=np.float32)
    output_4d_1 = net(Tensor(input_4d), size=size)
    assert np.allclose(output_4d_1.asnumpy(), except_4d_1, atol=1e-5, rtol=1e-5)

    # 2. 4D size=(5, 3), align_corners=True
    except_4d_2 = np.array([[[[0.003088, 0.481231, 0.190293],
                              [0.357352, 0.599676, 0.15607],
                              [0.711616, 0.718121, 0.121847],
                              [0.746466, 0.788153, 0.283243],
                              [0.781316, 0.858185, 0.444639]],

                             [[0.389884, 0.471427, 0.557449],
                              [0.693965, 0.521001, 0.777311],
                              [0.998047, 0.570574, 0.997173],
                              [0.596899, 0.286291, 0.852866],
                              [0.195751, 0.002008, 0.708559]]]], dtype=np.float32)
    output_4d_2 = net(Tensor(input_4d), size=size, align_corners=True)
    assert np.allclose(output_4d_2.asnumpy(), except_4d_2, atol=1e-5, rtol=1e-5)

    # 3. scale_factor=(0.5, 1.5), recompute_scale_factor=True
    scale_factor = (0.5, 1.5)
    except_4d_3 = np.array([[[[0.711616, 0.638687, 0.622313, 0.718121, 0.390051, 0.200119,
                               0.121847]],

                             [[0.998047, 0.645288, 0.434963, 0.570574, 0.67892, 0.840079,
                               0.997173]]]], dtype=np.float32)
    output_4d_3 = net(Tensor(input_4d), scale_factor=scale_factor, recompute_scale_factor=True)
    assert np.allclose(output_4d_3.asnumpy(), except_4d_3, atol=1e-5, rtol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_interpolate_nearest_exact_3d(mode):
    """
    Feature: interpolate
    Description: Verify the result of interpolate nearest-exact mode
                1. 3D size
                2. 3D scale_factor recompute_scale_factor=True
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net("nearest-exact")

    # 1. 3D(1, 3, 5) size=8
    input_3d = np.array([[[0.816117, 0.004037, 0.746452, 0.137449, 0.337593],
                          [0.970709, 0.558792, 0.053919, 0.734102, 0.432973],
                          [0.830186, 0.77753, 0.384094, 0.905231, 0.76362]]], dtype=np.float32)
    except_3d_1 = np.array([[[0.816117, 0.816117, 0.004037, 0.746452, 0.746452, 0.137449,
                              0.337593, 0.337593],
                             [0.970709, 0.970709, 0.558792, 0.053919, 0.053919, 0.734102,
                              0.432973, 0.432973],
                             [0.830186, 0.830186, 0.77753, 0.384094, 0.384094, 0.905231,
                              0.76362, 0.76362]]], dtype=np.float32)
    size = 8
    output_3d_1 = net(Tensor(input_3d), size=size)
    assert np.allclose(output_3d_1.asnumpy(), except_3d_1, atol=1e-5, rtol=1e-5)

    # 2. 3D(1, 3, 5) scale_factor=0.7 recompute_scale_factor=True
    except_3d_2 = np.array([[[0.816117, 0.746452, 0.337593],
                             [0.970709, 0.053919, 0.432973],
                             [0.830186, 0.384094, 0.76362]]], dtype=np.float32)
    scale_factor = 0.7
    output_3d_2 = net(Tensor(input_3d), scale_factor=scale_factor, recompute_scale_factor=True)
    assert np.allclose(output_3d_2.asnumpy(), except_3d_2, atol=1e-5, rtol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_interpolate_nearest_exact_4d(mode):
    """
    Feature: interpolate
    Description: Verify the result of interpolate nearest-exact mode
    Expectation: success
    1. 4D size
    2. 4D scale_factor recompute_scale_factor=True
    """
    ms.set_context(mode=mode)
    net = Net("nearest-exact")

    # 1. 4D(1, 3, 3, 5) size=(5, 8)
    size = (5, 8)
    input_4d = np.array([[[[0.4992, 0.743079, 0.570383, 0.942855, 0.833395],
                           [0.645754, 0.485126, 0.957497, 0.933144, 0.556276],
                           [0.298626, 0.594928, 0.150964, 0.447654, 0.267512],
                           [0.46943, 0.419558, 0.665492, 0.414906, 0.708427]]]], dtype=np.float32)
    except_4d_1 = np.array([[[[0.4992, 0.4992, 0.743079, 0.570383, 0.570383, 0.942855,
                               0.833395, 0.833395],
                              [0.645754, 0.645754, 0.485126, 0.957497, 0.957497, 0.933144,
                               0.556276, 0.556276],
                              [0.298626, 0.298626, 0.594928, 0.150964, 0.150964, 0.447654,
                               0.267512, 0.267512],
                              [0.298626, 0.298626, 0.594928, 0.150964, 0.150964, 0.447654,
                               0.267512, 0.267512],
                              [0.46943, 0.46943, 0.419558, 0.665492, 0.665492, 0.414906,
                               0.708427, 0.708427]]]], dtype=np.float32)
    output_4d_1 = net(Tensor(input_4d), size=size)
    assert np.allclose(output_4d_1.asnumpy(), except_4d_1, atol=1e-5, rtol=1e-5)

    # 2. 4D(1, 3, 3, 5) scale_factor=(1.5, 0.4) recompute_scale_factor=True
    scale_factor = (1.5, 0.4)
    except_4d_2 = np.array([[[[0.743079, 0.942855],
                              [0.485126, 0.933144],
                              [0.485126, 0.933144],
                              [0.594928, 0.447654],
                              [0.419558, 0.414906],
                              [0.419558, 0.414906]]]], dtype=np.float32)
    output_4d_2 = net(Tensor(input_4d), scale_factor=scale_factor, recompute_scale_factor=True)
    assert np.allclose(output_4d_2.asnumpy(), except_4d_2, atol=1e-5, rtol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_interpolate_nearest_3d(mode):
    """
    Feature: interpolate
    Description: Verify the result of interpolate nearest mode
                1. 3D size
                2. 3D scale_factor recompute_scale_factor=True
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net("nearest")

    # 1. 3D(1, 3, 5) size=8
    size = 8
    input_3d = np.array([[[0.816117, 0.004037, 0.746452, 0.137449, 0.337593],
                          [0.970709, 0.558792, 0.053919, 0.734102, 0.432973],
                          [0.830186, 0.77753, 0.384094, 0.905231, 0.76362]]], dtype=np.float32)
    except_3d_1 = np.array([[[0.816117, 0.816117, 0.004037, 0.004037, 0.746452, 0.137449,
                              0.137449, 0.337593],
                             [0.970709, 0.970709, 0.558792, 0.558792, 0.053919, 0.734102,
                              0.734102, 0.432973],
                             [0.830186, 0.830186, 0.77753, 0.77753, 0.384094, 0.905231,
                              0.905231, 0.76362]]], dtype=np.float32)
    output_3d_1 = net(Tensor(input_3d), size=size)
    assert np.allclose(output_3d_1.asnumpy(), except_3d_1, atol=1e-5, rtol=1e-5)

    # 2. 3D(1, 3, 5) scale_factor=0.7 recompute_scale_factor=True
    except_3d_2 = np.array([[[0.816117, 0.004037, 0.137449],
                             [0.970709, 0.558792, 0.734102],
                             [0.830186, 0.77753, 0.905231]]], dtype=np.float32)
    scale_factor = 0.7
    output_3d_2 = net(Tensor(input_3d), scale_factor=scale_factor, recompute_scale_factor=True)
    assert np.allclose(output_3d_2.asnumpy(), except_3d_2, atol=1e-5, rtol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_interpolate_nearest_4d(mode):
    """
    Feature: interpolate
    Description: Verify the result of interpolate nearest mode
                1. 4D size
                2. 4D scale_factor recompute_scale_factor=True
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net("nearest")

    # 1. 4D(1, 3, 3, 5) size=(5, 8)
    size = (5, 8)
    input_4d = np.array([[[[0.4992, 0.743079, 0.570383, 0.942855, 0.833395],
                           [0.645754, 0.485126, 0.957497, 0.933144, 0.556276],
                           [0.298626, 0.594928, 0.150964, 0.447654, 0.267512],
                           [0.46943, 0.419558, 0.665492, 0.414906, 0.708427]]]], dtype=np.float32)
    except_4d_1 = np.array([[[[0.4992, 0.4992, 0.743079, 0.743079, 0.570383, 0.942855,
                               0.942855, 0.833395],
                              [0.4992, 0.4992, 0.743079, 0.743079, 0.570383, 0.942855,
                               0.942855, 0.833395],
                              [0.645754, 0.645754, 0.485126, 0.485126, 0.957497, 0.933144,
                               0.933144, 0.556276],
                              [0.298626, 0.298626, 0.594928, 0.594928, 0.150964, 0.447654,
                               0.447654, 0.267512],
                              [0.46943, 0.46943, 0.419558, 0.419558, 0.665492, 0.414906,
                               0.414906, 0.708427]]]], dtype=np.float32)
    output_4d_1 = net(Tensor(input_4d), size=size)
    assert np.allclose(output_4d_1.asnumpy(), except_4d_1, atol=1e-5, rtol=1e-5)

    # 2. 4D(1, 3, 3, 5) scale_factor=(1.5, 0.4) recompute_scale_factor=True
    except_4d_2 = np.array([[[[0.4992, 0.570383],
                              [0.4992, 0.570383],
                              [0.645754, 0.957497],
                              [0.298626, 0.150964],
                              [0.298626, 0.150964],
                              [0.46943, 0.665492]]]], dtype=np.float32)
    scale_factor = (1.5, 0.4)
    output_4d_2 = net(Tensor(input_4d), scale_factor=scale_factor, recompute_scale_factor=True)
    assert np.allclose(output_4d_2.asnumpy(), except_4d_2, atol=1e-5, rtol=1e-5)
