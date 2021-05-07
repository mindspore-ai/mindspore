# Copyright 2021 Huawei Technologies Co., Ltd
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
from functools import reduce
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Tensor

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_avgpool_k2s1pv():
    x = np.arange(1 * 1 * 6 * 6).reshape((1, 1, 6, 6)).astype(np.float32)
    net = nn.AvgPool2d(kernel_size=2, stride=1, pad_mode='valid')
    out = net(Tensor(x))
    print(out)
    expect_result = np.array(
        [[[[3.5, 4.5, 5.5, 6.5, 7.5],
           [9.5, 10.5, 11.5, 12.5, 13.5],
           [15.5, 16.5, 17.5, 18.5, 19.5],
           [21.5, 22.5, 23.5, 24.5, 25.5],
           [27.5, 28.5, 29.5, 30.5, 31.5]]]]
    )
    assert np.allclose(out.asnumpy(), expect_result)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_avgpool_k2s2pv():
    x = np.arange(1 * 1 * 6 * 6).reshape((1, 1, 6, 6)).astype(np.float32)
    net = nn.AvgPool2d(kernel_size=2, stride=2, pad_mode='valid')
    out = net(Tensor(x))
    print(out)
    expect_result = np.array(
        [[[[3.5, 5.5, 7.5],
           [15.5, 17.5, 19.5],
           [27.5, 29.5, 31.5]]]]
    )
    assert np.allclose(out.asnumpy(), expect_result)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_avgpool_k3s2pv():
    x = np.arange(1 * 1 * 6 * 6).reshape((1, 1, 6, 6)).astype(np.float32)
    net = nn.AvgPool2d(kernel_size=3, stride=2, pad_mode='valid')
    out = net(Tensor(x))
    print(out)
    expect_result = np.array(
        [[[[7., 9.],
           [19., 21.]]]]
    )
    assert np.allclose(out.asnumpy(), expect_result)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_avgpool_k3s2ps():
    x = np.arange(1 * 1 * 6 * 6).reshape((1, 1, 6, 6)).astype(np.float32)
    net = nn.AvgPool2d(kernel_size=3, stride=2, pad_mode='same')
    out = net(Tensor(x))
    print(out)
    expect_result = np.array(
        [[[[7., 9., 10.5],
           [19., 21., 22.5],
           [28., 30., 31.5]]]]
    )
    assert np.allclose(out.asnumpy(), expect_result)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_avg_pool3d_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_shape = (2, 3, 2, 3, 4)
    kernel_size = (2, 2, 3)
    strides = 1
    pad_mode = 'VALID'
    x_val = np.arange(reduce(lambda x, y: x * y, x_shape))
    x_ms = Tensor(x_val).reshape(x_shape).astype(np.float32)
    output_ms = P.AvgPool3D(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode)(x_ms)
    expert_result = (np.array([[[[[9, 10],
                                  [13, 14]]],
                                [[[33, 34],
                                  [37, 38]]],
                                [[[57, 58],
                                  [61, 62]]]],
                               [[[[81, 82],
                                  [85, 86]]],
                                [[[105, 106],
                                  [109, 110]]],
                                [[[129, 130],
                                  [133, 134]]]]]))
    assert (output_ms.asnumpy() == expert_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_avg_pool3d_2():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_shape = (2, 3, 2, 3, 4)
    kernel_size = 2
    strides = 1
    pad_mode = 'VALID'
    x_val = np.arange(reduce(lambda x, y: x * y, x_shape))
    x_ms = Tensor(x_val).reshape(x_shape).astype(np.float32)
    output_ms = P.AvgPool3D(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode)(x_ms)
    expert_result = (np.array([[[[[8.5, 9.5, 10.5],
                                  [12.5, 13.5, 14.5]]],
                                [[[32.5, 33.5, 34.5],
                                  [36.5, 37.5, 38.5]]],
                                [[[56.5, 57.5, 58.5],
                                  [60.5, 61.5, 62.5]]]],
                               [[[[80.5, 81.5, 82.5],
                                  [84.5, 85.5, 86.5]]],
                                [[[104.5, 105.5, 106.5],
                                  [108.5, 109.5, 110.5]]],
                                [[[128.5, 129.5, 130.5],
                                  [132.5, 133.5, 134.5]]]]]))
    assert (output_ms.asnumpy() == expert_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_avg_pool3d_3():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_shape = (2, 3, 2, 3, 4)
    kernel_size = 2
    strides = 3
    pad_mode = 'VALID'
    x_val = np.arange(reduce(lambda x, y: x * y, x_shape))
    x_ms = Tensor(x_val).reshape(x_shape).astype(np.float32)
    output_ms = P.AvgPool3D(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode)(x_ms)
    expert_result = (np.array([[[[[8.5]]],
                                [[[32.5]]],
                                [[[56.5]]]],
                               [[[[80.5]]],
                                [[[104.5]]],
                                [[[128.5]]]]]))
    assert (output_ms.asnumpy() == expert_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_avg_pool3d_4():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_shape = (2, 3, 2, 3, 4)
    kernel_size = (2, 2, 3)
    strides = 1
    pad_mode = 'SAME'
    x_val = np.arange(reduce(lambda x, y: x * y, x_shape))
    x_ms = Tensor(x_val).reshape(x_shape).astype(np.float32)
    output_ms = P.AvgPool3D(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode)(x_ms)
    expert_result = (np.array([[[[[8.5, 9, 10, 10.5],
                                  [12.5, 13, 14, 14.5],
                                  [14.5, 15, 16, 16.5]],
                                 [[14.5, 15, 16, 16.5],
                                  [18.5, 19, 20, 20.5],
                                  [20.5, 21, 22, 22.5]]],
                                [[[32.5, 33, 34, 34.5],
                                  [36.5, 37, 38, 38.5],
                                  [38.5, 39, 40, 40.5]],
                                 [[38.5, 39, 40, 40.5],
                                  [42.5, 43, 44, 44.5],
                                  [44.5, 45, 46, 46.5]]],
                                [[[56.5, 57, 58, 58.5],
                                  [60.5, 61, 62, 62.5],
                                  [62.5, 63, 64, 64.5]],
                                 [[62.5, 63, 64, 64.5],
                                  [66.5, 67, 68, 68.5],
                                  [68.5, 69, 70, 70.5]]]],
                               [[[[80.5, 81, 82, 82.5],
                                  [84.5, 85, 86, 86.5],
                                  [86.5, 87, 88, 88.5]],
                                 [[86.5, 87, 88, 88.5],
                                  [90.5, 91, 92, 92.5],
                                  [92.5, 93, 94, 94.5]]],
                                [[[104.5, 105, 106, 106.5],
                                  [108.5, 109, 110, 110.5],
                                  [110.5, 111, 112, 112.5]],
                                 [[110.5, 111, 112, 112.5],
                                  [114.5, 115, 116, 116.5],
                                  [116.5, 117, 118, 118.5]]],
                                [[[128.5, 129, 130, 130.5],
                                  [132.5, 133, 134, 134.5],
                                  [134.5, 135, 136, 136.5]],
                                 [[134.5, 135, 136, 136.5],
                                  [138.5, 139, 140, 140.5],
                                  [140.5, 141, 142, 142.5]]]]]))
    assert (output_ms.asnumpy() == expert_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_avg_pool3d_5():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_shape = (2, 3, 2, 3, 4)
    kernel_size = (2, 2, 3)
    strides = 1
    pad_mode = 'SAME'
    x_val = np.arange(reduce(lambda x, y: x * y, x_shape))
    x_ms = Tensor(x_val).reshape(x_shape).astype(np.float32)
    output_ms = P.AvgPool3D(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode)(x_ms)
    expert_result = (np.array([[[[[8.5, 9, 10, 10.5],
                                  [12.5, 13, 14, 14.5],
                                  [14.5, 15, 16, 16.5]],
                                 [[14.5, 15, 16, 16.5],
                                  [18.5, 19, 20, 20.5],
                                  [20.5, 21, 22, 22.5]]],
                                [[[32.5, 33, 34, 34.5],
                                  [36.5, 37, 38, 38.5],
                                  [38.5, 39, 40, 40.5]],
                                 [[38.5, 39, 40, 40.5],
                                  [42.5, 43, 44, 44.5],
                                  [44.5, 45, 46, 46.5]]],
                                [[[56.5, 57, 58, 58.5],
                                  [60.5, 61, 62, 62.5],
                                  [62.5, 63, 64, 64.5]],
                                 [[62.5, 63, 64, 64.5],
                                  [66.5, 67, 68, 68.5],
                                  [68.5, 69, 70, 70.5]]]],
                               [[[[80.5, 81, 82, 82.5],
                                  [84.5, 85, 86, 86.5],
                                  [86.5, 87, 88, 88.5]],
                                 [[86.5, 87, 88, 88.5],
                                  [90.5, 91, 92, 92.5],
                                  [92.5, 93, 94, 94.5]]],
                                [[[104.5, 105, 106, 106.5],
                                  [108.5, 109, 110, 110.5],
                                  [110.5, 111, 112, 112.5]],
                                 [[110.5, 111, 112, 112.5],
                                  [114.5, 115, 116, 116.5],
                                  [116.5, 117, 118, 118.5]]],
                                [[[128.5, 129, 130, 130.5],
                                  [132.5, 133, 134, 134.5],
                                  [134.5, 135, 136, 136.5]],
                                 [[134.5, 135, 136, 136.5],
                                  [138.5, 139, 140, 140.5],
                                  [140.5, 141, 142, 142.5]]]]]))
    assert (output_ms.asnumpy() == expert_result).all()


if __name__ == '__main__':
    test_avgpool_k2s1pv()
    test_avgpool_k2s2pv()
    test_avgpool_k3s2pv()
    test_avgpool_k3s2ps()
