# Copyright 2019 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')


class Conv2dInput(nn.Cell):
    def __init__(self):
        super(Conv2dInput, self).__init__()
        out_channel = 1
        kernel_size = 3
        self.conv_input = P.Conv2DBackpropInput(out_channel,
                                                kernel_size,
                                                pad_mode="valid",
                                                pad=0,
                                                mode=1,
                                                stride=1,
                                                dilation=1,
                                                group=1)

        self.get_shape = P.Shape()

    @jit
    def construct(self, out, w, x):
        return self.conv_input(out, w, self.get_shape(x))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv2d_backprop_input():
    w = Tensor(np.array([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]]).astype(np.float32))
    x = Tensor(np.array([[[
        [3, 0, 1, 2, 7, 4],
        [1, 5, 8, 9, 3, 1],
        [2, 7, 2, 5, 1, 3],
        [0, 1, 3, 1, 7, 8],
        [4, 2, 1, 6, 2, 8],
        [2, 4, 5, 2, 3, 9]]]]).astype(np.float32))
    out = Tensor(np.array([[[
        [-5, -4, 0, 8],
        [-10, -2, 2, 3],
        [0, -2, -4, -7],
        [-3, -2, -3, -16]]]]).astype(np.float32))
    conv2d_input = Conv2dInput()
    output = conv2d_input(out, w, x)
    expect = np.array([[[[-5, -4, 5, 12, 0, -8],
                         [-15, -6, 17, 17, -2, -11],
                         [-15, -8, 13, 12, 2, -4],
                         [-13, -6, 8, -14, 5, 20],
                         [-3, -4, -4, -19, 7, 23],
                         [-3, -2, 0, -14, 3, 16]]]]).astype(np.float32)

    assert (abs(output.asnumpy() - expect) < np.ones(shape=[1, 1, 6, 6]) * 1.0e-4).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv2d_backprop_input_vmap():
    """
    Feature: Conv2DBackpropInput op
    Description: Test vmap rule for Conv2DBackpropInput op
    Expectation: The dataset is processed as expected
    """
    conv2d_input = Conv2dInput()

    batch_dout = Tensor(np.arange(1 * 2 * 1 * 4 * 4).reshape(1, 2, 1, 4, 4).astype(np.float32))
    x = Tensor(np.arange(1 * 1 * 3 * 3).reshape(1, 1, 3, 3).astype(np.float32))
    w = Tensor(np.ones([1, 1, 6, 6]).astype(np.float32))
    expected1 = np.array([[[[[0., 0., 1., 4., 7., 6.], [0., 7., 23., 38., 41., 29.],
                             [12., 45., 102., 138., 126., 81.], [48., 129., 246., 282., 234., 141.],
                             [84., 197., 341., 374., 287., 163.], [72., 162., 271., 292., 217., 120.]]]],
                          [[[[0., 16., 49., 52., 55., 38.], [48., 135., 263., 278., 233., 141.],
                             [156., 381., 678., 714., 558., 321.], [192., 465., 822., 858., 666., 381.],
                             [228., 517., 869., 902., 671., 371.], [168., 370., 607., 628., 457., 248.]]]]]
                        ).astype(np.float32)
    output1 = vmap(conv2d_input, (1, None, None))(batch_dout, x, w)
    assert np.allclose(output1.asnumpy(), expected1, 0.0001, 0.0001)

    dout = Tensor(np.arange(1 * 1 * 4 * 4).reshape(1, 1, 4, 4).astype(np.float32))
    batch_x = Tensor(np.arange(2 * 1 * 1 * 3 * 3).reshape(2, 1, 1, 3, 3).astype(np.float32))
    expected2 = np.array([[[[[0., 0., 1., 4., 7., 6.], [0., 7., 23., 38., 41., 29.],
                             [12., 45., 102., 138., 126., 81.], [48., 129., 246., 282., 234., 141.],
                             [84., 197., 341., 374., 287., 163.], [72., 162., 271., 292., 217., 120.]]]],
                          [[[[0., 9., 28., 58., 52., 33.], [36., 97., 185., 254., 203., 119.],
                             [120., 288., 507., 624., 477., 270.], [264., 588., 975., 1092., 801., 438.],
                             [264., 575., 935., 1022., 737., 397.], [180., 387., 622., 670., 478., 255.]]]]]
                        ).astype(np.float32)
    output2 = vmap(conv2d_input, (None, 0, None))(dout, batch_x, w)
    assert np.allclose(output2.asnumpy(), expected2, 0.0001, 0.0001)

    expected3 = np.array([[[[[0., 0., 1., 4., 7., 6.], [0., 7., 23., 38., 41., 29.],
                             [12., 45., 102., 138., 126., 81.], [48., 129., 246., 282., 234., 141.],
                             [84., 197., 341., 374., 287., 163.], [72., 162., 271., 292., 217., 120.]]]],
                          [[[[144., 313., 508., 538., 388., 209.], [372., 801., 1289., 1358., 971., 519.],
                             [696., 1488., 2379., 2496., 1773., 942.], [840., 1788., 2847., 2964., 2097., 1110.],
                             [696., 1471., 2327., 2414., 1697., 893.], [420., 883., 1390., 1438., 1006., 527.]]]]]
                        ).astype(np.float32)
    output3 = vmap(conv2d_input, (1, 0, None))(batch_dout, batch_x, w)
    assert np.allclose(output3.asnumpy(), expected3, 0.0001, 0.0001)
