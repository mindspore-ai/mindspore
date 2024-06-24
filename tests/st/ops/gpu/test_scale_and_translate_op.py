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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
from mindspore import context, Tensor
import mindspore.ops.operations.image_ops as P
from mindspore import nn


class NetScaleAndTranslate(nn.Cell):
    def __init__(self, kernel_type_="lanczos3", antialias_=True):
        super(NetScaleAndTranslate, self).__init__()
        self.op = P.ScaleAndTranslate(kernel_type_, antialias_)

    def construct(self, images, size, scale, translation):
        return self.op(images, size, scale, translation)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scale_and_translate_lanczos3_true_graph_mode():
    """
    Feature: Test ScaleAndTranslate.
    Description: output type is float32.
    Expectation: Check it by expected_output variable.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    image_type = np.float32
    np.random.seed(1023)
    batch = 2
    image_height = 10
    image_width = 20
    channels = 2
    images = Tensor(np.random.randint(0, 14, size=(batch, image_height, image_width, channels)).astype(image_type))
    size = Tensor(np.array([2, 4]).astype(np.int32))
    scale = Tensor(np.array([3, 4]).astype(np.float32))
    translation = Tensor(np.array([0.4, 0.5]).astype(np.float32))
    net = NetScaleAndTranslate(kernel_type_="lanczos3", antialias_=True)
    output = net(images, size, scale, translation)
    output_ms = output.asnumpy()
    expected_output = np.array([[[[7.676014, 0.11800793],
                                  [7.912036, 1.0476434],
                                  [8.322707, 2.526446],
                                  [8.938111, 4.7001324]],
                                 [[6.5569315, 0.45438203],
                                  [6.901859, 1.3688041],
                                  [7.476031, 2.8254664],
                                  [8.330903, 4.966852]]],
                                [[[15.988619, 12.512353],
                                  [15.678796, 12.088711],
                                  [15.155482, 11.547304],
                                  [14.383627, 10.806888]],
                                 [[14.453416, 10.870245],
                                  [14.206803, 10.48943],
                                  [13.797294, 9.987571],
                                  [13.197551, 9.306632]]]]).astype(np.float32)
    error = np.ones(shape=expected_output.shape) * 1.0e-4
    diff = output_ms - expected_output
    assert np.all(abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scale_and_translate_lanczos1_true_pynative_mode():
    """
    Feature: Test ScaleAndTranslate.
    Description: output type is float32.
    Expectation: Check it by expected_output variable.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    image_type = np.float32
    np.random.seed(1023)
    batch = 2
    image_height = 10
    image_width = 20
    channels = 2
    images = Tensor(np.random.randint(0, 14, size=(batch, image_height, image_width, channels)).astype(image_type))
    size = Tensor(np.array([2, 4]).astype(np.int32))
    scale = Tensor(np.array([3, 4]).astype(np.float32))
    translation = Tensor(np.array([0.4, 0.5]).astype(np.float32))
    net = NetScaleAndTranslate(kernel_type_="lanczos1", antialias_=True)
    output = net(images, size, scale, translation)
    output_ms = output.asnumpy()
    expected_output = np.array([[[[7., 3.],
                                  [7., 3.],
                                  [7., 3.],
                                  [7.4, 3.9]],
                                 [[7., 3.],
                                  [7., 3.],
                                  [7., 3.],
                                  [7.4, 3.9]]],
                                [[[13., 9.],
                                  [13., 9.],
                                  [13., 9.],
                                  [12.800001, 8.6]],
                                 [[13., 9.],
                                  [13., 9.],
                                  [13., 9.],
                                  [12.800001, 8.6]]]]).astype(np.float32)
    error = np.ones(shape=expected_output.shape) * 1.0e-4
    diff = output_ms - expected_output
    assert np.all(abs(diff) < error)
