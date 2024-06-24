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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore

from mindspore import Tensor
from mindspore.ops.operations import _inner_ops as inner
import mindspore.nn as nn
import mindspore.context as context


class Net(nn.Cell):
    def __init__(self, ksizes, strides, rates, padding="valid"):
        super(Net, self).__init__()
        self.extractimagepatches = inner.ExtractImagePatches(ksizes, strides, rates, padding)

    def construct(self, input_tensor):
        return self.extractimagepatches(input_tensor)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_extract_image_patches_valid():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = Net([1, 1, 2, 4], [1, 1, 7, 5], [1, 1, 2, 1], "valid")
    input_tensor = Tensor(np.arange(360).reshape(3, 2, 6, 10).astype(np.float32))
    output = net(input_tensor)
    expect = np.array([0., 5., 60., 65., 1., 6., 61., 66., 2., 7., 62., 67., 3., 8.,
                       63., 68., 20., 25., 80., 85., 21., 26., 81., 86., 22., 27., 82., 87.,
                       23., 28., 83., 88., 120., 125., 180., 185., 121., 126., 181., 186., 122., 127.,
                       182., 187., 123., 128., 183., 188., 140., 145., 200., 205., 141., 146., 201., 206.,
                       142., 147., 202., 207., 143., 148., 203., 208., 240., 245., 300., 305., 241., 246.,
                       301., 306., 242., 247., 302., 307., 243., 248., 303., 308., 260., 265., 320., 325.,
                       261., 266., 321., 326., 262., 267., 322., 327., 263., 268., 323., 328.]).reshape((3, 16, 1, 2))
    assert np.all(output.asnumpy() == expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = Net([1, 1, 2, 4], [1, 1, 7, 5], [1, 1, 2, 1], "valid")
    input_tensor = Tensor(np.arange(360).reshape(3, 2, 6, 10).astype(np.float32))
    output = net(input_tensor)
    expect = np.array([0., 5., 60., 65., 1., 6., 61., 66., 2., 7., 62., 67., 3., 8.,
                       63., 68., 20., 25., 80., 85., 21., 26., 81., 86., 22., 27., 82., 87.,
                       23., 28., 83., 88., 120., 125., 180., 185., 121., 126., 181., 186., 122., 127.,
                       182., 187., 123., 128., 183., 188., 140., 145., 200., 205., 141., 146., 201., 206.,
                       142., 147., 202., 207., 143., 148., 203., 208., 240., 245., 300., 305., 241., 246.,
                       301., 306., 242., 247., 302., 307., 243., 248., 303., 308., 260., 265., 320., 325.,
                       261., 266., 321., 326., 262., 267., 322., 327., 263., 268., 323., 328.]).reshape((3, 16, 1, 2))
    assert np.all(output.asnumpy() == expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_extract_image_patches_same():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = Net([1, 1, 5, 2], [1, 1, 8, 2], [1, 1, 3, 3], "same")
    input_tensor = Tensor(np.arange(6).reshape(1, 1, 2, 3).astype(np.float32))
    output = net(input_tensor)
    expect = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 5., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0.]).reshape((1, 10, 1, 2))
    assert np.all(output.asnumpy() == expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = Net([1, 1, 5, 2], [1, 1, 8, 2], [1, 1, 3, 3], "same")
    input_tensor = Tensor(np.arange(6).reshape(1, 1, 2, 3).astype(np.float32))
    output = net(input_tensor)
    expect = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 5., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0.]).reshape((1, 10, 1, 2))
    assert np.all(output.asnumpy() == expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_extract_image_patches_dynamic_shape():
    """
    Feature: test dynamic shape of extract_image_patches
    Description: test dynamic shape of extract_image_patches
    Expectation: same to none dynamic
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = Net([1, 1, 5, 2], [1, 1, 8, 2], [1, 1, 3, 3], "same")
    input_tensor = Tensor(np.arange(6).reshape(1, 1, 2, 3).astype(np.float32))
    in_dyn = Tensor(shape=[1, 1, 2, None], dtype=mindspore.float32)
    net.set_inputs(in_dyn)
    output = net(input_tensor)
    expect = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 5., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0.]).reshape((1, 10, 1, 2))
    assert np.all(output.asnumpy() == expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = Net([1, 1, 5, 2], [1, 1, 8, 2], [1, 1, 3, 3], "same")
    input_tensor = Tensor(np.arange(6).reshape(1, 1, 2, 3).astype(np.float32))
    net.set_inputs(in_dyn)
    output = net(input_tensor)
    expect = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 5., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0.]).reshape((1, 10, 1, 2))
    assert np.all(output.asnumpy() == expect)
