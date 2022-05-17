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

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor


class NetDiag(nn.Cell):
    def __init__(self):
        super(NetDiag, self).__init__()
        self.diag = P.Diag()

    def construct(self, x):
        return self.diag(x)


class NetDiagWithDynamicShape(nn.Cell):
    def __init__(self):
        super(NetDiagWithDynamicShape, self).__init__()
        self.diag = P.Diag()
        self.unique = P.Unique()

    def construct(self, x):
        x, _ = self.unique(x)
        return self.diag(x)


def diag_1d(dtype):
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")

        x = Tensor(np.array([1, 2, 5]).astype(dtype))
        diag_1d_net = NetDiag()
        output = diag_1d_net(x)
        expect = np.array([[1, 0, 0],
                           [0, 2, 0],
                           [0, 0, 5]]).astype(dtype)
        assert (output.asnumpy() == expect).all()


def diag_2d(dtype):
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")

        x = Tensor(np.array([[1, 2, 3],
                             [4, 5, 6]]).astype(dtype))
        diag_2d_net = NetDiag()
        output = diag_2d_net(x)
        expect = np.array([[[[1, 0, 0],
                             [0, 0, 0]],
                            [[0, 2, 0],
                             [0, 0, 0]],
                            [[0, 0, 3],
                             [0, 0, 0]]],
                           [[[0, 0, 0],
                             [4, 0, 0]],
                            [[0, 0, 0],
                             [0, 5, 0]],
                            [[0, 0, 0],
                             [0, 0, 6]]]]).astype(dtype)
        assert (output.asnumpy() == expect).all()


def diag_with_dynamic_shape(dtype):
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")

        x = Tensor(np.array([1, 2, 5, 5, 2, 1]).astype(dtype))
        diag_with_dynamic_shape_net = NetDiagWithDynamicShape()
        output = diag_with_dynamic_shape_net(x)
        expect = np.array([[1, 0, 0],
                           [0, 2, 0],
                           [0, 0, 5]]).astype(dtype)
        assert (output.asnumpy() == expect).all()


@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_diag_1d_float16():
    """
    Feature: Diag op.
    Description: Test diag op with 1d and float16.
    Expectation: The value and shape of output are the expected values.
    """
    diag_1d(np.float16)


@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_diag_1d_float32():
    """
    Feature: Diag op.
    Description: Test diag op with 1d and float32.
    Expectation: The value and shape of output are the expected values.
    """
    diag_1d(np.float32)


@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_diag_2d_int32():
    """
    Feature: Diag op.
    Description: Test diag op with 2d and int32.
    Expectation: The value and shape of output are the expected values.
    """
    diag_2d(np.int32)


@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_diag_2d_int64():
    """
    Feature: Diag op.
    Description: Test diag op with 2d and int64.
    Expectation: The value and shape of output are the expected values.
    """
    diag_2d(np.int64)


@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_diag_with_dynamic_shape():
    """
    Feature: Diag op with dynamic shape.
    Description: Test diag op with unique.
    Expectation: The value and shape of output are the expected values.
    """
    diag_with_dynamic_shape(np.float32)
