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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class DataFormatVecPermuteNet(nn.Cell):

    def __init__(self, src_format, dst_format):
        super().__init__()
        self.op = P.nn_ops.DataFormatVecPermute(src_format, dst_format)

    def construct(self, x):
        return self.op(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_data_format_vec_permute_1d_input_int32():
    """
    Feature: DataFormatVecPermute gpu TEST.
    Description: 1d test case for DataFormatVecPermute, "NHWC" to "NCHW"
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x_ms = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
    net = DataFormatVecPermuteNet(src_format="NHWC", dst_format="NCHW")
    z_ms = net(x_ms)
    expect = np.array([1, 4, 2, 3]).astype(np.int32)

    assert (z_ms.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_data_format_vec_permute_2d_input_int64():
    """
    Feature: DataFormatVecPermute gpu TEST.
    Description: 2d test case for DataFormatVecPermute, "NCHW" to "NHWC"
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x_ms = Tensor(np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype(np.int64))
    net = DataFormatVecPermuteNet(src_format="NCHW", dst_format="NHWC")
    z_ms = net(x_ms)
    expect = np.array([[1, 1], [3, 3], [4, 4], [2, 2]]).astype(np.int64)

    assert (z_ms.asnumpy() == expect).all()
