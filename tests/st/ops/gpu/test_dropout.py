# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

import mindspore.nn as nn
from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner


class Net(nn.Cell):
    def __init__(self, keep_prob):
        super(Net, self).__init__()
        self.drop = P.Dropout(keep_prob)

    def construct(self, x_):
        return self.drop(x_)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dropout():
    """"
    Feature: Test dropout
    Description: Test gpu dropout operator
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_shape = [32, 16, 2, 5]
    x = np.ones(x_shape).astype(np.float32)
    keep_prob = 0.4
    dropout = Net(keep_prob)
    tx = Tensor(x)
    output, mask = dropout(tx)
    # check output
    output_np = output.asnumpy()
    elem_count = x.size
    nonzero_count = np.count_nonzero(output_np)
    assert (elem_count * (keep_prob - 0.1)) < nonzero_count < (elem_count * (keep_prob + 0.1))
    output_sum = np.sum(output_np)
    x_sum = np.sum(x)
    assert abs(output_sum - x_sum) / x_sum < 0.1
    # check mask
    mask_np = mask.asnumpy()
    mask_sum = np.sum(mask_np)
    assert np.count_nonzero(mask_np) == nonzero_count
    assert abs(mask_sum - nonzero_count) / nonzero_count < 0.1


class DropoutDynamic(nn.Cell):
    def __init__(self, keep_prob):
        super(DropoutDynamic, self).__init__()
        self.test_dynamic = inner.GpuConvertToDynamicShape()
        self.drop = P.Dropout(keep_prob)

    def construct(self, x):
        x = self.test_dynamic(x)
        return self.drop(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dropout_dynamic():
    """"
    Feature: Test dropout dynamic
    Description: Test gpu dropout supports dynamic shape
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_1 = np.ones([32, 16, 2, 5]).astype(np.float32)
    x_2 = np.ones([32, 16, 2, 5, 6]).astype(np.float32)
    keep_prob = 0.4
    net = DropoutDynamic(keep_prob)

    output_1, mask_1 = net(Tensor(x_1))
    elem_count_1 = x_1.size
    nonzero_count_1 = np.count_nonzero(output_1.asnumpy())
    assert (elem_count_1 * (keep_prob - 0.1)) < nonzero_count_1 < (elem_count_1 * (keep_prob + 0.1))
    output_sum_1 = np.sum(output_1.asnumpy())
    x_sum_1 = np.sum(x_1)
    assert abs(output_sum_1 - x_sum_1) / x_sum_1 < 0.1
    mask_sum_1 = np.sum(mask_1.asnumpy())
    assert np.count_nonzero(mask_1.asnumpy()) == nonzero_count_1
    assert abs(mask_sum_1 - nonzero_count_1) / nonzero_count_1 < 0.1

    output_2, mask_2 = net(Tensor(x_2))
    elem_count_2 = x_2.size
    nonzero_count_2 = np.count_nonzero(output_2.asnumpy())
    assert (elem_count_2 * (keep_prob - 0.1)) < nonzero_count_2 < (elem_count_2 * (keep_prob + 0.1))
    output_sum_2 = np.sum(output_2.asnumpy())
    x_sum_2 = np.sum(x_2)
    assert abs(output_sum_2 - x_sum_2) / x_sum_2 < 0.1
    mask_sum_2 = np.sum(mask_2.asnumpy())
    assert np.count_nonzero(mask_2.asnumpy()) == nonzero_count_2
    assert abs(mask_sum_2 - nonzero_count_2) / nonzero_count_2 < 0.1


class NetFilter(nn.Cell):
    def __init__(self, keep_prob, use_first_output):
        super(NetFilter, self).__init__()
        self.drop = P.Dropout(keep_prob)
        self.add = P.Add()
        self.use_first_output = use_first_output

    def construct(self, x_):
        output, mask = self.drop(x_)
        if self.use_first_output:
            return self.add(output, x_)
        return self.add(mask, x_)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dropout_attrs():
    """"
    Feature: Test dropout gpu optimization
    Description: Test only use first or second output of dropout
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_shape = [32, 16, 2, 5]
    x = np.ones(x_shape).astype(np.float32)
    x_sum = np.sum(x)
    keep_prob = 0.4
    tx = Tensor(x)

    # only use first output
    dropout = NetFilter(keep_prob, True)
    output = dropout(tx)
    output_np = output.asnumpy()
    output_sum = np.sum(output_np)
    assert abs(output_sum - x_sum) / x_sum < 1.1

    # only use second output
    dropout1 = NetFilter(keep_prob, False)
    mask = dropout1(tx)
    mask_sum = np.sum(mask.asnumpy())
    assert abs(mask_sum - x_sum) / x_sum < (keep_prob + 0.1)
