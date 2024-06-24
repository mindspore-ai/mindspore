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
from mindspore.ops.functional import vmap


class BceWithLogitsLossNet(nn.Cell):
    def __init__(self, reduction):
        super(BceWithLogitsLossNet, self).__init__()
        self.loss = P.BCEWithLogitsLoss(reduction=reduction)

    def construct(self, logits, label, weight, pos_weight):
        return self.loss(logits, label, weight, pos_weight)


class BceWithLogitsLossVMapNet(nn.Cell):
    def __init__(self, forward_net, in_axes, out_axes):
        super(BceWithLogitsLossVMapNet, self).__init__()
        self.net = forward_net
        self.in_axes = in_axes
        self.out_axes = out_axes

    def construct(self, logits, label, weight, pos_weight):
        return vmap(self.net, self.in_axes, self.out_axes)(logits,
                                                           label,
                                                           weight, pos_weight)


def bce_np_bencmark(data_type, reduction):
    """
    Feature: generate a BCEWithLogitsLoss numpy benchmark.
    Description: The benchmark generate by different data type.
    Expectation: match to np mindspore BCEWithLogitsLoss.
    """
    if reduction == "none":
        expected = np.array([[0.6111006, 0.5032824, 0.26318598],
                             [0.58439666, 0.55301523, -0.436814]]).astype(data_type)
    elif reduction == "mean":
        expected = 0.3463612
    else:
        expected = 2.0781672
    return expected


def get_dy_shape(real_shape):
    """
    Feature: generate a dynamic shape for mindspore dynamic shape
    Description: The shape only remain last dim of none.
    Expectation: match to mindspore BCEWithLogitsLoss input real shape.
    """
    part_shape_list = [shape for shape in real_shape]
    part_shape_list.pop()
    part_shape_list.append(None)
    return part_shape_list


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_bce_with_logits_loss(reduction, data_type):
    """
    Feature: test BCEWithLogitsLoss.
    Description: The output generate by different data type and reduction.
    Expectation: match to expected benchmark output.
    """
    context.set_context(mode=context.GRAPH_MODE)
    loss_net = BceWithLogitsLossNet(reduction)
    logits = Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]).astype(data_type))
    label = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]).astype(data_type))
    weight = Tensor(np.array([1.0, 1.0, 1.0]).astype(data_type))
    pos_weight = Tensor(np.array([1.0, 1.0, 1.0]).astype(data_type))
    loss = 1e-6
    if data_type == np.float16:
        loss = 1e-3
    benchmark = bce_np_bencmark(data_type, reduction)
    output = loss_net(logits, label, weight, pos_weight)
    np.testing.assert_allclose(benchmark, output.asnumpy(), rtol=loss, atol=loss)
    context.set_context(mode=context.PYNATIVE_MODE)
    output = loss_net(logits, label, weight, pos_weight)
    np.testing.assert_allclose(benchmark, output.asnumpy(), rtol=loss, atol=loss)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_bce_with_logits_loss_vmap_cpu(reduction, data_type):
    """
    Feature: test BCeWithLogitsLoss vmap on CPU.
    Description: inputs(logits, label, weight, pos_weight) with batch.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    loss_net = BceWithLogitsLossNet(reduction)
    logits = Tensor(
        np.array([[[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]],
                  [[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]]).astype(data_type))
    label = Tensor(np.array([[[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]],
                             [[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]]).astype(data_type))
    weight = Tensor(np.array([1.0, 1.0, 1.0]).astype(data_type))
    pos_weight = Tensor(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).astype(data_type))
    loss = 1e-4
    if data_type == np.float16:
        loss = 1e-3
    single_benchmark = bce_np_bencmark(data_type, reduction)
    batch_shape = np.shape(single_benchmark)
    if batch_shape is None:
        batch_shape = (2,)
    else:
        batch_shape = (2,) + batch_shape
    benchmark = np.broadcast_to(single_benchmark, batch_shape)
    in_axes = (0, 0, None, 0)
    out_axes = 0
    output = BceWithLogitsLossVMapNet(loss_net, in_axes, out_axes)(logits,
                                                                   label,
                                                                   weight,
                                                                   pos_weight)
    np.testing.assert_allclose(benchmark, output.asnumpy(), rtol=loss, atol=loss)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_bce_with_logits_loss_dy_shape(reduction, data_type):
    """
    Feature: Test BCEWithLogitsLoss DynamicShape.
    Description: The input data type only float16 and float32.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    loss_net = BceWithLogitsLossNet(reduction)
    logits = Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]).astype(data_type))
    label = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]).astype(data_type))
    weight = Tensor(np.array([1.0, 1.0, 1.0]).astype(data_type))
    pos_weight = Tensor(np.array([1.0, 1.0, 1.0]).astype(data_type))
    loss = 1e-4
    if data_type == np.float16:
        loss = 1e-3
    benchmark_output = bce_np_bencmark(data_type, reduction)

    logits_dyn = Tensor(shape=get_dy_shape(logits.shape), dtype=logits.dtype)
    label_dyn = Tensor(shape=get_dy_shape(label.shape), dtype=label.dtype)
    weight_dyn = Tensor(shape=get_dy_shape(weight.shape), dtype=weight.dtype)
    pos_weight_dyn = Tensor(shape=get_dy_shape(pos_weight.shape), dtype=pos_weight.dtype)
    loss_net.set_inputs(logits_dyn, label_dyn, weight_dyn, pos_weight_dyn)
    ms_result = loss_net(logits, label, weight, pos_weight)
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)
    context.set_context(mode=context.PYNATIVE_MODE)
    ms_result = loss_net(logits, label, weight, pos_weight)
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)
