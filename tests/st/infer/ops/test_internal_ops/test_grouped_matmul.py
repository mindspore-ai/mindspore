# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Tests for grouped matmul operator.
"""
import os
import numpy as np
import random
import pytest

import mindspore as ms
from mindspore import context
from mindspore.nn import Cell
from mindspore.ops.auto_generate import GroupedMatmul
from .st_utils import custom_compare
from tests.mark_utils import arg_mark

# GroupedMatmul has 8 inputs and 1 outputs
# -----------------Input-----------------
# 1.x:                   TensorList ((N, h), ) or ((bs, N, h), )
# 2.weight:              TensorList ((h, 4h)...(h, 4h)) or ((E, h, 4h))
# optional input
# 3.bias:                TensorList (empty_tensor,)
# 4.scale:               TensorList (empty_tensor,)
# 5.offset:              TensorList (empty_tensor,)
# 6.antiquant_scale:     TensorList (empty_tensor,)
# 7.antiquant_offset:    TensorList (empty_tensor,)
# 8.group_list:          Tensor
# -----------------Attr-----------------
# split_item:            int(0/1/2/3, current only support 0/3)
# ------------------------------
# y:                     TensorList ((N, 4h), ) or ((bs, N, 4h), )

np.set_printoptions(precision=2, suppress=True, linewidth=200)


def process_deq_scale(deq_scale) -> np.ndarray:
    new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
    return new_deq_scale.astype(np.int64)


def split_x(x, group_list):
    x_split = []
    for i, group_val in enumerate(group_list):
        if i == 0:
            x_split.append(x[0: group_val, ])
        else:
            x_split.append(x[group_list[i - 1]: group_val, ])
    return x_split


def split_w(w):
    tmp_split = np.split(w, w.shape[0], axis=0)
    w_split = []
    for t in tmp_split:
        w_split.append(np.squeeze(t, 0))
    return w_split


def np_qbmm_compute(a, b, tmp, bias=None):
    c = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.int32)
    if bias is not None:
        c = c + bias
    c = c.astype(np.float32) * tmp
    c = c.astype(np.float16)
    return c


class GroupedMatmulNet(Cell):
    def __init__(self, weight, bias=None, scale=None, offset=None, antiquant_scale=None, antiquant_offset=None,
                 split_item=3, group_type=0, trans_a=False, trans_b=False):
        super().__init__()
        self.gmm = GroupedMatmul(split_item, group_type, trans_a, trans_b)
        self.weight = ms.Parameter(weight, requires_grad=False)

    def construct(self, x, group_list=None):
        out = self.gmm(x, [self.weight], group_list=group_list)
        return out


class GroupedMatmulQuantNet(Cell):
    def __init__(self, weight, bias=None, scale=None, offset=None, antiquant_scale=None, antiquant_offset=None,
                 split_item=3, group_type=0, trans_a=False, trans_b=False):
        super().__init__()
        self.gmm = GroupedMatmul(split_item, group_type, trans_a, trans_b)
        self.weight = ms.Parameter(weight, requires_grad=False)
        self.scale = ms.Parameter(scale, requires_grad=False)
        self.bias = bias

    def construct(self, x, group_list=None):
        out = self.gmm(x, [self.weight], self.bias, [
                       self.scale], None, None, None, group_list)
        return out


def grouped_quant_matmul(m, k, n, e, group_list_np, trans_a=False, trans_b=False, profiling=False,
                         net_phase="increment", bias_none=False):
    os.environ['INTERNAL_PRINT_TILING'] = "on"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    context.set_context(device_target="Ascend")

    # numpy calculate
    seed = 0
    np.random.seed(seed)
    np_x_all = np.random.uniform(-20, 20, size=[m, k]).astype(np.int8)
    np.random.seed(seed)
    np_w_all = np.random.uniform(-20, 20, size=[e, k, n]).astype(np.int8)
    np.random.seed(seed)
    np_b_all = np.random.randint(-10, 10, (e, n)).astype(np.int32)
    np.random.seed(seed)
    np_s_all = np.random.rand(e, n).astype(np.float32) / 1000

    scale = process_deq_scale(np_s_all)

    # use group_list split x. [(G0, n), (G1, n)....(GN, n)]
    np_x = split_x(np_x_all, group_list_np)
    np_w = split_w(np_w_all)  # [(k, n), (k, n)....(k, n)]
    np_b = split_w(np_b_all)  # [(n), (n)....(n)]
    np_s = split_w(np_s_all)  # [(n), (n)....(n)]
    if bias_none:
        res_np = [np_qbmm_compute(x0, w0, s0)
                  for x0, w0, s0 in zip(np_x, np_w, np_s)]
    else:
        res_np = [np_qbmm_compute(x0, w0, s0, b0)
                  for x0, w0, s0, b0 in zip(np_x, np_w, np_s, np_b)]
    except_np = np.concatenate(res_np, axis=0)

    # ms calculate
    if trans_b:
        np_w_all = np.transpose(np_w_all, (0, 2, 1))
    x = [ms.Tensor(np_x_all)]  # [m, k]
    w = ms.Tensor(np_w_all)  # [e, k, n]
    s = ms.Tensor(scale, ms.int64)  # [e, n]

    # [e, n]
    if bias_none:
        b = None
    else:
        b = [ms.Parameter(ms.Tensor(np_b_all, ms.int32), requires_grad=False)]

    group_list = ms.Tensor(group_list_np, dtype=ms.int32)
    gmm_net = GroupedMatmulQuantNet(weight=w, bias=b, scale=s, offset=None, antiquant_scale=None,
                                    antiquant_offset=None, split_item=3, group_type=0, trans_a=trans_a,
                                    trans_b=trans_b)
    gmm_net.phase = net_phase

    if profiling:
        for _ in range(50):
            output = gmm_net(x, group_list=group_list)
        return

    output = gmm_net(x, group_list=group_list)
    output_fp32 = output[0].astype(ms.float32)
    output_np = output_fp32.asnumpy()
    res = custom_compare(except_np, output_np, ms.float16)
    assert res, "matmul compare fail."


def grouped_matmul(m, k, n, e, group_list_np, trans_a=False, trans_b=False, profiling=False, net_phase="increment"):
    os.environ['INTERNAL_PRINT_TILING'] = "on"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    context.set_context(device_target="Ascend")

    # numpy calculate
    np_x_all = np.random.uniform(0.1, 2, size=[m, k]).astype(np.float16)
    np_w_all = np.random.uniform(0.1, 1, size=[e, k, n]).astype(np.float16)
    # use group_list split x. [(G0, n), (G1, n)....(GN, n)]
    np_x = split_x(np_x_all, group_list_np)
    np_w = split_w(np_w_all)  # [(k, n), (k, n)....(k, n)]
    res_np = [np.matmul(x0, w0) for x0, w0 in zip(np_x, np_w)]
    except_np = np.concatenate(res_np, axis=0)

    # ms calculate
    if trans_b:
        np_w_all = np.transpose(np_w_all, (0, 2, 1))
    x = [ms.Tensor(np_x_all)]  # [m, k]
    w = ms.Tensor(np_w_all)  # [e, k, n]

    group_list = ms.Tensor(group_list_np, dtype=ms.int32)
    gmm_net = GroupedMatmulNet(weight=w, bias=None, scale=None, offset=None, antiquant_scale=None,
                               antiquant_offset=None, split_item=3, group_type=0, trans_a=trans_a,
                               trans_b=trans_b)
    gmm_net.phase = net_phase

    if profiling:
        for _ in range(50):
            output = gmm_net(x, group_list=group_list)
        return

    output = gmm_net(x, group_list=group_list)
    output_fp32 = output[0].astype(ms.float32)
    output_np = output_fp32.asnumpy()
    res = custom_compare(except_np, output_np, ms.float16)
    assert res, "matmul compare fail."


def generate_random_numbers(m, e):
    # 生成e-1个互不相同的随机数，范围是1到n，但不包括m
    random_numbers = random.choices(list(range(1, m+1)), k=e-1)
    # 将m添加到列表的末尾
    random_numbers.append(m)
    # 将列表从小到大排序
    random_numbers.sort()
    return random_numbers


@arg_mark(plat_marks=['platform_ascend310p'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('m', [32])
@pytest.mark.parametrize('k', [32])
@pytest.mark.parametrize('n', [64])
@pytest.mark.parametrize('e', [8])
def test_gmm_increment(m, k, n, e):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    group_list = np.array(generate_random_numbers(m, e)).astype(np.int32)
    #print("group_list = ", group_list)
    grouped_matmul(m, k, n, e, group_list, trans_b=True, net_phase="increment")
    grouped_quant_matmul(m, k, n, e, group_list,
                         trans_b=True, net_phase="increment")


@arg_mark(plat_marks=['platform_ascend310p'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('m', [1000])
@pytest.mark.parametrize('k', [256])
@pytest.mark.parametrize('n', [528])
@pytest.mark.parametrize('e', [8])
def test_gmm_prefill(m, k, n, e):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    group_list = np.array(generate_random_numbers(m, e)).astype(np.int32)
    #print("group_list = ", group_list)
    grouped_matmul(m, k, n, e, group_list, trans_b=True, net_phase="prefill")
    grouped_quant_matmul(m, k, n, e, group_list,
                         trans_b=True, net_phase="prefill")


@arg_mark(plat_marks=['platform_ascend310p'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('m', [32])
@pytest.mark.parametrize('k', [64])
@pytest.mark.parametrize('n', [128])
@pytest.mark.parametrize('e', [8])
def test_gmm_quant_with_bias(m, k, n, e):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    group_list = np.array(generate_random_numbers(m, e)).astype(np.int32)
    #print("group_list = ", group_list)
    grouped_quant_matmul(m, k, n, e, group_list, trans_b=True,
                         net_phase="increment", bias_none=True)
    grouped_quant_matmul(m, k, n, e, group_list, trans_b=True,
                         net_phase="prefill", bias_none=True)
