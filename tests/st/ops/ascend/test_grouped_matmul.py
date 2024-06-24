# Copyright 2024 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore as ms
from mindspore import dtype as mstype
from mindspore import context, Tensor, ops
from mindspore.nn import Cell
from mindspore.ops.auto_generate import GroupedMatmul

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

def my_cmp(np1, np2, rtol=1e-3):
    print("np1.shape: ", np1.shape)
    print("np2.shape: ", np2.shape)
    print("max diff:  ", np.max(np1 - np2))
    diffidx = ~np.isclose(np1, np2, rtol=rtol)  # true is not close
    diffratio = np.around(diffidx.sum() / diffidx.size, 4)
    print("np1 diff num: ", np1[diffidx])
    print("np2 diff num: ", np2[diffidx])
    print("diff(", str(rtol), ") ratio: ", diffratio)


def get_empty_tensor(dtype=mstype.float32):
    x = Tensor([1], dtype)
    output = ops.slice(x, (0,), (0,))
    return output

def split_x(x, group_list):
    x_split = []
    for i in range(len(group_list)):
        if i == 0:
            x_split.append(x[0 : group_list[i],])
        else:
            x_split.append(x[group_list[i - 1] : group_list[i],])
    return x_split

def split_w(w):
    tmp_split = np.split(w, w.shape[0], axis=0)
    w_split = []
    for t in tmp_split:
        w_split.append(np.squeeze(t, 0))
    return w_split

class GroupedMatmulNet(Cell):
    def __init__(self, split_item=3, group_type=-1):
        super().__init__()
        self.gmm = GroupedMatmul(split_item, group_type)

    def construct(self, x, weight, bias=None, scale=None, offset=None, antiquant_scale=None, antiquant_offset=None,
                  group_list=None):
        out = self.gmm(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, group_list)
        return out

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_grouped_matmul_x2d_w2d_splititem0_grouptypeneg1_emptytensor_case0(mode):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend", mode=mode)
    gmm_net = GroupedMatmulNet(split_item=0, group_type=-1)

    # (16, 256) * (256, 128)   (127, 88) * (88, 64)
    M0 = 16
    K0 = 256
    N0 = 128

    M1 = 127
    K1 = 88
    N1 = 64

    # numpy calculate
    np_x0 = np.random.uniform(0.1, 2, size=[M0, K0]).astype(np.float16)
    np_w0 = np.random.uniform(0.1, 1, size=[K0, N0]).astype(np.float16)

    np_x1 = np.random.uniform(0.1, 2, size=[M1, K1]).astype(np.float16)
    np_w1 = np.random.uniform(0.1, 1, size=[K1, N1]).astype(np.float16)

    except0 = np.matmul(np_x0, np_w0)
    except1 = np.matmul(np_x1, np_w1)

    # ms calculate
    x = [ms.Tensor(np_x0), ms.Tensor(np_x1)]
    w = [ms.Tensor(np_w0), ms.Tensor(np_w1)]

    b = [get_empty_tensor(dtype=mstype.float16)]
    scale = [get_empty_tensor(dtype=mstype.uint64)]
    offset = [get_empty_tensor(dtype=mstype.float32)]
    antiquant_scale = [get_empty_tensor(dtype=mstype.float16)]
    antiquant_offset = [get_empty_tensor(dtype=mstype.float16)]

    res = gmm_net(x, w, b, scale, offset, antiquant_scale, antiquant_offset)

    # compare
    np.testing.assert_allclose(except0, res[0].asnumpy(), rtol=1e-3)
    np.testing.assert_allclose(except1, res[1].asnumpy(), rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_grouped_matmul_x2d_w2d_splititem0_grouptypeneg1_none_case1(mode):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend", mode=mode)
    gmm_net = GroupedMatmulNet(split_item=0, group_type=-1)

    # (16, 256) * (256, 128)   (127, 88) * (88, 64)
    M0 = 16
    K0 = 256
    N0 = 128

    M1 = 127
    K1 = 88
    N1 = 64

    # numpy calculate
    np_x0 = np.random.uniform(0.1, 2, size=[M0, K0]).astype(np.float16)
    np_w0 = np.random.uniform(0.1, 1, size=[K0, N0]).astype(np.float16)

    np_x1 = np.random.uniform(0.1, 2, size=[M1, K1]).astype(np.float16)
    np_w1 = np.random.uniform(0.1, 1, size=[K1, N1]).astype(np.float16)

    except0 = np.matmul(np_x0, np_w0)
    except1 = np.matmul(np_x1, np_w1)

    # ms calculate
    x = [ms.Tensor(np_x0), ms.Tensor(np_x1)]
    w = [ms.Tensor(np_w0), ms.Tensor(np_w1)]

    res = gmm_net(x, w)

    # compare
    np.testing.assert_allclose(except0, res[0].asnumpy(), rtol=1e-3)
    np.testing.assert_allclose(except1, res[1].asnumpy(), rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_grouped_matmul_x6d_w2d_splititem0_grouptypeneg1_none_case2(mode):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend", mode=mode)
    gmm_net = GroupedMatmulNet(split_item=0, group_type=-1)

    # (16, 256) * (256, 128)   (127, 88) * (88, 64)
    M0 = 16
    K0 = 256
    N0 = 128

    M1 = 127
    K1 = 88
    N1 = 64

    # numpy calculate
    np_x0 = np.random.uniform(0.1, 2, size=[2, 3, 4, 5, M0, K0]).astype(np.float16)
    np_w0 = np.random.uniform(0.1, 1, size=[K0, N0]).astype(np.float16)

    np_x1 = np.random.uniform(0.1, 2, size=[2, 3, 4, 5, M1, K1]).astype(np.float16)
    np_w1 = np.random.uniform(0.1, 1, size=[K1, N1]).astype(np.float16)

    except0 = np.matmul(np_x0, np_w0)
    except1 = np.matmul(np_x1, np_w1)

    # ms calculate
    x = [ms.Tensor(np_x0), ms.Tensor(np_x1)]
    w = [ms.Tensor(np_w0), ms.Tensor(np_w1)]

    res = gmm_net(x, w)

    # compare
    np.testing.assert_allclose(except0, res[0].asnumpy(), rtol=1e-3)
    np.testing.assert_allclose(except1, res[1].asnumpy(), rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_grouped_matmul_x2d_w2d_b1d_splititem0_grouptypeneg1_none_case3(mode):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend", mode=mode)
    gmm_net = GroupedMatmulNet(split_item=0, group_type=-1)

    # (16, 256) * (256, 128)   (127, 88) * (88, 64)
    M0 = 16
    K0 = 256
    N0 = 128

    M1 = 127
    K1 = 88
    N1 = 64

    # numpy calculate
    np_x0 = np.random.uniform(0.1, 2, size=[M0, K0]).astype(np.float16)
    np_w0 = np.random.uniform(0.1, 1, size=[K0, N0]).astype(np.float16)
    np_b0 = np.random.uniform(0.1, 1, size=[N0]).astype(np.float16)

    np_x1 = np.random.uniform(0.1, 2, size=[M1, K1]).astype(np.float16)
    np_w1 = np.random.uniform(0.1, 1, size=[K1, N1]).astype(np.float16)
    np_b1 = np.random.uniform(0.1, 1, size=[N1]).astype(np.float16)

    except0 = np.matmul(np_x0, np_w0) + np_b0
    except1 = np.matmul(np_x1, np_w1) + np_b1

    # ms calculate
    x = [ms.Tensor(np_x0), ms.Tensor(np_x1)]
    w = [ms.Tensor(np_w0), ms.Tensor(np_w1)]
    b = [ms.Tensor(np_b0), ms.Tensor(np_b1)]

    res = gmm_net(x, w, b)

    # compare
    np.testing.assert_allclose(except0, res[0].asnumpy(), rtol=1e-3)
    np.testing.assert_allclose(except1, res[1].asnumpy(), rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_grouped_matmul_x2d_w3d_splititem3_grouptype0_none_case4(mode):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend", mode=mode)
    gmm_net = GroupedMatmulNet(split_item=3, group_type=0)

    M0 = 32
    K0 = 256
    N0 = 128
    E0 = 8
    group_list_np = [1, 3, 10, 14, 18, 22, 24, M0]

    # numpy calculate
    np_x_all = np.random.uniform(0.1, 2, size=[M0, K0]).astype(np.float16)
    np_w_all = np.random.uniform(0.1, 1, size=[E0, K0, N0]).astype(np.float16)

    np_x = split_x(np_x_all, group_list_np) # use group_list split x. [(G0, N0), (G1, N0)....(GN, N0)]
    np_w = split_w(np_w_all)                # [(K0, N0), (K0, N0)....(K0, N0)]
    res_np = [np.matmul(x0, w0) for x0, w0 in zip(np_x, np_w)]
    except_np = np.concatenate(res_np, axis=0)

    # ms calculate
    x = [ms.Tensor(np_x_all)] # [M0, K0]
    w = [ms.Tensor(np_w_all)] # [E0, K0, N0]

    group_list = ms.Tensor(group_list_np, dtype=mstype.int64)

    res = gmm_net(x, w, group_list=group_list)

    np.testing.assert_allclose(except_np, res[0].asnumpy(), rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_grouped_matmul_x2d_w3d_b2d_splititem3_grouptype0_none_case5(mode):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend", mode=mode)
    gmm_net = GroupedMatmulNet(split_item=3, group_type=0)

    M0 = 32
    K0 = 256
    N0 = 128
    E0 = 8
    group_list_np = [1, 3, 10, 14, 18, 22, 24, M0]

    # numpy calculate
    np_x_all = np.random.uniform(0.1, 2, size=[M0, K0]).astype(np.float16)
    np_w_all = np.random.uniform(0.1, 1, size=[E0, K0, N0]).astype(np.float16)
    np_b_all = np.random.uniform(0.1, 1, size=[E0, N0]).astype(np.float16)

    np_x = split_x(np_x_all, group_list_np) # use group_list split x. [(G0, N0), (G1, N0)....(GN, N0)]
    np_w = split_w(np_w_all)                # [(K0, N0), (K0, N0)....(K0, N0)]
    np_b = split_w(np_b_all)

    res_np = [np.matmul(x0, w0) + b0 for x0, w0, b0 in zip(np_x, np_w, np_b)]
    except_np = np.concatenate(res_np, axis=0)

    # ms calculate
    x = [ms.Tensor(np_x_all)] # [M0, K0]
    w = [ms.Tensor(np_w_all)] # after cann update 0515, w should be a [3DTensor,]
    b = [ms.Tensor(np_b_all)]

    group_list = ms.Tensor(group_list_np, dtype=mstype.int64)

    res = gmm_net(x, w, b, group_list=group_list)

    np.testing.assert_allclose(except_np, res[0].asnumpy(), rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_grouped_matmul_x2d_w2d_splititem0_grouptypeneg1_none_a16w8_case6(mode):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend", mode=mode)
    gmm_net = GroupedMatmulNet(split_item=0, group_type=-1)

    # (16, 256) * (256, 128)   (127, 88) * (88, 64)
    M0 = 16
    K0 = 256
    N0 = 128

    M1 = 127
    K1 = 88
    N1 = 64

    # numpy calculate
    np_x0 = np.random.uniform(0.1, 0.5, size=[M0, K0]).astype(np.float16)
    np_w0 = np.random.uniform(-128, 127, size=[K0, N0]).astype(np.int8)
    antiquant_scale0 = np.array([0.02] * N0).astype(np.float16)
    antiquant_offset0 = np.array([-2] * N0).astype(np.float16)

    np_x1 = np.random.uniform(0.1, 1, size=[M1, K1]).astype(np.float16)
    np_w1 = np.random.uniform(-128, 127, size=[K1, N1]).astype(np.int8)
    antiquant_scale1 = np.array([0.01] * N1).astype(np.float16)
    antiquant_offset1 = np.array([1] * N1).astype(np.float16)

    except0 = np.matmul(np_x0, (np_w0 + antiquant_offset0) * antiquant_scale0)
    except1 = np.matmul(np_x1, (np_w1 + antiquant_offset1) * antiquant_scale1)

    # ms calculate
    x = [ms.Tensor(np_x0), ms.Tensor(np_x1)]
    w = [ms.Tensor(np_w0), ms.Tensor(np_w1)]

    b = None
    scale = None
    offset = None
    antiquant_scale = [ms.Tensor(antiquant_scale0), ms.Tensor(antiquant_scale1)]
    antiquant_offset = [ms.Tensor(antiquant_offset0), ms.Tensor(antiquant_offset1)]
    group_list = None

    res = gmm_net(x, w, b, scale, offset, antiquant_scale, antiquant_offset, group_list)

    # compare
    np.testing.assert_allclose(except0, res[0].asnumpy(), rtol=5e-3)
    np.testing.assert_allclose(except1, res[1].asnumpy(), rtol=5e-3)
