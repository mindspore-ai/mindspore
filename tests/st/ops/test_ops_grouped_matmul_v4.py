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
import numpy as np
import pytest
import mindspore as ms
from mindspore import dtype as mstype
from mindspore import context, Tensor, ops
from mindspore.nn import Cell
from mindspore.ops.auto_generate import grouped_matmul_v4

from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


# GroupedMatmul has 8 inputs and 1 outputs
# -----------------Input-----------------
# 1.x:                         TensorList ((N, h), ) or ((bs, N, h), )
# 2.weight:                    TensorList ((h, 4h)...(h, 4h)) or ((E, h, 4h))
# optional input
# 3.bias:                      TensorList (empty_tensor,)
# 4.scale:                     TensorList (empty_tensor,)
# 5.offset:                    TensorList (empty_tensor,)
# 6.antiquant_scale:           TensorList (empty_tensor,)
# 7.antiquant_offset:          TensorList (empty_tensor,)
# 8.pre_token_scale:           TensorList (empty_tensor,)
# 9.group_list:                Tensor
# 10.activation_input:         TensorList (empty_tensor,)
# 11.activation_quant_scale:   TensorList (empty_tensor,)
# 12.activation_quant_offset:  TensorList (empty_tensor,)
# 13.split_item:               int(0/1/2/3, currently only support 0/3)
# 14.group_type:               int(-/0/1/2, currently only support -1/0)
# 15.group_list_type:          int(0/1)
# 16.act_type:                 int(0/1/2/3/4/5, currently not supported)
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
            x_split.append(x[0: group_list[i],])
        else:
            x_split.append(x[group_list[i - 1]: group_list[i],])
    return x_split


def split_w(w):
    tmp_split = np.split(w, w.shape[0], axis=0)
    w_split = []
    for t in tmp_split:
        w_split.append(np.squeeze(t, 0))
    return w_split


class GroupedMatmulV4Net(Cell):
    def __init__(self):
        super().__init__()
        self.gmm_v4 = grouped_matmul_v4

    def construct(self, x, weight, bias=None, scale=None, offset=None, antiquant_scale=None,
                  antiquant_offset=None, pertoken_scale=None, group_list=None, split_item=3,
                  group_type=-1, group_list_type=0, output_dtype=None):
        out = self.gmm_v4(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset,
                          pertoken_scale, group_list, split_item=split_item, group_type=group_type,
                          group_list_type=group_list_type, output_dtype=output_dtype)
        return out


@test_utils.run_with_cell
def grouped_matmul_v4_forward_func(x, weight, group_list):
    net = GroupedMatmulV4Net()
    out = net([x,], [weight,], group_list=group_list, split_item=3, group_type=0, group_list_type=1)
    return out[0]


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=26.0)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_grouped_matmul_v4_x2d_w2d_splititem0_grouptypeneg1_none(mode):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend")
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    gmm_v4_net = GroupedMatmulV4Net()

    split_item = 0
    group_type = -1

    M0 = 16
    K0 = 256
    N0 = 128

    M1 = 127
    K1 = 88
    N1 = 64

    # numpy calculate
    np_x0 = np.random.uniform(1, 2, size=[2, 3, 4, 5, M0, K0]).astype(np.float32)
    np_w0 = np.random.uniform(1, 2, size=[K0, N0]).astype(np.float32)
    np_b0 = np.random.uniform(1, 5, size=[N0]).astype(np.float32)

    np_x1 = np.random.uniform(1, 2, size=[2, 3, 4, 5, M1, K1]).astype(np.float32)
    np_w1 = np.random.uniform(1, 2, size=[K1, N1]).astype(np.float32)
    np_b1 = np.random.uniform(1, 5, size=[N1]).astype(np.float32)

    except0 = np.matmul(np_x0, np_w0) + np_b0
    except1 = np.matmul(np_x1, np_w1) + np_b1

    # ms calculate
    x = [ms.Tensor(np_x0, dtype=mstype.bfloat16), ms.Tensor(np_x1, dtype=mstype.bfloat16)]
    w = [ms.Tensor(np_w0, dtype=mstype.bfloat16), ms.Tensor(np_w1, dtype=mstype.bfloat16)]
    b = [ms.Tensor(np_b0), ms.Tensor(np_b1)]

    res = gmm_v4_net(x, w, b, split_item=split_item, group_type=group_type)

    # compare
    np.testing.assert_allclose(except0, res[0].float().asnumpy(), rtol=4e-3)
    np.testing.assert_allclose(except1, res[1].float().asnumpy(), rtol=4e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=20.0)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_grouped_matmul_v4_x2d_w3d_splititem3_grouptype0_a16w8(mode):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend")
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    gmm_v4_net = GroupedMatmulV4Net()

    split_item = 3
    group_type = 0
    group_list_type = 0

    M0 = 32
    K0 = 256
    N0 = 128
    E0 = 8
    group_list_np = [1, 3, 10, 14, 18, 22, 24, 30] # last value can be less than total token numbers

    # numpy calculate
    np_x_all = np.random.uniform(-128, 127, size=[M0, K0]).astype(np.float16)
    np_w_all = np.random.uniform(-128, 127, size=[E0, K0, N0]).astype(np.int8)
    antiquant_scale0 = np.array(np.full([E0, N0], 0.01)).astype(np.float16)
    antiquant_offset0 = np.array(np.full([E0, N0], 1)).astype(np.float16)

    np_x = split_x(np_x_all, group_list_np)
    np_w = split_w(np_w_all)
    np_s = split_w(antiquant_scale0)
    np_o = split_w(antiquant_offset0)
    res_np = [np.matmul(x0, (w0 + o0) * s0) for x0, w0, s0, o0 in zip(np_x, np_w, np_s, np_o)]
    except_np = np.concatenate(res_np, axis=0)

    # ms calculate
    x = [ms.Tensor(np_x_all)]
    w = [ms.Tensor(np_w_all)]
    antiquant_scale = [ms.Tensor(antiquant_scale0)]
    antiquant_offset = [ms.Tensor(antiquant_offset0)]

    b = None
    scale = None
    offset = None
    pertoken_scale = None
    group_list = ms.Tensor(group_list_np, dtype=mstype.int64)

    res = gmm_v4_net(x, w, b, scale, offset, antiquant_scale, antiquant_offset, pertoken_scale, group_list,
                     split_item, group_type, group_list_type)

    # compare
    np.testing.assert_allclose(except_np, res[0][:30].asnumpy(), rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=20.0)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_grouped_matmul_v4_x2d_w3d_splititem3_grouptype0_a16w4(mode):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend")
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    gmm_v4_net = GroupedMatmulV4Net()

    split_item = 3
    group_type = 0
    group_list_type = 0

    M0 = 32
    K0 = 256
    N0 = 128
    E0 = 8
    group_list_np = [1, 3, 10, 14, 18, 22, 24, 30] # last value can be less than total token numbers

    # numpy calculate
    np_x_all = np.random.uniform(-128, 127, size=[M0, K0]).astype(np.float16)
    np_w_all = np.random.uniform(0, 2, size=[E0, K0, N0]).astype(np.int8)
    antiquant_scale0 = np.array(np.full([E0, N0], 0.01)).astype(np.float16)
    antiquant_offset0 = np.array(np.full([E0, N0], 1)).astype(np.float16)

    for i in range(E0):
        for j in range(K0):
            for k in range(N0):
                np_w_all[i, j, k] = np_w_all[i, j, k] & 0xf

    np_w_all_int4 = np.ones((E0 * K0 * N0 // 2,), dtype=np.int8)
    np_w_all_one_rank = np_w_all.reshape(-1,)
    for i in range(E0 * K0 * N0 // 2):
        np_w_all_int4[i] = np_w_all_one_rank[i * 2] | ((np_w_all_one_rank[(i * 2) + 1] & 15) << 4)

    np_w_all_int4_3_rank = np_w_all_int4.reshape((E0, K0, N0 // 2))

    np_x = split_x(np_x_all, group_list_np)
    np_w = split_w(np_w_all)
    np_s = split_w(antiquant_scale0)
    np_o = split_w(antiquant_offset0)
    res_np = [np.matmul(x0, (w0 + o0) * s0) for x0, w0, s0, o0 in zip(np_x, np_w, np_s, np_o)]
    expect_np = np.concatenate(res_np, axis=0)

    # ms calculate
    x = [ms.Tensor(np_x_all)]
    w = [ms.Tensor(np_w_all_int4_3_rank, dtype=ms.qint4x2)]
    antiquant_scale = [ms.Tensor(antiquant_scale0)]
    antiquant_offset = [ms.Tensor(antiquant_offset0)]

    b = None
    scale = None
    offset = None
    pertoken_scale = None
    group_list = ms.Tensor(group_list_np, dtype=mstype.int64)

    res = gmm_v4_net(x, w, b, scale, offset, antiquant_scale, antiquant_offset, pertoken_scale, group_list,
                     split_item, group_type, group_list_type)

    # compare
    np.testing.assert_allclose(expect_np, res[0][:30].asnumpy(), rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=22.0)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_grouped_matmul_v4_x2d_w3d_splititem3_grouptype0_none_pertoken(mode):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend")
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    gmm_v4_net = GroupedMatmulV4Net()

    split_item = 3
    group_type = 0
    group_list_type = 1

    M0 = 32
    K0 = 256
    N0 = 128
    E0 = 8
    group_list_np = [1, 2, 7, 4, 4, 4, 2, 8]

    # numpy calculate
    np_x_all = np.random.uniform(-128, 127, size=[M0, K0]).astype(np.int8)
    np_w_all = np.random.uniform(-128, 127, size=[E0, K0, N0]).astype(np.int8)
    np_s_all = np.array(np.full([E0, N0], 10)).astype(np.float32)
    np_pts = np.array([10] * M0).astype(np.float32)

    np_x = split_x(np_x_all, np.cumsum(group_list_np))
    np_w = split_w(np_w_all)
    np_s = split_w(np_s_all)
    res_np = [np.matmul(x0, w0 * s0) for x0, w0, s0 in zip(np_x, np_w, np_s)]
    except_np = np.concatenate(res_np, axis=0) * np_pts.reshape(M0, 1)

    # ms calculate
    x = [ms.Tensor(np_x_all)]
    w = [ms.Tensor(np_w_all)]
    scale = [ms.Tensor(np_s_all, dtype=mstype.bfloat16)]
    pertoken_scale = [ms.Tensor(np_pts)]

    b = None
    offset = None
    antiquant_scale = None
    antiquant_offset = None
    group_list = ms.Tensor(group_list_np, dtype=mstype.int64)

    res = gmm_v4_net(x, w, b, scale, offset, antiquant_scale, antiquant_offset, pertoken_scale, group_list,
                     split_item, group_type, group_list_type)

    # compare
    np.testing.assert_allclose(except_np, res[0].float().asnumpy(), rtol=4e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=22.0)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_grouped_matmul_v4_x2d_w3d_splititem3_grouptype0_none_perchannel(mode):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend")
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    gmm_v4_net = GroupedMatmulV4Net()

    split_item = 3
    group_type = 0
    group_list_type = 1

    M0 = 32
    K0 = 256
    N0 = 128
    E0 = 8
    group_list_np = [1, 2, 7, 4, 4, 4, 2, 8]

    # numpy calculate
    np_x_all = np.random.uniform(-128, 127, size=[M0, K0]).astype(np.int8)
    np_w_all = np.random.uniform(-128, 127, size=[E0, K0, N0]).astype(np.int8)
    np_s_all = np.array(np.full([E0, N0], 10)).astype(np.float32)
    np_b_all = np.array(np.full([E0, N0], 1)).astype(np.float32)

    np_x = split_x(np_x_all, np.cumsum(group_list_np))
    np_w = split_w(np_w_all)
    np_s = split_w(np_s_all)
    np_b = split_w(np_b_all)
    res_np = [np.matmul(x0, w0 * s0) + b0 * s0 for x0, w0, s0, b0 in zip(np_x, np_w, np_s, np_b)]
    except_np = np.concatenate(res_np, axis=0)

    # ms calculate
    x = [ms.Tensor(np_x_all)]
    w = [ms.Tensor(np_w_all)]
    scale = [ms.Tensor(np_s_all, dtype=mstype.bfloat16)]
    bias = [ms.Tensor(np_b, dtype=mstype.int32)]

    offset = None
    antiquant_scale = None
    antiquant_offset = None
    group_list = ms.Tensor(group_list_np, dtype=mstype.int64)

    res = gmm_v4_net(x, w, bias, scale, offset, antiquant_scale, antiquant_offset, None, group_list,
                     split_item, group_type, group_list_type)

    # compare
    np.testing.assert_allclose(except_np, res[0].float().asnumpy(), rtol=4e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=20.01
)
def test_grouped_matmul_v4_dyn_shape():
    """
    Feature: Ops
    Description: test op GroupedMatmulV4 with group type 0
    Expectation: expect correct result.
    """
    context.set_context(runtime_num_threads=1)  # multi-threads have none-initialized bug now.

    m = 10
    k = 20
    n = 14
    group_list = Tensor([2, 4, 2, 2])
    x = Tensor(np.random.randn(m, k).astype(np.float32))
    w = Tensor(np.random.randn(group_list.shape[0], k, n).astype(np.float32))
    inputs_0 = [x, w, group_list]

    m = 20
    k = 30
    n = 8
    group_list = Tensor([2, 4, 2, 2, 4, 3, 3])
    x = Tensor(np.random.randn(m, k).astype(np.float32))
    w = Tensor(np.random.randn(group_list.shape[0], k, n).astype(np.float32))
    inputs_1 = [x, w, group_list]

    TEST_OP(
        grouped_matmul_v4_forward_func,
        [
            inputs_0,
            inputs_1,
        ],
        disable_mode=['GRAPH_MODE_GE'],
        disable_case=['DiscontiguousInput', 'EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True,
                     'disable_grad': True,
                     'deterministic_use_origin_inputs': True}
    )


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=20.0)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_ops_grouped_mamtul_v4_multi_dyn(mode):
    """
    Feature: pyboost function.
    Description: test GroupedMatmulV4 forward with dynamic rank/shape.
    Expectation: success.
    """
    context.set_context(device_target="Ascend")
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    gmm_v4_net = GroupedMatmulV4Net()

    split_item = 0
    group_type = -1
    group_list_type = 0

    x = ms.mutable([Tensor(shape=(None, None), dtype=mstype.float16), Tensor(shape=(None, None), dtype=mstype.float16)])
    weight = ms.mutable([Tensor(shape=(None, None), dtype=mstype.float16),
                         Tensor(shape=(None, None), dtype=mstype.float16)])
    gmm_v4_net.set_inputs(x, weight, None, None, None, None, None, None, None, split_item,
                          group_type, group_list_type, None)

    np_x0 = np.random.uniform(0.1, 2, size=[16, 256]).astype(np.float32)
    np_w0 = np.random.uniform(0.1, 1, size=[256, 128]).astype(np.float32)
    expect0 = np.matmul(np_x0, np_w0)

    np_x1 = np.random.uniform(0.1, 2, size=[127, 88]).astype(np.float32)
    np_w1 = np.random.uniform(0.1, 1, size=[88, 64]).astype(np.float32)
    expect1 = np.matmul(np_x1, np_w1)

    x1 = ms.mutable([ms.Tensor(np_x0, dtype=mstype.float16), ms.Tensor(np_x1, dtype=mstype.float16)])
    weight1 = ms.mutable([ms.Tensor(np_w0, dtype=mstype.float16), ms.Tensor(np_w1, dtype=mstype.float16)])
    res1 = gmm_v4_net(x1, weight1, split_item=split_item, group_type=group_type)
    np.testing.assert_allclose(expect0, res1[0].asnumpy(), rtol=1e-1)
    np.testing.assert_allclose(expect1, res1[1].asnumpy(), rtol=1e-1)

    x2 = ms.mutable([ms.Tensor(np_x0, dtype=mstype.float16), ms.Tensor(np_x1, dtype=mstype.float16)])
    weight2 = ms.mutable([ms.Tensor(np_w0, dtype=mstype.float16), ms.Tensor(np_w1, dtype=mstype.float16)])
    res2 = gmm_v4_net(x2, weight2, split_item=split_item, group_type=group_type)
    np.testing.assert_allclose(expect0, res2[0].asnumpy(), rtol=1e-1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=100.0)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
@pytest.mark.parametrize('output_dtype', [ms.float16, ms.bfloat16])
def test_ops_grouped_mamtul_v4_a8w4(mode, output_dtype):
    """
    Feature: pyboost function.
    Description: test GroupedMatmulV4 forward with a8w4.
    Expectation: success.
    """
    np.random.seed(1)
    context.set_context(device_target="Ascend")
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    gmm_v4_net = GroupedMatmulV4Net()

    E = 8
    M = 32
    K = 256
    N = 128
    split_item = 3
    group_type = 0
    group_list_type = 1

    x_np = np.random.randint(-5, 5, size=(M, K)).astype(np.int8)
    w_np = np.random.randint(-5, 5, size=(E, K, N)).astype(np.int8)
    w_int4_np = w_np.reshape(-1) & 0x000F
    w_int4_np = w_int4_np[0::2] | (w_int4_np[1::2] << 4)
    w_int4_np = w_int4_np.reshape(E, K, N // 2)
    scale_np = np.random.normal(0, 0.01, size=(E, 1, N)).astype(np.float32)
    scale_np_uint64 = np.frombuffer(scale_np.tobytes(), dtype=np.uint32).astype(np.uint64).reshape(E, 1, N)
    bias_np = 8 * (w_np.astype(np.float32) * scale_np).sum(axis=1)
    pertoken_scale_np = np.random.normal(0, 0.01, (M, 1)).astype(np.float32)
    group_list_np = np.array([1, 2, 7, 4, 4, 4, 2, 8], dtype=np.int64)

    index = np.cumsum(group_list_np)
    x_np_split = np.split(x_np, index, axis=0)
    pertoken_scale_np_split = np.split(pertoken_scale_np, index, axis=0)
    out_list = []
    scale_fp32 = scale_np_uint64.astype(np.uint32)
    scale_fp32.dtype = np.float32
    for i in range(E):
        mm = np.matmul(x_np_split[i].astype(np.int32), w_np[i].astype(np.int32)).astype(np.float32)
        mm = mm * scale_fp32[i] * pertoken_scale_np_split[i]
        out_list.append(mm)
    expect = np.concatenate(out_list, axis=0)

    x = [Tensor(x_np, ms.int8)]
    weight = [Tensor(w_int4_np, dtype=ms.qint4x2)]
    bias = [Tensor(bias_np, ms.float32)]
    scale = [Tensor(scale_np_uint64, ms.uint64)]
    pertoken_scale = [Tensor(pertoken_scale_np, ms.float32)]
    group_list = Tensor(group_list_np, ms.int64)
    out = gmm_v4_net(x, weight, bias=bias, scale=scale, pertoken_scale=pertoken_scale,
                     group_list=group_list, split_item=split_item, group_type=group_type,
                     group_list_type=group_list_type, output_dtype=output_dtype)[0]
    cnt = expect.shape[0]
    np.testing.assert_allclose(expect.astype(np.float32), out[:cnt].astype(ms.float32).asnumpy(), rtol=5e-3, atol=5e-3)
