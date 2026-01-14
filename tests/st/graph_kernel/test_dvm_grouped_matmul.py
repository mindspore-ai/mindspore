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
# ============================================================================
"""Test cases for GroupedMatmul operations."""

from mindspore.ops.auto_generate import GroupedMatmul
from mindspore import Tensor, nn, ops, context, mint
import mindspore as ms
import numpy as np
import pytest
from tests.mark_utils import arg_mark


class GroupedMatmulNetGroupType0(nn.Cell):
    """Neural network for testing GroupedMatmul with group type 0."""
    def __init__(self):
        super().__init__()
        self.gmm = GroupedMatmul(split_item=3, group_type=0)

    def construct(self, x, weight, bias, group_list):
        out = self.gmm([x], [weight], [bias], None, None, None, None, group_list)
        return out


class GroupedMatmulNetGroupType2(nn.Cell):
    """Neural network for testing GroupedMatmul with group type 2."""
    def __init__(self):
        super().__init__()
        self.gmm = GroupedMatmul(split_item=3, group_type=2)
        self.reshape = ops.Reshape()

    def construct(self, x, weight, group_list):
        x = mint.transpose(x, -1, -2)
        out = self.gmm([x], [weight], None, None, None, None, None, group_list)
        out_shape = out[0].shape
        new_shape = (out_shape[0], out_shape[2]*out_shape[1])
        res = self.reshape(out[0], new_shape)
        return [ops.cast(res, ms.float32) + 2]


class GroupedMatmulNetGroupListScalar(nn.Cell):
    """Neural network for testing GroupedMatmul with group type 0."""

    def __init__(self, group_list):
        super().__init__()
        self.group_list = group_list
        self.gmm = GroupedMatmul(split_item=3, group_type=0)

    def construct(self, x, weight, bias):
        out = self.gmm([x], [weight], [bias], None,
                       None, None, None, self.group_list)
        return out


def get_output(net, args, args_dyn=None, enable_graph_kernel=False):
    """Get output from network with optional dynamic shape and graph kernel settings."""
    if enable_graph_kernel:
        context.set_context(jit_config={"jit_level": "O1"})
        context.set_context(graph_kernel_flags="--enable_cluster_ops=GroupedMatmul,Reshape")
    else:
        context.set_context(jit_config={"jit_level": "O0"})
    if args_dyn:
        net.set_inputs(*args_dyn)
    output = net(*args)
    return output


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=20.01)
@pytest.mark.parametrize(
    "M0 ,K0, N0,E0, group_list_np",
    [(320, 256, 128, 10, [0, 10, 30, 100, 140, 180, 220, 220, 240, 320])],
)
def test_dvm_grouped_matmul_splititem3_grouptype0(M0, K0, N0, E0, group_list_np):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend")
    ms.set_context(mode=ms.GRAPH_MODE)

    np_x_all = np.random.uniform(0.1, 2, size=[M0, K0]).astype(np.float16)
    np_w_all = np.random.uniform(0.1, 1, size=[E0, K0, N0]).astype(np.float16)
    np_b_all = np.random.uniform(0.1, 1, size=[E0, N0]).astype(np.float16)

    x = ms.Tensor(np_x_all)
    w = ms.Tensor(np_w_all)
    b = ms.Tensor(np_b_all)

    group_list = ms.Tensor(group_list_np, dtype=ms.int64)

    expect = get_output(
        GroupedMatmulNetGroupType0(), [x, w, b, group_list], enable_graph_kernel=False
    )
    output = get_output(
        GroupedMatmulNetGroupType0(), [x, w, b, group_list], enable_graph_kernel=True
    )
    assert np.allclose(expect[0].asnumpy(), output[0].asnumpy(), 1e-3, 1e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=124.01
)
@pytest.mark.parametrize(
    "M0 ,K0, N0,E0, group_list_np",
    [(1024, 2560, 1024, 11, [0, 10, 300, 1024, 1024, 1400, 1800, 2048, 2048, 2200, 2560])],
)
def test_dvm_grouped_matmul_splititem3_grouptype2(M0, K0, N0, E0, group_list_np):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    np_x_all = np.random.uniform(0.1, 2, size=[K0, M0]).astype(np.float16)
    np_w_all = np.random.uniform(0.1, 1, size=[K0, N0]).astype(np.float16)

    x = ms.Tensor(np_x_all)
    w = ms.Tensor(np_w_all)

    group_list = ms.Tensor(group_list_np, dtype=ms.int64)

    expect = get_output(
        GroupedMatmulNetGroupType2(), [x, w, group_list], enable_graph_kernel=False
    )
    # Produce dirty data
    a = ms.Tensor(np.full([1024, 1024], np.nan, np.float16))
    b = ms.Tensor(np.full([1024, 1024], np.nan, np.float16))
    c = ms.ops.MatMul()(a, b)
    c.asnumpy()

    output = get_output(
        GroupedMatmulNetGroupType2(), [x, w, group_list], enable_graph_kernel=True
    )
    assert np.allclose(expect[0].asnumpy(), output[0].asnumpy(), 1e-3, 1e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=20.01
)
@pytest.mark.parametrize(
    "M0 ,K0, N0,E0, group_list_np",
    [(320, 256, 128, 12, [0, 10, 30, 30, 100, 140, 140, 180, 220, 220, 240, 320])],
)
def test_dvm_grouped_matmul_dyn_shape(M0, K0, N0, E0, group_list_np):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    np_x_all = np.random.uniform(0.1, 2, size=[M0, K0]).astype(np.float16)
    np_w_all = np.random.uniform(0.1, 1, size=[E0, K0, N0]).astype(np.float16)
    np_b_all = np.random.uniform(0.1, 1, size=[E0, N0]).astype(np.float16)

    x = ms.Tensor(np_x_all)
    w = ms.Tensor(np_w_all)
    b = ms.Tensor(np_b_all)

    group_list = ms.Tensor(group_list_np, dtype=ms.int64)
    args = [x, w, b, group_list]
    args_dyn = [
        Tensor(shape=(None, K0), dtype=ms.float16),
        Tensor(shape=(None, K0, None), dtype=ms.float16),
        Tensor(shape=(E0, None), dtype=ms.float16),
        group_list,
    ]
    expect = get_output(
        GroupedMatmulNetGroupType0(), args, args_dyn, enable_graph_kernel=False
    )
    output = get_output(
        GroupedMatmulNetGroupType0(), args, args_dyn, enable_graph_kernel=True
    )
    assert np.allclose(expect[0].asnumpy(), output[0].asnumpy(), 1e-3, 1e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
    mem_peak=0.01
)
@pytest.mark.parametrize(
    "M0, K0, N0, E0, group_list_np",
    [
        (320, 256, 128, 1, [320]),
        (320, 256, 128, 2, [256, 320]),
    ],
)
def test_dvm_grouped_matmul_grouplist_scalar(M0, K0, N0, E0, group_list_np):
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    np_x_all = np.random.uniform(0.1, 2, size=[M0, K0]).astype(np.float16)
    np_w_all = np.random.uniform(0.1, 1, size=[E0, K0, N0]).astype(np.float16)
    np_b_all = np.random.uniform(0.1, 1, size=[E0, N0]).astype(np.float16)

    x = ms.Tensor(np_x_all)
    w = ms.Tensor(np_w_all)
    b = ms.Tensor(np_b_all)

    group_list = ms.Tensor(group_list_np, dtype=ms.int64)
    args = [x, w, b]
    args_dyn = [
        Tensor(shape=(None, K0), dtype=ms.float16),
        Tensor(shape=(None, K0, None), dtype=ms.float16),
        Tensor(shape=(E0, None), dtype=ms.float16),
    ]
    expect = get_output(
        GroupedMatmulNetGroupListScalar(group_list), args, args_dyn, enable_graph_kernel=False
    )
    output = get_output(
        GroupedMatmulNetGroupListScalar(group_list), args, args_dyn, enable_graph_kernel=True
    )
    assert np.allclose(expect[0].asnumpy(), output[0].asnumpy(), 1e-3, 1e-3)
