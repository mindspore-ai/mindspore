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
import numpy as np

import mindspore as ms
from mindspore import dtype as mstype
from mindspore import context, Tensor, Parameter, ops
from mindspore.nn import Cell
from mindspore.ops.auto_generate import GroupedMatmul
from parallel.utils.utils import ParallelValidator, compile_net

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

class GroupedMatmulNetSplit0Weight(Cell):
    def __init__(self, np_w0, np_w1, split_item=3, group_type=0, mul_stra=None, gmm_stra=None, relu_stra=None):
        super().__init__()
        self.w = [Parameter(ms.Tensor(np_w0), "w0"), Parameter(ms.Tensor(np_w1), "w1")]
        self.b = None
        self.scale = None
        self.offset = None
        self.antiquant_scale = None
        self.antiquant_offset = None

        self.mul = ops.Mul().shard(mul_stra)
        self.gmm = GroupedMatmul(split_item, group_type).shard(gmm_stra)
        self.relu1 = ops.ReLU().shard(relu_stra)
        self.relu2 = ops.ReLU().shard(relu_stra)

    def construct(self, x0, x1, group_list, one0, one1):
        x0 = self.mul(x0, one0)
        x1 = self.mul(x1, one1)
        x = [x0, x1]

        out = self.gmm(x, self.w, self.b, self.scale, self.offset, self.antiquant_scale, self.antiquant_offset,
                       group_list)

        out0 = self.relu1(out[0])
        out1 = self.relu2(out[1])
        out = [out0, out1]
        return out


def test_grouped_matmul_case0():
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)

    mp = 4
    mul_stra = ((1, 1), (1, 1))
    gmm_stra = (((1, 1),) * 2, ((1, mp),) * 2, ((),), ((),), ((),), ((),), ((),), ()) # x,w / b 4 quant + grouplist
    relu_stra = ((1, mp),)

    M0 = 16
    K0 = 256
    N0 = 128
    np_x0 = np.random.uniform(0.1, 2, size=[M0, K0]).astype(np.float16)
    np_w0 = np.random.uniform(0.1, 1, size=[K0, N0]).astype(np.float16)

    gmm_net = GroupedMatmulNetSplit0Weight(np_w0, np_w0, split_item=0, group_type=-1,
                                           mul_stra=mul_stra, gmm_stra=gmm_stra, relu_stra=relu_stra)
    # ms calculate
    x0 = ms.Tensor(np_x0)
    x1 = ms.Tensor(np_x0)
    group_list = None
    one0 = ms.Tensor(np.ones_like(np_x0).astype(np.float16))
    one1 = ms.Tensor(np.ones_like(np_x0).astype(np.float16))

    gmm_net.set_inputs(x0, x1, group_list, one0, one1)
    phase = compile_net(gmm_net, x0, x1, group_list, one0, one1)

    validator = ParallelValidator(gmm_net, phase)
    assert validator.check_parameter_shape('w0', [K0, N0/mp])
    assert validator.check_parameter_shape('w1', [K0, N0/mp])


def test_grouped_matmul_case1():
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)

    mp = 4
    mul_stra = ((1, mp), (1, mp))
    gmm_stra = (((1, mp),) * 2, ((mp, 1),) * 2, ((),), ((),), ((),), ((),), ((),), ()) # x,w / b 4 quant grouplist
    relu_stra = ((1, 1),)

    M0 = 16
    K0 = 256
    N0 = 128
    np_x0 = np.random.uniform(0.1, 2, size=[M0, K0]).astype(np.float16)
    np_w0 = np.random.uniform(0.1, 1, size=[K0, N0]).astype(np.float16)

    gmm_net = GroupedMatmulNetSplit0Weight(np_w0, np_w0, split_item=0, group_type=-1,
                                           mul_stra=mul_stra, gmm_stra=gmm_stra, relu_stra=relu_stra)
    # ms calculate
    x0 = ms.Tensor(np_x0)
    x1 = ms.Tensor(np_x0)
    group_list = None
    one0 = ms.Tensor(np.ones_like(np_x0).astype(np.float16))
    one1 = ms.Tensor(np.ones_like(np_x0).astype(np.float16))

    gmm_net.set_inputs(x0, x1, group_list, one0, one1)
    phase = compile_net(gmm_net, x0, x1, group_list, one0, one1)

    validator = ParallelValidator(gmm_net, phase)
    assert validator.check_parameter_shape('w0', [K0/mp, N0])
    assert validator.check_parameter_shape('w1', [K0/mp, N0])


class GroupedMatmulNetSplit3WeightBias(Cell):
    def __init__(self, np_w0, np_b0, split_item=3, group_type=0, mul_stra=None, gmm_stra=None, relu_stra=None):
        super().__init__()
        self.w = [Parameter(ms.Tensor(np_w0), "w0")]
        self.b = [Parameter(ms.Tensor(np_b0), "b0")]
        self.scale = None
        self.offset = None
        self.antiquant_scale = None
        self.antiquant_offset = None

        self.mul = ops.Mul().shard(mul_stra)
        self.gmm = GroupedMatmul(split_item, group_type).shard(gmm_stra)
        self.relu = ops.ReLU().shard(relu_stra)

    def construct(self, x, group_list, one):
        x = self.mul(x, one)
        x_list = [x]
        out = self.gmm(x_list, self.w, self.b, self.scale, self.offset, self.antiquant_scale, self.antiquant_offset,
                       group_list)
        out = self.relu(out[0])
        return out


def test_grouped_matmul_case2():
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)

    mp = 4
    mul_stra = ((1, 1), (1, 1))
    gmm_stra = (((1, 1),), ((1, 1, mp),), ((1, mp),), ((),), ((),), ((),), ((),), (1,)) # x, w, b, grouplist
    relu_stra = ((1, mp),)

    M0 = 32
    K0 = 256
    N0 = 128
    E0 = 8
    group_list_np = [1, 3, 10, 14, 18, 22, 24, M0]

    np_x0 = np.random.uniform(0.1, 2, size=[M0, K0]).astype(np.float16)
    np_w0 = np.random.uniform(0.1, 1, size=[E0, K0, N0]).astype(np.float16)
    np_b0 = np.random.uniform(0.1, 1, size=[E0, N0]).astype(np.float16)

    gmm_net = GroupedMatmulNetSplit3WeightBias(np_w0, np_b0, split_item=3, group_type=0,
                                               mul_stra=mul_stra, gmm_stra=gmm_stra, relu_stra=relu_stra)

    # ms calculate
    x = ms.Tensor(np_x0)
    group_list = ms.Tensor(group_list_np)
    one = ms.Tensor(np.ones_like(np_x0).astype(np.float16))


    gmm_net.set_inputs(x, group_list, one)
    phase = compile_net(gmm_net, x, group_list, one)

    validator = ParallelValidator(gmm_net, phase)
    assert validator.check_parameter_shape('w0', [E0, K0, N0/mp])
    assert validator.check_parameter_shape('b0', [E0, N0/mp])


def test_grouped_matmul_case3():
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)

    mp = 4
    mul_stra = ((1, mp), (1, mp))
    gmm_stra = (((1, mp),), ((1, mp, 1),), ((1, 1),), ((),), ((),), ((),), ((),), (1,)) # x, w, b, grouplist
    relu_stra = ((1, 1),)

    M0 = 32
    K0 = 256
    N0 = 128
    E0 = 8
    group_list_np = [1, 3, 10, 14, 18, 22, 24, M0]

    np_x0 = np.random.uniform(0.1, 2, size=[M0, K0]).astype(np.float16)
    np_w0 = np.random.uniform(0.1, 1, size=[E0, K0, N0]).astype(np.float16)
    np_b0 = np.random.uniform(0.1, 1, size=[E0, N0]).astype(np.float16)

    gmm_net = GroupedMatmulNetSplit3WeightBias(np_w0, np_b0, split_item=3, group_type=0,
                                               mul_stra=mul_stra, gmm_stra=gmm_stra, relu_stra=relu_stra)

    x = ms.Tensor(np_x0)
    group_list = ms.Tensor(group_list_np)
    one = ms.Tensor(np.ones_like(np_x0).astype(np.float16))

    gmm_net.set_inputs(x, group_list, one)
    phase = compile_net(gmm_net, x, group_list, one)

    validator = ParallelValidator(gmm_net, phase)
    assert validator.check_parameter_shape('w0', [E0, K0/mp, N0])
    assert validator.check_parameter_shape('b0', [E0, N0])


def test_grouped_matmul_case4():
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    dp = 2
    mp = 4
    mul_stra = ((dp, mp), (dp, mp))
    gmm_stra = (((mp, dp),), ((1, dp, 1),), ((1, 1),), ((),), ((),), ((),), ((),), (1,)) # x, w, b, grouplist
    relu_stra = ((1, 1),)

    M0 = 32
    K0 = 256
    N0 = 128
    E0 = 8
    group_list_np = [1, 3, 10, 14, 18, 22, 24, M0]

    np_x0 = np.random.uniform(0.1, 2, size=[M0, K0]).astype(np.float16)
    np_w0 = np.random.uniform(0.1, 1, size=[E0, K0, N0]).astype(np.float16)
    np_b0 = np.random.uniform(0.1, 1, size=[E0, N0]).astype(np.float16)

    gmm_net = GroupedMatmulNetSplit3WeightBias(np_w0, np_b0, split_item=3, group_type=0,
                                               mul_stra=mul_stra, gmm_stra=gmm_stra, relu_stra=relu_stra)

    x = ms.Tensor(np_x0)
    group_list = ms.Tensor(group_list_np)
    one = ms.Tensor(np.ones_like(np_x0).astype(np.float16))

    gmm_net.set_inputs(x, group_list, one)
    phase = compile_net(gmm_net, x, group_list, one)

    validator = ParallelValidator(gmm_net, phase)
    assert validator.check_parameter_shape('w0', [E0, K0/dp, N0])
    assert validator.check_parameter_shape('b0', [E0, N0])


class GroupedMatmulNetSplit3WeightBiasReshape(Cell):
    def __init__(self, np_w0, np_b0, split_item=3, group_type=0, mul_stra=None, gmm_stra=None, relu_stra=None,
                 is_reshape=False):
        super().__init__()
        self.w = [Parameter(ms.Tensor(np_w0), "w0")]
        self.b = [Parameter(ms.Tensor(np_b0), "b0")]
        self.scale = None
        self.offset = None
        self.antiquant_scale = None
        self.antiquant_offset = None

        self.is_reshape = is_reshape
        self.mul = ops.Mul().shard(mul_stra)
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.gmm = GroupedMatmul(split_item, group_type).shard(gmm_stra)
        self.relu = ops.ReLU().shard(relu_stra)

    def construct(self, x, group_list, one):
        x = self.mul(x, one)
        if self.is_reshape:
            input_shape = self.shape(x)
            x = self.reshape(x, (-1, input_shape[2])) # 3d -> 2d
        x_list = [x]
        out = self.gmm(x_list, self.w, self.b, self.scale, self.offset, self.antiquant_scale, self.antiquant_offset,
                       group_list)
        out = self.relu(out[0].reshape(1, *out[0].shape)) # 2d -> 3d
        return out


def test_grouped_matmul_case5():
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)

    dp = 2
    mp = 4
    mul_stra = ((dp, 1, mp), (dp, 1, mp))
    gmm_stra = (((dp, mp),), ((1, mp, 1),), ((1, 1),), ((),), ((),), ((),), ((),), (1,)) # x, w, b, grouplist
    relu_stra = ((1, dp * mp, 1),)

    BS = 8
    M0 = 32
    K0 = 256
    N0 = 128
    E0 = 8
    group_list_np = [1, 3, 10, 14, 18, 22, 24, M0]

    np_x0 = np.random.uniform(0.1, 2, size=[BS, M0, K0]).astype(np.float16)
    np_w0 = np.random.uniform(0.1, 1, size=[E0, K0, N0]).astype(np.float16)
    np_b0 = np.random.uniform(0.1, 1, size=[E0, N0]).astype(np.float16)

    gmm_net = GroupedMatmulNetSplit3WeightBiasReshape(np_w0, np_b0, split_item=3, group_type=0,
                                                      mul_stra=mul_stra, gmm_stra=gmm_stra, relu_stra=relu_stra,
                                                      is_reshape=True)

    x = ms.Tensor(np_x0)
    group_list = ms.Tensor(group_list_np)
    one = ms.Tensor(np.ones_like(np_x0).astype(np.float16))

    gmm_net.set_inputs(x, group_list, one)
    phase = compile_net(gmm_net, x, group_list, one)

    validator = ParallelValidator(gmm_net, phase)
    assert validator.check_parameter_shape('w0', [E0, K0/mp, N0])
    assert validator.check_parameter_shape('b0', [E0, N0])
