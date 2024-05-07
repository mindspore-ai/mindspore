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
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.context import set_auto_parallel_context
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore.common.api import flops_collection
from mindspore.train import Model, FlopsUtilizationCollector
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from tests.ut.python.ops.test_math_ops import VirtualLoss

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


context.set_context(save_graphs=True)
grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, *inputs):
        return grad_all(self.network)(*inputs)


def compile_net(net, *inputs):
    net.set_train()
    _cell_graph_executor.compile(net, *inputs)


# model_parallel test
def test_matmul_flops():
    """
    Feature: same mode, stride < kernel_size, need exchange
    Description: split n/c-in/c-out/h
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy)

        def construct(self, x, y):
            out = self.matmul(x, y)
            return out

    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy = ((2, 2), (2, 2))
    net = GradWrap(NetWithLoss(Net(strategy)))
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    compile_net(net, x, y)
    full_mfu, full_hfu, shard_mfu, shard_hfu, _ = flops_collection(net.current_phase)
    print(f"full_mfu: {full_mfu}, full_hfu: {full_hfu}")
    # origin_shape is [128, 32] [32,128] -> [128, 128]
    assert full_mfu == 2 * 128 * 32 * 128 * 3
    # shard_shape is [64, 16] [16,64] -> [64, 64]
    print(f"shard_mfu: {shard_mfu}, shard_hfu: {shard_hfu}")
    assert shard_hfu == 2 * 64 * 16 * 64 * 3

# model_parallel test
def test_matmul_flops_dynamic_shape():
    """
    Feature: same mode, stride < kernel_size, need exchange
    Description: split n/c-in/c-out/h
    Expectation: compile success
    """
    context.reset_auto_parallel_context()
    class Net(nn.Cell):
        def __init__(self, strategy):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy)

        def construct(self, x, y):
            out = self.matmul(x, y)
            return out

    strategy = ((2, 2), (2, 2))
    net = GradWrap(NetWithLoss(Net(strategy)))
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    input_dyn = ms.Tensor(shape=[None, 32], dtype=ms.float32)
    net.set_inputs(input_dyn, y)
    compile_net(net, input_dyn, y)
    _, _, _, _, is_dynamic_shape = flops_collection(net.current_phase)
    assert is_dynamic_shape
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class DatasetLenet():
    def __init__(self, data, label, length=3):
        self.data = data
        self.label = label
        self.index = 1
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.data, self.label

    def reset(self):
        self.index = 0

    def get_dataset_size(self):
        return 32

    def get_repeat_count(self):
        return 1

    def get_batch_size(self):
        return 32

    def create_tuple_iterator(self, num_epochs=1, do_copy=True):
        return self


def test_matmul_with_call_back_flops():
    """
    Feature: same mode, stride < kernel_size, need exchange
    Description: split n/c-in/c-out/h
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy)
            self.param = Parameter(initializer(
                "zeros", [64, 64]), name="param")

        def construct(self, x):
            out = self.matmul(x, self.param)
            return out

    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy = ((2, 2), (2, 2))
    network = Net(strategy)
    net_loss = nn.BCELoss(weight=None, reduction='mean')
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)

    model = Model(network, loss_fn=net_loss,
                  optimizer=net_opt, metrics={'acc'})
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([32, 64]), dtype=ms.float32)
    ds_train = DatasetLenet(data, label, 3)
    model.train(5, ds_train, callbacks=[FlopsUtilizationCollector(
        ds_train.get_dataset_size())], dataset_sink_mode=False)

    auto_parallel_context().reset()
    network = Net(strategy)
    net_loss = nn.BCELoss(weight=None, reduction='mean')
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)

    model = Model(network, loss_fn=net_loss,
                  optimizer=net_opt, metrics={'acc'})
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([32, 64]), dtype=ms.float32)
    ds_train = DatasetLenet(data, label, 3)
    model.train(5, ds_train, callbacks=[FlopsUtilizationCollector(
        ds_train.get_dataset_size())], dataset_sink_mode=False)


def test_batch_matmul_flops():
    """
    Feature: same mode, stride < kernel_size, need exchange
    Description: split n/c-in/c-out/h
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy):
            super().__init__()
            self.batch_matmul = P.BatchMatMul().shard(strategy)

        def construct(self, x, y):
            out = self.batch_matmul(x, y)
            return out

    set_auto_parallel_context(device_num=16, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy = ((1, 2, 2, 2), (2, 2, 2))
    net = GradWrap(NetWithLoss(Net(strategy)))
    x = Tensor(np.ones([1, 2, 128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([2, 32, 128]), dtype=ms.float32)
    compile_net(net, x, y)
    full_mfu, full_hfu, shard_mfu, shard_hfu, _ = flops_collection(net.current_phase)
    print(f"full_mfu: {full_mfu}, full_hfu: {full_hfu}")
    # origin_shape is [2, 128, 32] [2, 32,128] -> [2, 128, 128]
    assert full_mfu == 2 * 2 * 128 * 32 * 128 * 3
    # shard_shape is [1, 64, 16] [1, 16,64] -> [1, 64, 64]
    print(f"shard_mfu: {shard_mfu}, shard_hfu: {shard_hfu}")
    assert shard_hfu == 2 * 1 * 64 * 16 * 64 * 3


def test_conv2d_kernel_size_larger_than_stride_and_split_h_flops():
    """
    Feature: same mode, stride < kernel_size, need exchange
    Description: split n/c-in/c-out/h
    Expectation: compile success
    """
    context.set_context(save_graphs=True)
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=32, global_rank=0)
    strategy1 = ((2, 2, 4, 1), (2, 2, 1, 1))
    strategy2 = ((2, 2, 4, 1),)

    class Net(nn.Cell):
        def __init__(self, out_channel, kernel_size, pad_mode, stride, dilation=1, group=1, pad=0,
                     strategy1=None, strategy2=None):
            super().__init__()
            self.conv2d = P.Conv2D(out_channel=out_channel, kernel_size=kernel_size, pad_mode=pad_mode, pad=pad,
                                   stride=stride, dilation=dilation, group=group).shard(strategy1)
            self.neg = P.Neg().shard(strategy2)

        def construct(self, x, b):
            out = self.conv2d(x, b)
            out = self.neg(out)
            return out

    net = Net(out_channel=8, kernel_size=3, pad_mode="same",
              stride=1, strategy1=strategy1, strategy2=strategy2)
    net = GradWrap(NetWithLoss(net))
    x = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)
    y = Tensor(np.ones([8, 16, 3, 3]), dtype=ms.float32)
    compile_net(net, x, y)
    full_mfu, full_hfu, shard_mfu, shard_hfu, _ = flops_collection(net.current_phase)
    print(f"full_mfu: {full_mfu}, full_hfu: {full_hfu}")
    full_N = 32
    full_C_IN = 16
    full_K0 = 3
    full_K1 = 3
    full_C_OUT = 8
    full_H_OUT = 8
    full_W_OUT = 8
    # origin_shape is [32, 16, 8, 8] [8, 16, 3, 3] -> [32, 8, 8, 8]
    assert full_mfu == 3 * full_N * 2 * full_C_IN * full_K0 * \
           full_K1 * full_H_OUT * full_W_OUT * full_C_OUT
    print(f"shard_mfu: {shard_mfu}, shard_hfu: {shard_hfu}")
    # shard_shape is [16, 8, 3, 8] [4, 8, 3, 3] -> [16, 4, 2, 8]
    assert shard_mfu == 3 * full_N // 2 * 2 * full_C_IN // 2 * full_K0 * \
           full_K1 * full_H_OUT // 4 * full_W_OUT // 1 * full_C_OUT // 2


def generate_inputs(B, N, S, D, input_layout, use_mqa=False, with_real_shift=True, sparse_mode=0):
    N_Q = N
    N_KV = 1 if use_mqa else N
    compressed_mask_mode = [2, 3, 4]
    if input_layout == "BSH":
        H_Q = N_Q * D
        H_KV = N_KV * D
        query = Tensor(np.ones((B, S, H_Q), dtype=np.float16))
        key = Tensor(np.ones((B, S, H_KV), dtype=np.float16))
        value = Tensor(np.ones((B, S, H_KV), dtype=np.float16))
    elif input_layout == "BNSD":
        query = Tensor(np.ones((B, N_Q, S, D), dtype=np.float16))
        key = Tensor(np.ones((B, N_KV, S, D), dtype=np.float16))
        value = Tensor(np.ones((B, N_KV, S, D), dtype=np.float16))
    else:
        raise ValueError(f"input_layout is invalid.")
    real_shift = Tensor(np.ones((B, N, S, S), dtype=np.float16)
                        ) if with_real_shift else None
    if sparse_mode not in compressed_mask_mode:
        attn_mask = Tensor(np.ones((B, 1, S, S), dtype=np.uint8))
    else:
        attn_mask = Tensor(np.ones((2048, 2048), dtype=np.uint8))
    return query, key, value, real_shift, attn_mask


@pytest.mark.parametrize('input_layout', ["BSH"])
@pytest.mark.parametrize('use_mqa', [True])
@pytest.mark.parametrize('with_real_shift', [True])
def test_flash_attention_semi_auto_parallel_flops(input_layout, use_mqa, with_real_shift):
    """
    Features: test FlashAttentionScoreInfo
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=16, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    dp = 2
    mp = 4
    sp = 2
    B, N, S, D = 8, 16, 1024, 128
    query, key, value, real_shift, attn_mask = generate_inputs(B, N, S, D,
                                                               input_layout,
                                                               use_mqa,
                                                               with_real_shift)

    class Grad(nn.Cell):
        def __init__(self, network):
            super().__init__()
            self.grad = GradOperation(get_all=True, sens_param=False)
            self.network = network

        def construct(self, *inputs):
            gout = self.grad(self.network)(*inputs)
            return gout

    class Net(nn.Cell):
        def __init__(self, head_num, keep_prob=0.9, input_layout="BSH", sparse_mode=0, use_mqa=False,
                     with_real_shift=True, dp=None, mp=None, sp=1):
            super(Net, self).__init__()
            self.reshape = P.Reshape()
            self.drop_gen_mask = P.DropoutGenMask()
            self.keep_prob = Tensor(keep_prob, ms.float16)
            compressed_mask_mode = [2, 3, 4]
            self.head_num = head_num
            self.input_layout = input_layout
            pre_tokens = 2147483647 if sparse_mode not in compressed_mask_mode else 512
            next_tokens = 2147483647 if sparse_mode not in compressed_mask_mode else 0
            self.fa_op = FlashAttentionScore(head_num=head_num,
                                             keep_prob=keep_prob,
                                             pre_tokens=pre_tokens,
                                             next_tokens=next_tokens,
                                             input_layout=input_layout,
                                             sparse_mode=sparse_mode)
            if dp is not None and mp is not None:
                kv_head_stra = 1 if use_mqa else mp
                if input_layout == "BSH":
                    stra = ((dp, sp, mp), (dp, 1, kv_head_stra),
                            (dp, 1, kv_head_stra))
                else:
                    stra = ((dp, mp, sp, 1), (dp, kv_head_stra, 1, 1),
                            (dp, kv_head_stra, 1, 1))
                if with_real_shift:
                    stra += ((dp, mp, sp, 1),)
                if keep_prob < 1.0:
                    stra += ((dp, mp, sp, 1),)
                if sparse_mode not in compressed_mask_mode:
                    stra += ((dp, 1, sp, 1),)
                else:
                    stra += ((1, 1),)
                self.fa_op.shard(stra)

        def construct(self, query, key, value, real_shift, attn_mask):
            if self.input_layout == "BSH":
                bsz, seq_len, _ = query.shape
            else:
                bsz, _, seq_len, _ = query.shape
            if self.keep_prob < 1.0:
                drop_mask_bits = self.reshape(self.drop_gen_mask((bsz, self.head_num, seq_len, seq_len),
                                                                 self.keep_prob),
                                              (bsz, self.head_num, seq_len, 128))
            else:
                drop_mask_bits = None
            return self.fa_op(query, key, value, real_shift, drop_mask_bits, None, attn_mask, None)[0]

    net = Grad(Net(N, input_layout=input_layout, use_mqa=use_mqa,
                   with_real_shift=with_real_shift, dp=dp, mp=mp, sp=sp))
    compile_net(net, query, key, value, real_shift, attn_mask)

    full_mfu, full_hfu, shard_mfu, shard_hfu, _ = flops_collection(net.current_phase)
    print(f"full_mfu: {full_mfu}, full_hfu: {full_hfu}")
    # origin_shape is [8, 1024, 2048] [8, 1024, 128] [8, 1024, 128]-> [8, 1024, 2048]
    assert full_mfu == 3 * 2 * 2 * B * S * S * N * D
    print(f"shard_mfu: {shard_mfu}, shard_hfu: {shard_hfu}")
    # shard_shape is [4, 512, 512] [4, 1024, 128] [4, 1024, 128]-> [4, 512, 512]
    assert shard_hfu == 4 * 2 * 2 * (B // dp) * (S // sp) * S * (N // mp) * D
