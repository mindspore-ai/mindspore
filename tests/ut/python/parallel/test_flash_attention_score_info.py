# Copyright 2019 Huawei Technologies Co., Ltd
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
import json
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Layout, context, Symbol
from mindspore.graph.api import _cell_graph_executor
from mindspore.common.dtype import _pytype_to_dtype
from mindspore.context import set_auto_parallel_context
from mindspore.ops import composite as C
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


def generate_tensor(shape, dtype):
    is_dynamic = False
    for dim in shape:
        if isinstance(dim, Symbol):
            is_dynamic = True
            break
    if is_dynamic:
        return Tensor(shape=shape, dtype=_pytype_to_dtype(dtype))  # pylint:disable=protected-access
    return Tensor(np.ones(shape, dtype))


def generate_inputs(B, N1, N2, S, D1, D2, input_layout, with_real_shift=True, sparse_mode=0):
    N_Q = N1
    N_KV = N2
    D_QK = D1
    D_V = D2
    compressed_mask_mode = [2, 3, 4, 5, 6, 7, 8]
    if input_layout == "BSH":
        H_Q = N_Q * D_QK
        H_K = N_KV * D_QK
        H_V = N_KV * D_V
        query = generate_tensor((B, S, H_Q), dtype=np.float16)
        key = generate_tensor((B, S, H_K), dtype=np.float16)
        value = generate_tensor((B, S, H_V), dtype=np.float16)
    elif input_layout == "SBH":
        H_Q = N_Q * D_QK
        H_K = N_KV * D_QK
        H_V = N_KV * D_V
        query = generate_tensor((S, B, H_Q), dtype=np.float16)
        key = generate_tensor((S, B, H_K), dtype=np.float16)
        value = generate_tensor((S, B, H_V), dtype=np.float16)
    elif input_layout == "BNSD":
        query = generate_tensor((B, N_Q, S, D_QK), dtype=np.float16)
        key = generate_tensor((B, N_KV, S, D_QK), dtype=np.float16)
        value = generate_tensor((B, N_KV, S, D_V), dtype=np.float16)
    elif input_layout == "BSND":
        query = generate_tensor((B, S, N_Q, D_QK), dtype=np.float16)
        key = generate_tensor((B, S, N_KV, D_QK), dtype=np.float16)
        value = generate_tensor((B, S, N_KV, D_V), dtype=np.float16)
    elif input_layout == "TND":
        B_value = B.divisor if isinstance(B, Symbol) else B
        S_value = S.divisor if isinstance(S, Symbol) else S
        T = B_value * S_value
        if isinstance(B, Symbol) or isinstance(S, Symbol):
            T = Symbol(divisor=B_value * S_value)
        query = generate_tensor((T, N_Q, D_QK), dtype=np.float16)
        key = generate_tensor((T, N_KV, D_QK), dtype=np.float16)
        value = generate_tensor((T, N_KV, D_V), dtype=np.float16)
    else:
        raise ValueError(f"input_layout is invalid.")
    real_shift = generate_tensor((B, N_Q, S, S), dtype=np.float16) if with_real_shift else None
    if sparse_mode not in compressed_mask_mode:
        attn_mask = generate_tensor((B, 1, S, S), dtype=np.uint8)
    else:
        attn_mask = generate_tensor((2048, 2048), dtype=np.uint8)
    return query, key, value, real_shift, attn_mask


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x):
        predict = self.network(x)
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


class Net(nn.Cell):
    def __init__(self, head_num, keep_prob=0.9, input_layout="BSH", sparse_mode=0, use_mqa=False,
                 with_real_shift=True, dp=None, mp=None, sp=1, use_layout=False, self_defined_strategy=None):
        super(Net, self).__init__()
        self.reshape = P.Reshape()
        self.drop_gen_mask = P.DropoutGenMask()
        self.keep_prob = Tensor(keep_prob, ms.float16)
        compressed_mask_mode = [2, 3, 4, 5, 6, 7, 8]
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
        if self_defined_strategy is not None:
            self.fa_op.shard(in_strategy=self_defined_strategy)
        elif dp is not None and mp is not None:
            if use_layout:
                if input_layout == "TND":
                    layout = Layout(device_matrix=(dp, sp, mp), alias_name=("dp", "sp", "mp"))
                    kv_head_map_name = "None" if use_mqa else "mp"
                    query_layout = layout(("dp", "sp"), "mp", "None") if sp > 1 else layout("dp", "mp", "None")
                    self.fa_op.shard(in_strategy=(query_layout,
                                                  layout("dp", kv_head_map_name, "None"),
                                                  layout("dp", kv_head_map_name, "None"),
                                                  layout("None", "None"),
                                                  layout("dp"),
                                                  layout("dp")))
                else:
                    raise ValueError("Only TND can be config by layout.")
            else:
                kv_head_stra = 1 if use_mqa else mp
                if input_layout == "BSH":
                    stra = ((dp, sp, mp), (dp, 1, kv_head_stra), (dp, 1, kv_head_stra))
                elif input_layout == "SBH":
                    stra = ((sp, dp, mp), (1, dp, kv_head_stra), (1, dp, kv_head_stra))
                elif input_layout == "BNSD":
                    stra = ((dp, mp, sp, 1), (dp, kv_head_stra, 1, 1), (dp, kv_head_stra, 1, 1))
                elif input_layout == "BSND":
                    stra = ((dp, sp, mp, 1), (dp, 1, kv_head_stra, 1), (dp, 1, kv_head_stra, 1))
                elif input_layout == "TND":
                    stra = ((dp * sp, mp, 1), (dp, kv_head_stra, 1), (dp, kv_head_stra, 1))
                else:
                    raise ValueError(f"input_layout is invalid.")
                if with_real_shift:
                    stra += ((dp, mp, sp, 1),)
                if keep_prob < 1.0:
                    stra += ((dp, mp, sp, 1),)
                if sparse_mode not in compressed_mask_mode:
                    stra += ((dp, 1, sp, 1),)
                else:
                    stra += ((1, 1),)
                if input_layout == "TND":
                    stra += ((dp,),)
                    stra += ((dp,),)
                self.fa_op.shard(stra)

    def construct(self, query, key, value, real_shift, attn_mask, actual_seq_qlen=None, actual_seq_kvlen=None):
        drop_mask_bits = None
        if self.input_layout != "TND":
            if self.input_layout == "BSH":
                bsz, seq_len, _ = query.shape
            elif self.input_layout == "SBH":
                seq_len, bsz, _ = query.shape
            elif self.input_layout == "BNSD":
                bsz, _, seq_len, _ = query.shape
            elif self.input_layout == "BSND":
                bsz, seq_len, _, _ = query.shape
            else:
                raise ValueError(f"input_layout is invalid.")
            if self.keep_prob < 1.0:
                drop_mask_bits = self.reshape(self.drop_gen_mask((bsz, self.head_num, seq_len, seq_len),
                                                                 self.keep_prob),
                                              (bsz, self.head_num, seq_len, 128))
        return self.fa_op(query, key, value, real_shift, drop_mask_bits, None, attn_mask, None, actual_seq_qlen,
                          actual_seq_kvlen)


@pytest.mark.parametrize('keep_prob', [0.9, 1.0])
@pytest.mark.parametrize('input_layout', ["BSH", "SBH", "BNSD", "BSND"])
@pytest.mark.parametrize('with_real_shift', [True, False])
@pytest.mark.parametrize('shape', [(8, 16, 1024, 128, 128), (8, 16, 1024, 192, 128)])
def test_self_attention_standalone(keep_prob, input_layout, with_real_shift, shape):
    """
    Features: test FlashAttentionScoreInfo
    Description: StandAlone
    Expectation: compile success
    """
    context.reset_auto_parallel_context()
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    B, N, S, D1, D2 = shape
    query, key, value, real_shift, attn_mask = generate_inputs(B, N, N, S, D1, D2, input_layout,
                                                               with_real_shift=with_real_shift)
    net = Net(N, keep_prob, input_layout, with_real_shift=with_real_shift)
    compile_net(net, query, key, value, real_shift, attn_mask)


@pytest.mark.parametrize('input_layout', ["BSH", "SBH", "BNSD", "BSND"])
@pytest.mark.parametrize('sparse_mode', [2, 3, 4])
@pytest.mark.parametrize('shape', [(8, 16, 1024, 128, 128), (8, 16, 1024, 192, 128)])
def test_self_attention_standalone_with_compressed_mask(input_layout, sparse_mode, shape):
    """
    Features: test FlashAttentionScoreInfo with compressed mask
    Description: StandAlone
    Expectation: compile success
    """
    context.reset_auto_parallel_context()
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    B, N, S, D1, D2 = shape
    query, key, value, real_shift, attn_mask = generate_inputs(B, N, N, S, D1, D2, input_layout=input_layout,
                                                               sparse_mode=sparse_mode)
    net = Net(N, input_layout=input_layout, sparse_mode=sparse_mode)
    compile_net(net, query, key, value, real_shift, attn_mask)


@pytest.mark.parametrize('input_layout', ["BSH", "SBH", "BNSD", "BSND"])
@pytest.mark.parametrize('use_mqa', [True, False])
@pytest.mark.parametrize('with_real_shift', [True, False])
@pytest.mark.parametrize('shape', [(8, 16, 1024, 128, 128), (8, 16, 1024, 192, 128)])
def test_flash_attention_semi_auto_parallel(input_layout, use_mqa, with_real_shift, shape):
    """
    Features: test FlashAttentionScoreInfo
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    dp = 2
    mp = 4
    B, N, S, D1, D2 = shape
    query, key, value, real_shift, attn_mask = generate_inputs(B, N, 1 if use_mqa else N, S, D1, D2,
                                                               input_layout=input_layout,
                                                               with_real_shift=with_real_shift)
    net = Net(N, input_layout=input_layout, use_mqa=use_mqa,
              with_real_shift=with_real_shift, dp=dp, mp=mp)
    compile_net(net, query, key, value, real_shift, attn_mask)


@pytest.mark.parametrize('input_layout', ["BSH", "SBH", "BNSD", "BSND"])
@pytest.mark.parametrize('sparse_mode', [2, 3, 4])
@pytest.mark.parametrize('shape', [(8, 16, 1024, 128, 128), (8, 16, 1024, 192, 128)])
def test_flash_attention_semi_auto_parallel_with_compressed_mask(input_layout, sparse_mode, shape):
    """
    Features: test FlashAttentionScoreInfo with compressed mask
    Description: semi_auto_parallel with strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    dp = 2
    mp = 4
    B, N, S, D1, D2 = shape
    query, key, value, real_shift, attn_mask = generate_inputs(B, N, N, S, D1, D2,
                                                               input_layout,
                                                               sparse_mode=sparse_mode)
    net = Net(N, input_layout=input_layout, sparse_mode=sparse_mode, dp=dp, mp=mp)
    compile_net(net, query, key, value, real_shift, attn_mask)


@pytest.mark.parametrize('keep_prob', [0.9, 1.0])
@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
@pytest.mark.parametrize('with_real_shift', [True, False])
@pytest.mark.parametrize('shape', [(8, 16, 1024, 128, 128), (8, 16, 1024, 192, 128)])
def test_flash_attention_dp(keep_prob, input_layout, with_real_shift, shape):
    """
    Features: test FlashAttentionScore under semi_auto_parallel
    Description: semi_auto_parallel without strategy
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    B, N, S, D1, D2 = shape
    query, key, value, real_shift, attn_mask = generate_inputs(B, N, N, S, D1, D2, input_layout=input_layout,
                                                               with_real_shift=with_real_shift)
    net = Net(N, keep_prob, input_layout, with_real_shift=with_real_shift)
    compile_net(net, query, key, value, real_shift, attn_mask)


@pytest.mark.parametrize('keep_prob', [0.9, 1.0])
@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
@pytest.mark.parametrize('use_mqa', [True, False])
@pytest.mark.parametrize('with_real_shift', [True, False])
@pytest.mark.parametrize('shape', [(8, 16, 1024, 128, 128), (8, 16, 1024, 192, 128)])
def test_flash_attention_auto_parallel(keep_prob, input_layout, use_mqa, with_real_shift, shape):
    """
    Features: test FlashAttentionScoreInfo
    Description: auto_parallel
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    B, N, S, D1, D2 = shape
    query, key, value, real_shift, attn_mask = generate_inputs(B, N, 1 if use_mqa else N, S, D1, D2,
                                                               input_layout=input_layout,
                                                               with_real_shift=with_real_shift)
    net = Net(N, keep_prob, input_layout, use_mqa=use_mqa, with_real_shift=with_real_shift)
    compile_net(net, query, key, value, real_shift, attn_mask)


@pytest.mark.parametrize('input_layout', ["BSH", "SBH", "BNSD", "BSND"])
@pytest.mark.parametrize('use_mqa', [True, False])
@pytest.mark.parametrize('with_real_shift', [True, False])
@pytest.mark.parametrize('shape', [(8, 16, 1024, 128, 128), (8, 16, 1024, 192, 128)])
def test_flash_attention_with_seq_parallel(input_layout, use_mqa, with_real_shift, shape):
    """
    Features: test FlashAttentionScoreInfo with sequence parallel, sparse_mode=0
    Description: semi_auto_parallel with strategy, seq_parallel
    Expectation: compile success
    """
    dp = 2
    mp = 2
    sp = 2
    set_auto_parallel_context(device_num=dp * mp * sp, global_rank=0)
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')
    B, N, S, D1, D2 = shape
    query, key, value, real_shift, attn_mask = generate_inputs(B, N, 1 if use_mqa else N, S, D1, D2,
                                                               input_layout=input_layout,
                                                               with_real_shift=with_real_shift)
    net = Net(N, input_layout=input_layout, use_mqa=use_mqa,
              with_real_shift=with_real_shift, dp=dp, mp=mp, sp=sp)
    compile_net(net, query, key, value, real_shift, attn_mask)


@pytest.mark.parametrize('input_layout', ["BSH", "SBH", "BNSD", "BSND"])
@pytest.mark.parametrize('sparse_mode', [2, 3, 4])
@pytest.mark.parametrize('shape', [(8, 16, 1024, 128, 128), (8, 16, 1024, 192, 128)])
def test_flash_attention_compressed_mask_with_seq_parallel(input_layout, sparse_mode, shape):
    """
    Features: test FlashAttentionScoreInfo with sequence parallel, sparse_mode=[2, 3, 4]
    Description: semi_auto_parallel with strategy, seq_parallel
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')
    dp = 2
    mp = 2
    sp = 2
    B, N, S, D1, D2 = shape
    query, key, value, real_shift, attn_mask = generate_inputs(B, N, N, S, D1, D2,
                                                               input_layout,
                                                               sparse_mode=sparse_mode)
    net = Net(N, input_layout=input_layout, sparse_mode=sparse_mode,
              dp=dp, mp=mp, sp=sp)
    compile_net(net, query, key, value, real_shift, attn_mask)


@pytest.mark.parametrize('input_layout', ["BSH", "SBH", "BNSD", "BSND"])
@pytest.mark.parametrize('use_mqa', [True, False])
@pytest.mark.parametrize('with_real_shift', [True, False])
@pytest.mark.parametrize('shape', [(8, 16, 1024, 128, 128), (8, 16, 1024, 192, 128)])
def test_flash_attention_with_load_balance(input_layout, use_mqa, with_real_shift, shape):
    """
    Features: test FlashAttentionScoreInfo with sequence parallel load balance, sparse_mode=0
    Description: semi_auto_parallel with strategy, seq_parallel and load_balance
    Expectation: compile success
    """
    config = {"enable_flash_attention_load_balance": True,}
    with open("./parallel_speed_up_for_fa_1.json", "w") as file:
        json.dump(config, file, indent=4, separators=(',', ': '))
    context.set_context(
        ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_for_fa_1.json"})
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')
    dp = 2
    mp = 2
    sp = 2
    B, N, S, D1, D2 = shape
    query, key, value, real_shift, attn_mask = generate_inputs(B, N, 1 if use_mqa else N, S, D1, D2,
                                                               input_layout=input_layout,
                                                               with_real_shift=with_real_shift)
    net = Net(N, input_layout=input_layout, use_mqa=use_mqa, with_real_shift=with_real_shift,
              dp=dp, mp=mp, sp=sp)
    compile_net(net, query, key, value, real_shift, attn_mask)
    config = {"enable_flash_attention_load_balance": False,}
    with open("./parallel_speed_up_for_fa_1.json", "w") as file:
        json.dump(config, file, indent=4, separators=(',', ': '))
    context.set_context(
        ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_for_fa_1.json"})


@pytest.mark.parametrize('input_layout', ["BSH", "SBH", "BNSD", "BSND"])
@pytest.mark.parametrize('sparse_mode', [2, 3, 4])
@pytest.mark.parametrize('shape', [(8, 16, 1024, 128, 128), (8, 16, 1024, 192, 128)])
def test_flash_attention_compressed_mask_with_load_balance(input_layout, sparse_mode, shape):
    """
    Features: test FlashAttentionScoreInfo with sequence parallel load balance, sparse_mode=[2, 3, 4]
    Description: semi_auto_parallel with strategy, seq_parallel and load_balance
    Expectation: compile success
    """
    config = {"enable_flash_attention_load_balance": True,}
    with open("./parallel_speed_up_for_fa_2.json", "w") as file:
        json.dump(config, file, indent=4, separators=(',', ': '))
    context.set_context(
        ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_for_fa_2.json"})
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')
    dp = 2
    mp = 2
    sp = 2
    B, N, S, D1, D2 = shape
    query, key, value, real_shift, attn_mask = generate_inputs(B, N, N, S, D1, D2,
                                                               input_layout,
                                                               sparse_mode=sparse_mode)
    net = Net(N, input_layout=input_layout, sparse_mode=sparse_mode,
              dp=dp, mp=mp, sp=sp)
    compile_net(net, query, key, value, real_shift, attn_mask)
    config = {"enable_flash_attention_load_balance": False,}
    with open("./parallel_speed_up_for_fa_2.json", "w") as file:
        json.dump(config, file, indent=4, separators=(',', ': '))
    context.set_context(
        ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_for_fa_2.json"})


def generate_dynamic_inputs(B, N, S, D):
    H = N * D
    query = Tensor(shape=[B, S, H], dtype=ms.float16)
    key = Tensor(shape=[B, S, H], dtype=ms.float16)
    value = Tensor(shape=[B, S, H], dtype=ms.float16)
    attn_mask = Tensor(shape=[B, 1, S, S], dtype=ms.uint8)
    return query, key, value, None, attn_mask


@pytest.mark.parametrize('keep_prob', [0.9])
def test_flash_attention_dynamic_shape_constraint(keep_prob):
    """
    Features: test FlashAttentionScoreInfo dynamic shape
    Description: semi_auto_parallel with strategy
    Expectation: compile failed
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=False)
    dp = 2
    mp = 4
    B, N, S, D = None, 16, 1024, 128
    inputs = generate_dynamic_inputs(B, N, S, D)
    net = Net(N, keep_prob, dp=dp, mp=mp)
    with pytest.raises(RuntimeError):
        compile_net(net, *inputs)


@pytest.mark.parametrize('is_actual_tuple', [True, False])
@pytest.mark.parametrize('dp_sp_mp', [(2, 2, 2), (4, 1, 2), (1, 4, 2)])
@pytest.mark.parametrize('use_layout', [True, False])
@pytest.mark.parametrize('shape', [(8, 16, 1024, 128, 128), (8, 16, 1024, 192, 128)])
def test_flash_attention_tnd(is_actual_tuple, dp_sp_mp, use_layout, shape):
    """
    Features: test FlashAttentionScoreInfo FlashAttentionScore
    Description: Test for TND layout
    Expectation: compile success if use_layout else raise RuntimeError
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')
    dp, sp, mp = dp_sp_mp
    B, N, S, D1, D2 = shape
    input_layout = "TND"
    sparse_mode = 3
    query, key, value, real_shift, attn_mask = generate_inputs(B, N, N, S, D1, D2,
                                                               input_layout,
                                                               sparse_mode=sparse_mode,
                                                               with_real_shift=False
                                                               )
    inter = 512
    if is_actual_tuple:
        actual_seq_qlen = tuple(range(inter, B * S + 1, inter))
        actual_seq_kvlen = tuple(range(inter, B * S + 1, inter))
    else:
        actual_seq_qlen = Tensor(np.array(range(inter, B * S + 1, inter), np.int64))
        actual_seq_kvlen = Tensor(np.array(range(inter, B * S + 1, inter), np.int64))
    net = Net(N, input_layout=input_layout, use_mqa=False, keep_prob=1.0, sparse_mode=sparse_mode,
              with_real_shift=False, dp=dp, mp=mp, sp=sp, use_layout=use_layout)

    if sp > 1 and not use_layout:
        # Cannot slice seq-dim if config by strategy
        with pytest.raises(RuntimeError):
            compile_net(net, query, key, value, real_shift, attn_mask, actual_seq_qlen, actual_seq_kvlen)
    else:
        compile_net(net, query, key, value, real_shift, attn_mask, actual_seq_qlen, actual_seq_kvlen)


@pytest.mark.parametrize('shape', [(8, 64, 8, 1024, 128, 128), (8, 64, 8, 1024, 192, 128)])
def test_flash_attention_bsh_layout_with_gqa(shape):
    """
    Features: test FlashAttentionScoreInfo FlashAttentionScore
    Description: Test for BSH with GQA
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=128, global_rank=0)
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')
    dp, sp, mp = 2, 8, 8
    B, N1, N2, S, D1, D2 = shape
    input_layout = "BSH"
    sparse_mode = 2
    query, key, value, real_shift, attn_mask = generate_inputs(B, N1, N2, S, D1, D2,
                                                               input_layout,
                                                               sparse_mode=sparse_mode,
                                                               with_real_shift=False
                                                               )
    layout = Layout((dp, sp, mp), ("dp", "sp", "mp"))
    in_strategy = (layout("dp", "None", ("sp", "mp")),
                   layout("dp", "None", "mp"),
                   layout("dp", "None", "mp"),
                   layout("None", "None"))
    net = Net(N1, input_layout=input_layout, use_mqa=False, keep_prob=1.0, sparse_mode=sparse_mode,
              with_real_shift=False, dp=dp, mp=mp, sp=sp, use_layout=True, self_defined_strategy=in_strategy)

    compile_net(net, query, key, value, real_shift, attn_mask)


@pytest.mark.parametrize('enable_load_balance', [True, False])
@pytest.mark.parametrize('with_real_shift', [True, False])
@pytest.mark.parametrize('sparse_mode', [0, 1, 2])
@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD'])
@pytest.mark.parametrize('shape', [(Symbol(divisor=2), 16, Symbol(divisor=8), 128, 128),
                                   (Symbol(divisor=2), 16, Symbol(divisor=8), 192, 128)])
def test_flash_attention_dynamic_with_strategy(enable_load_balance, with_real_shift, sparse_mode, input_layout, shape):
    """
    Features: test FlashAttentionScoreInfo FlashAttentionScore
    Description: Test for batch dim and seq dim dynamic
    Expectation: compile success
    """
    dp, sp, mp = 2, 2, 2
    set_auto_parallel_context(device_num=dp * sp * mp, global_rank=0)
    if enable_load_balance:
        config = {"enable_flash_attention_load_balance": True,}
        with open("./parallel_speed_up_for_fa_3.json", "w") as file:
            json.dump(config, file, indent=4, separators=(',', ': '))
        context.set_context(
            ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_for_fa_3.json"})
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    keep_prob = 1.0
    B, N, S, D1, D2 = shape
    query, key, value, real_shift, attn_mask = generate_inputs(B, N, N, S, D1, D2, sparse_mode=sparse_mode,
                                                               with_real_shift=with_real_shift,
                                                               input_layout=input_layout)
    net = Net(N, input_layout=input_layout, keep_prob=keep_prob, sparse_mode=sparse_mode,
              with_real_shift=with_real_shift, dp=dp, sp=sp, mp=mp)
    net.set_inputs(query, key, value, real_shift, attn_mask)
    compile_net(net, query, key, value, real_shift, attn_mask)


@pytest.mark.parametrize('is_actual_tuple', [True, False])
@pytest.mark.parametrize('use_layout', [True, False])
@pytest.mark.parametrize('shape', [(8, 16, Symbol(divisor=1), 128, 128), (8, 16, Symbol(divisor=1), 192, 128)])
def test_flash_attention_dynamic_tnd(is_actual_tuple, use_layout, shape):
    """
    Features: test FlashAttentionScoreInfo FlashAttentionScore
    Description: Test for TND layout
    Expectation: compile success if use_layout else raise RuntimeError
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')
    dp, sp, mp = 4, 1, 2
    B, N, S, D1, D2 = shape
    input_layout = "TND"
    sparse_mode = 3
    query, key, value, real_shift, attn_mask = generate_inputs(B, N, N, S, D1, D2,
                                                               input_layout,
                                                               sparse_mode=sparse_mode,
                                                               with_real_shift=False
                                                               )
    S = 1024
    inter = 512
    if is_actual_tuple:
        actual_seq_qlen = tuple(range(inter, B * S + 1, inter))
        actual_seq_kvlen = tuple(range(inter, B * S + 1, inter))
    else:
        actual_seq_qlen = Tensor(np.array(range(inter, B * S + 1, inter), np.int64))
        actual_seq_kvlen = Tensor(np.array(range(inter, B * S + 1, inter), np.int64))
    net = Net(N, input_layout=input_layout, use_mqa=False, keep_prob=1.0, sparse_mode=sparse_mode,
              with_real_shift=False, dp=dp, mp=mp, sp=sp, use_layout=use_layout)

    if sp > 1 and not use_layout:
        # Cannot slice seq-dim if config by strategy
        with pytest.raises(RuntimeError):
            compile_net(net, query, key, value, real_shift, attn_mask, actual_seq_qlen, actual_seq_kvlen)
    else:
        compile_net(net, query, key, value, real_shift, attn_mask, actual_seq_qlen, actual_seq_kvlen)
