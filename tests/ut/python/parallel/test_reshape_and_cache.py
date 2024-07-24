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
from mindspore.nn import Cell
from mindspore import context, Tensor, Parameter
from mindspore.ops.auto_generate import ReshapeAndCache
from parallel.utils.utils import ParallelValidator, compile_net
import pytest


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

def generate_inputs(bs=1, seq_len=1, num_head=40, head_dim=128, block_size=16, max_seq=2048):
    hidden_size = num_head * head_dim
    key = Parameter(Tensor(np.ones([bs, seq_len, hidden_size]), dtype=ms.float16), "key")
    value = Parameter(Tensor(np.ones([bs, seq_len, hidden_size]), dtype=ms.float16), "value")
    key_cache = Parameter(Tensor(np.ones([max_seq, block_size, num_head, head_dim]), dtype=ms.float16), "key_cache")
    value_cache = Parameter(Tensor(np.ones([max_seq, block_size, num_head, head_dim]), dtype=ms.float16), "value_cache")
    slot_mapping = Parameter(Tensor(np.ones([bs * seq_len]), dtype=ms.int32), "slot_mapping")
    return key, value, key_cache, value_cache, slot_mapping


class ReshapeAndCacheNet(Cell):
    def __init__(self, mp=None, strategy=None):
        super(ReshapeAndCacheNet, self).__init__()
        self.reshape_and_cache = ReshapeAndCache()
        if strategy is not None:
            self.reshape_and_cache.shard(strategy)
        elif mp is not None:
            strategy = ((1, 1, mp), (1, 1, mp), (1, 1, mp, 1), (1, 1, mp, 1), (1,))
            self.reshape_and_cache.shard(strategy)

    def construct(self, key, value, key_cache, value_cache, slot_mapping):
        return self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)


def test_reshape_and_cache_semi_auto_parallel():
    """
    Feature: test reshape_and_cache semi auto parallel
    Description: semi auto parallel with strategy
    Expectation: compile success and shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    mp = 8
    net = ReshapeAndCacheNet(mp)

    bs, seq_len, num_head, head_dim, blk_size, max_seq = 1, 1, 40, 128, 16, 2048
    net_inputs = generate_inputs(bs, seq_len, num_head, head_dim, blk_size, max_seq)
    net.set_inputs(*net_inputs)

    phase = compile_net(net, *net_inputs)
    validator = ParallelValidator(net, phase)
    hidden_size = num_head * head_dim
    assert validator.check_parameter_shape('key', [bs, seq_len, hidden_size // mp])
    assert validator.check_parameter_shape('value', [bs, seq_len, hidden_size // mp])
    assert validator.check_parameter_shape('key_cache', [max_seq, blk_size, num_head // mp, head_dim])
    assert validator.check_parameter_shape('value_cache', [max_seq, blk_size, num_head // mp, head_dim])
    assert validator.check_parameter_shape('slot_mapping', [bs * seq_len])


def test_reshape_and_cache_standalone():
    """
    Feature: test reshape_and_cache standalone
    Description: StandAlone
    Expectation: compile success
    """
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    net = ReshapeAndCacheNet()

    bs, seq_len, num_head, head_dim, blk_size, max_seq = 1, 1, 40, 128, 16, 2048
    net_inputs = generate_inputs(bs, seq_len, num_head, head_dim, blk_size, max_seq)
    net.set_inputs(*net_inputs)
    compile_net(net, *net_inputs)


def test_reshape_and_cache_strategy_error():
    """
    Feature: test reshape_and_cache parallel with wrong strategy
    Description: semi auto parallel with strategy
    Expectation: error and catch
    """
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    mp = 4
    mp2 = 8
    strategies = [
        ((1, 1, mp), (1, mp2, 1), (1, 1, 1, mp), (1, 1, 1, mp), (1,)),
        ((1, 1, mp), (1, 1, mp), (1, 1, 1, mp), (1, 1, mp2, 1), (1,)),
        ((1, 1, mp), (1, mp2, 1), (1, 1, 1, mp), (1, 1, mp2, 1), (1,)),
        ((1, 1, mp), (1, 1, mp), (1, 1, 1, mp), (1, 1, 1, mp), (2,))
    ]
    for strategy in strategies:
        net = ReshapeAndCacheNet(strategy=strategy)
        with pytest.raises(RuntimeError):
            bs, seq_len, num_head, head_dim, blk_size, max_seq = 1, 1, 40, 128, 16, 2048
            net_inputs = generate_inputs(bs, seq_len, num_head, head_dim, blk_size, max_seq)
            net.set_inputs(*net_inputs)
            compile_net(net, *net_inputs)
    context.reset_auto_parallel_context()
