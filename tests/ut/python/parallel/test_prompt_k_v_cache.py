# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore.ops.operations._inner_ops import PromptKVCache
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class PromptKVCacheNet(Cell):
    def __init__(self, padding_mode, strategy):
        super(PromptKVCacheNet, self).__init__()
        self.prompt_k_v_cache = PromptKVCache(padding_mode).shard(strategy)

    def construct(self, cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len):
        return self.prompt_k_v_cache(cache, update, valid_seq_len, batch_index, seq_len_axis,
                                     new_max_seq_len, cur_max_seq_len)


def test_prompt_k_v_cache_net_dim4():
    """
    Feature: test prompt_k_v_cache auto parallel 4 dims
    Description: auto parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((4, 1, 1, 1,), (4, 1, 1, 1,), (4,), (1,), (1,), (1,), (1,))
    padding_mode = "right"
    net = PromptKVCacheNet(padding_mode, strategy)
    cache = Parameter(Tensor(np.ones([4, 40, 1024, 128]), dtype=ms.float16), "cache")
    update = Parameter(Tensor(np.ones([4, 40, 1, 128]), dtype=ms.float16), "update")
    valid_seq_len = Parameter(Tensor(np.ones([4]), dtype=ms.int64), "valid_seq_len")
    batch_index = Parameter(Tensor(np.ones([1]), dtype=ms.int64), "batch_index")
    seq_len_axis = Parameter(Tensor(np.ones([1]), dtype=ms.int64), "seq_len_axis")
    new_max_seq_len = Parameter(Tensor(np.ones([1]), dtype=ms.int64), "new_max_seq_len")
    cur_max_seq_len = Parameter(Tensor(np.ones([1]), dtype=ms.int64), "cur_max_seq_len")
    net.set_inputs(cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len)

    phase = compile_net(net, cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('cache', [1, 40, 1024, 128])
    assert validator.check_parameter_shape('update', [1, 40, 1, 128])
    assert validator.check_parameter_shape('valid_seq_len', [1])


def test_prompt_k_v_cache_net_dim3():
    """
    Feature: test prompt_k_v_cache auto parallel 4 dims
    Description: auto parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((4, 1, 1,), (4, 1, 1,), (4,), (1,), (1,), (1,), (1,))
    padding_mode = "right"
    net = PromptKVCacheNet(padding_mode, strategy)
    cache = Parameter(Tensor(np.ones([4, 1024, 5120]), dtype=ms.float16), "cache")
    update = Parameter(Tensor(np.ones([4, 1, 5120]), dtype=ms.float16), "update")
    valid_seq_len = Parameter(Tensor(np.ones([4]), dtype=ms.int64), "valid_seq_len")
    batch_index = Parameter(Tensor(np.ones([1]), dtype=ms.int64), "batch_index")
    seq_len_axis = Parameter(Tensor(np.ones([1]), dtype=ms.int64), "seq_len_axis")
    new_max_seq_len = Parameter(Tensor(np.ones([1]), dtype=ms.int64), "new_max_seq_len")
    cur_max_seq_len = Parameter(Tensor(np.ones([1]), dtype=ms.int64), "cur_max_seq_len")
    net.set_inputs(cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len)

    phase = compile_net(net, cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('cache', [1, 1024, 5120])
    assert validator.check_parameter_shape('update', [1, 1, 5120])
    assert validator.check_parameter_shape('valid_seq_len', [1])
