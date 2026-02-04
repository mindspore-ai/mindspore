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

"""test hccl allreduce performance with 8p"""

import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import mint, ops, Tensor

def test_communication_op_with_stream():
    '''
    Feature: Communication op with stream.
    Description: Test communication op with stream.
    Expectation: Run success.
    '''
    D.init()

    a = Tensor(np.ones([1024, 2048]), ms.float32)
    b = Tensor(np.ones([2048, 1024]), ms.float32)
    c = ops.matmul(a, b)
    c.numpy()
    mem_stats = ms.runtime.memory_stats()
    assert mem_stats["common_mem_pool_stats"]["block_counts"] == 1

    s1 = ms.runtime.Stream()
    with ms.runtime.StreamCtx(s1):
        input_x = Tensor(np.ones([3, 4]).astype(np.float32))
        mint.distributed.all_reduce(input_x)
    input_x.numpy()
    mem_stats = ms.runtime.memory_stats()
    # communication op allocate memory with default stream!
    assert mem_stats["common_mem_pool_stats"]["block_counts"] == 1
