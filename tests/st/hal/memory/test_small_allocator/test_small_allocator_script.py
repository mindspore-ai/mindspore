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
import mindspore as ms
from mindspore import Tensor, ops
import numpy as np
import gc


def small_allocator_test_case():
    """
    Perform a series of operation and returns the memory fragment.

    return:
        int: memory fragment size of this test case
    """
    t1 = Tensor(np.random.rand(1024, 1024, 1), ms.float32)
    transpose = ops.Transpose()
    transpose(t1, (0, 2, 1))

    t2 = Tensor(np.random.rand(1, 2), ms.float32)
    output_2 = transpose(t2, (1, 0))
    print(output_2)

    # force garbage collection to make sure t1 is released
    del t1
    gc.collect()

    t3 = Tensor(np.random.rand(1024, 1024, 3), ms.float32)
    t4 = Tensor(np.random.rand(1024, 1024, 3), ms.float32)
    add = ops.Add()
    output = add(t3, t4)
    print(output)

    memory_stats = ms.runtime.memory_stats()
    return memory_stats["total_reserved_memory"] - memory_stats["total_allocated_memory"]


if __name__ == '__main__':
    print(small_allocator_test_case())
