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
"""
Test for race check.
"""
import numpy as np
from mindspore.mint.distributed import init_process_group
from mindspore.mint.distributed import all_reduce
from mindspore import Tensor

init_process_group()
tensor = Tensor(np.ones([2, 8]).astype(np.float32))
tensor_1 = Tensor(np.ones([2, 8]).astype(np.float32))
tensor_2 = Tensor(np.ones([2, 8]).astype(np.float32))
output = all_reduce(tensor, async_op=True)
tensor_3 = tensor_2.add(tensor_1)  # add, no data race
output.wait()
tensor.add_(tensor_1)  # add inplace
