# Copyright 2022 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.ops.operations._rl_inner_ops as rl_ops
from mindspore import context, Tensor


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_gru():
    """
    Feature: test gru cpu operation.
    Description: test gru cpu operation.
    Expectation: no exception.
    """
    context.set_context(device_target="CPU")
    input_size = 10
    hidden_size = 2
    num_layers = 1
    max_seq_len = 5
    batch_size = 2

    net = rl_ops.GRUV2(input_size, hidden_size, num_layers, True, False, 0.0)
    input_tensor = Tensor(np.ones([max_seq_len, batch_size, input_size]).astype(np.float32))
    h0 = Tensor(np.ones([num_layers, batch_size, hidden_size]).astype(np.float32))
    w = Tensor(np.ones([84, 1, 1]).astype(np.float32))
    seq_lengths = Tensor(np.array([4, 3]).astype(np.int32))
    output, hn, out1, out2 = net(input_tensor, h0, w, seq_lengths)
    print("output:", output)
    print("hn:", hn)
    print("out1:", out1)
    print("out2:", out2)
