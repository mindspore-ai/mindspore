# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
from mindspore.common.api import ms_function
import numpy as np
import mindspore.context as context
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
class Net(nn.Cell):
  def __init__(self):
    super(Net, self).__init__()
    self.rnnt_loss = P.RNNTLoss(blank_label=0)

  def construct(self, acts, labels, act_lens, label_lens):
    return self.rnnt_loss(acts, labels, act_lens, label_lens)


def test_net():
  B, T, U, V = 1, 2, 3, 5
  acts = np.random.random((B, T, U, V)).astype(np.float32)
  labels = np.array([[np.random.randint(1, V-1) for _ in range(U-1)]]).astype(np.int32)
  input_length = np.array([T] * B).astype(np.int32)
  label_length = np.array([len(l) for l in labels]).astype(np.int32)
  
  rnnt_loss = Net()
  costs, grads = rnnt_loss(Tensor(acts), Tensor(labels), Tensor(input_length), Tensor(label_length))
  print(Tensor(acts), Tensor(labels), Tensor(input_length), Tensor(label_length))
  print(costs.asnumpy())
  print(grads.asnumpy())
