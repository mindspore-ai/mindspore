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
"""lenet_export."""

import sys
import numpy as np
from mindspore import context, Tensor, FixedLossScaleManager
import mindspore.common.dtype as mstype
from mindspore.train.serialization import export
from lenet import LeNet5
from train_utils import train_wrap


n = LeNet5()
n.set_train()
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU", save_graphs=False)

BATCH_SIZE = int(sys.argv[1])
x = Tensor(np.ones((BATCH_SIZE, 1, 32, 32)), mstype.float32)
label = Tensor(np.zeros([BATCH_SIZE]).astype(np.int32))
net = train_wrap(n)
export(net, x, label, file_name="lenet_tod", file_format='MINDIR')
loss_scale = 128.0
loss_scale_manager = FixedLossScaleManager(loss_scale, False)
mix_precision_net = train_wrap(n, None, None, None, loss_scale_manager)
export(mix_precision_net, x, label, file_name="mix_lenet_tod", file_format='MINDIR')
print("finished exporting")
