# Copyright 2021 Huawei Technologies Co., Ltd
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
"""lenet_train_export."""

import sys
import numpy as np
from train_utils import SaveInOut, TrainWrap
from official.cv.lenet.src.lenet import LeNet5
import mindspore.common.dtype as mstype
from mindspore import context, Tensor, nn
from mindspore.train.serialization import export

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", save_graphs=False)

n = LeNet5()
loss_fn = nn.MSELoss()
optimizer = nn.Adam(n.trainable_params(), learning_rate=1e-2, beta1=0.5, beta2=0.7, eps=1e-2, use_locking=True,
                    use_nesterov=False, weight_decay=0.0, loss_scale=0.3)
net = TrainWrap(n, loss_fn, optimizer)

x = Tensor(np.random.randn(32, 1, 32, 32), mstype.float32)
label = Tensor(np.zeros([32, 10]).astype(np.float32))
export(net, x, label, file_name="mindir/lenet_train", file_format='MINDIR')

if len(sys.argv) > 1:
    SaveInOut(sys.argv[1] + "lenet", x, label, n, net, sparse=False)
