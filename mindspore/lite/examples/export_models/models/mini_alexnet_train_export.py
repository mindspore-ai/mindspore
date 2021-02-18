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
"""mini_alexnet_train_export."""

import sys
import numpy as np
from train_utils import SaveInOut, TrainWrap
from mini_alexnet import AlexNet
from mindspore import context, Tensor, nn
from mindspore.train.serialization import export

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", save_graphs=False)

# Mini alexnet is designed for MNIST data
batch = 2
number_of_classes = 10
n = AlexNet(phase='test')

loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
optimizer = nn.Adam(n.trainable_params(), learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, use_locking=False,
                    use_nesterov=False, weight_decay=0.0, loss_scale=1.0)
net = TrainWrap(n, loss_fn, optimizer)

x = Tensor(np.ones([batch, 1, 32, 32]).astype(np.float32) * 0.01)
label = Tensor(np.zeros([batch, number_of_classes]).astype(np.float32))
export(net, x, label, file_name="mindir/mini_alexnet_train", file_format='MINDIR')

if len(sys.argv) > 1:
    SaveInOut(sys.argv[1] + "mini_alexnet", x, label, n, net, sparse=False)
