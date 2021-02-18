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
"""mobilenetv3_train_export."""

import sys
import numpy as np
from train_utils import SaveInOut, TrainWrap
from official.cv.mobilenetv3.src.mobilenetV3 import mobilenet_v3_small
import mindspore.common.dtype as mstype
from mindspore import context, Tensor, nn
from mindspore.train.serialization import export

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", save_graphs=False)

n = mobilenet_v3_small(num_classes=10)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
optimizer = nn.Adam(n.trainable_params(), learning_rate=1e-2, beta1=0.5, beta2=0.7, eps=1e-2, use_locking=True,
                    use_nesterov=False, weight_decay=0.1, loss_scale=0.3)
net = TrainWrap(n, loss_fn, optimizer)

batch = 2
x = Tensor(np.random.randn(batch, 3, 224, 224), mstype.float32)
label = Tensor(np.zeros([batch, 10]).astype(np.float32))
export(net, x, label, file_name="mindir/mobilenetv3_train", file_format='MINDIR')

if len(sys.argv) > 1:
    SaveInOut(sys.argv[1] + "mobilenetv3", x, label, n, net, sparse=False)
