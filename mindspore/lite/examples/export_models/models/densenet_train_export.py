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
"""densenet_train_export."""

import sys
import os
import numpy as np
from train_utils import SaveInOut, TrainWrap
import mindspore.common.dtype as mstype
from mindspore import context, Tensor, nn
from mindspore.train.serialization import export

sys.path.append(os.environ['CLOUD_MODEL_ZOO'] + 'official/cv/densenet121/')
#pylint: disable=wrong-import-position
from official.cv.densenet121.src.network.densenet import DenseNet121




context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", save_graphs=False)

n = DenseNet121(num_classes=10)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
optimizer = nn.SGD(n.trainable_params(), learning_rate=0.001, momentum=0.9, dampening=0.0, weight_decay=0.0,
                   nesterov=True, loss_scale=0.9)
net = TrainWrap(n, loss_fn, optimizer)

batch = 2
x = Tensor(np.random.randn(batch, 3, 224, 224), mstype.float32)
label = Tensor(np.zeros([batch, 10]).astype(np.float32))
export(net, x, label, file_name="mindir/densenet_train", file_format='MINDIR')

if len(sys.argv) > 1:
    SaveInOut(sys.argv[1] + "densenet", x, label, n, net)
