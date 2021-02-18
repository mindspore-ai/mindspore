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
"""mobilenetv2_train_export."""

import sys
import numpy as np
from train_utils import SaveInOut, TrainWrap
from official.cv.mobilenetv2.src.mobilenetV2 import MobileNetV2Backbone, MobileNetV2Head, mobilenet_v2
import mindspore.common.dtype as mstype
from mindspore import context, Tensor, nn
from mindspore.train.serialization import export

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", save_graphs=False)
batch = 16

#n = MobileNetV2()
backbone_net = MobileNetV2Backbone()
head_net = MobileNetV2Head(input_channel=backbone_net.out_channels, num_classes=10)
n = mobilenet_v2(backbone_net, head_net)

loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
optimizer = nn.Momentum(n.trainable_params(), 0.01, 0.9, use_nesterov=False)
net = TrainWrap(n, loss_fn, optimizer)

x = Tensor(np.random.randn(batch, 3, 224, 224), mstype.float32)
label = Tensor(np.zeros([batch, 10]).astype(np.float32))
export(net, x, label, file_name="mindir/mobilenetv2_train", file_format='MINDIR')

if len(sys.argv) > 1:
    SaveInOut(sys.argv[1] + "mobilenetv2", x, label, n, net, sparse=False)
