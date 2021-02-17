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
"""effnet_tune_train_export."""

import sys
from os import path
import numpy as np
from train_utils import TrainWrap, SaveT
from effnet import effnet
import mindspore.common.dtype as mstype
from mindspore import context, Tensor, nn
from mindspore.train.serialization import export, load_checkpoint
from mindspore.common.parameter import ParameterTuple

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", save_graphs=False)

class TransferNet(nn.Cell):
    def __init__(self, backbone, head):
        super().__init__(TransferNet)
        self.backbone = backbone
        self.head = head
    def construct(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

CHECKPOINT_WEIGHT_FILE = "efficient_net_b0.ckpt"
if not path.exists(CHECKPOINT_WEIGHT_FILE):
    import subprocess
    print("weight file is missing, downloading from hub")
    url = "https://download.mindspore.cn/model_zoo/official/lite/efficient_net/" + CHECKPOINT_WEIGHT_FILE
    subprocess.run(["wget", url], check=True)

BACKBONE = effnet(num_classes=1000)
load_checkpoint(CHECKPOINT_WEIGHT_FILE, BACKBONE)
HEAD = nn.Dense(1000, 10)
HEAD.weight.set_data(Tensor(np.random.normal(
    0, 0.1, HEAD.weight.data.shape).astype("float32")))
HEAD.bias.set_data(Tensor(np.zeros(HEAD.bias.data.shape, dtype="float32")))

n = TransferNet(BACKBONE, HEAD)
trainable_weights_list = []
trainable_weights_list.extend(n.head.trainable_params())
trainable_weights = ParameterTuple(trainable_weights_list)
sgd = nn.SGD(trainable_weights, learning_rate=0.01, momentum=0.9,
             dampening=0.01, weight_decay=0.0, nesterov=False, loss_scale=1.0)
net = TrainWrap(n, optimizer=sgd, weights=trainable_weights)

BATCH_SIZE = 8
X = Tensor(np.random.randn(BATCH_SIZE, 3, 224, 224), mstype.float32)
label = Tensor(np.zeros([BATCH_SIZE, 10]).astype(np.float32))
export(net, X, label, file_name="mindir/effnet_tune_train", file_format='MINDIR')

if len(sys.argv) > 1:
    name_prefix = sys.argv[1] + "effnet_tune"
    x_name = name_prefix + "_input1.bin"
    SaveT(Tensor(X.asnumpy().transpose(0, 2, 3, 1)), x_name)

    l_name = name_prefix + "_input2.bin"
    SaveT(label, l_name)

    #train network
    n.head.set_train(True)
    n.backbone.set_train(False)
    net(X, label)

    #save Y after training
    n.set_train(False)
    y = n(X)
    y_name = name_prefix + "_output1.bin"
    SaveT(y, y_name)
