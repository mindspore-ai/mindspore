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
"""transfer_learning_export."""

import numpy as np
import mindspore as M
from mindspore.nn import Cell
from mindspore.train.serialization import load_checkpoint, export
from effnet import effnet
from train_utils import TrainWrap


class TransferNet(Cell):
    def __init__(self, backbone, head):
        super().__init__(TransferNet)
        self.backbone = backbone
        self.head = head

    def construct(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


BACKBONE = effnet(num_classes=1000)
load_checkpoint("efficient_net_b0.ckpt", BACKBONE)

M.context.set_context(mode=M.context.PYNATIVE_MODE,
                      device_target="GPU", save_graphs=False)
BATCH_SIZE = 16
X = M.Tensor(np.ones((BATCH_SIZE, 3, 224, 224)), M.float32)
export(BACKBONE, X, file_name="transfer_learning_tod_backbone", file_format='MINDIR')

label = M.Tensor(np.zeros([BATCH_SIZE, 10]).astype(np.float32))
HEAD = M.nn.Dense(1000, 10)
HEAD.weight.set_data(M.Tensor(np.random.normal(
    0, 0.1, HEAD.weight.data.shape).astype("float32")))
HEAD.bias.set_data(M.Tensor(np.zeros(HEAD.bias.data.shape, dtype="float32")))

sgd = M.nn.SGD(HEAD.trainable_params(), learning_rate=0.015, momentum=0.9,
               dampening=0.01, weight_decay=0.0, nesterov=False, loss_scale=1.0)
net = TrainWrap(HEAD, optimizer=sgd)
backbone_out = M.Tensor(np.zeros([BATCH_SIZE, 1000]).astype(np.float32))
export(net, backbone_out, label, file_name="transfer_learning_tod_head", file_format='MINDIR')

print("Exported")
