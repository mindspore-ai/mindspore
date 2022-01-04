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

import argparse
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore import export
from src.model import LeNet5
from src.adam import AdamWeightDecayOp

parser = argparse.ArgumentParser(description="export mindir for lenet")
parser.add_argument("--device_target", type=str, default="CPU")
parser.add_argument("--mindir_path", type=str,
                    default="lenet_train.mindir")  # the mindir file path of the model to be export

args, _ = parser.parse_known_args()
device_target = args.device_target
mindir_path = args.mindir_path

context.set_context(mode=context.GRAPH_MODE, device_target=device_target)

if __name__ == "__main__":
    epoch = 1
    np.random.seed(0)
    network = LeNet5(62)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
    net_adam_opt = AdamWeightDecayOp(network.trainable_params(), weight_decay=0.1)
    net_with_criterion = WithLossCell(network, criterion)
    train_network = TrainOneStepCell(net_with_criterion, net_opt)
    train_network.set_train()
    losses = []

    for _ in range(epoch):
        data = Tensor(np.random.rand(32, 3, 32, 32).astype(np.float32))
        label = Tensor(np.random.randint(0, 61, (32)).astype(np.int32))
        loss = train_network(data, label).asnumpy()
        losses.append(loss)
        export(train_network, data, label, file_name=mindir_path,
               file_format='MINDIR')  # Add the export statement to obtain the model file in MindIR format.
    print(losses)
