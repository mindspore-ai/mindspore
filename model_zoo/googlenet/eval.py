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
"""
##############test googlenet example on cifar10#################
python eval.py
"""
import mindspore.nn as nn
from mindspore import context
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import cifar_cfg as cfg
from src.dataset import create_dataset
from src.googlenet import GoogleNet


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
    context.set_context(device_id=cfg.device_id)

    net = GoogleNet(num_classes=cfg.num_classes)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, cfg.momentum,
                   weight_decay=cfg.weight_decay)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean', is_grad=False)
    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

    param_dict = load_checkpoint(cfg.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    dataset = create_dataset(cfg.data_path, 1, False)
    acc = model.eval(dataset)
    print("accuracy: ", acc)
