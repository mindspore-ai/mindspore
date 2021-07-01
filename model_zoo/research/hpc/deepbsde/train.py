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
"""DeepBSDE train script"""
import os
from mindspore import dtype as mstype
from mindspore import context, Tensor, Model
from mindspore import nn
from mindspore.nn.dynamic_lr import piecewise_constant_lr
from mindspore.train.callback import TimeMonitor, LossMonitor
from src.net import DeepBSDE, WithLossCell
from src.config import config
from src.equation import get_bsde, create_dataset
from src.eval_utils import EvalCallBack

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    config.ckpt_path = os.path.join(config.log_dir, "deepbsde_{}_{}.ckpt".format(config.eqn_name, "{}"))
    bsde = get_bsde(config)
    dataset = create_dataset(bsde)
    print('Begin to solve', config.eqn_name)
    net = DeepBSDE(config, bsde)
    net_with_loss = WithLossCell(net)
    config.lr_boundaries.append(config.num_iterations)
    lr = Tensor(piecewise_constant_lr(config.lr_boundaries, config.lr_values), dtype=mstype.float32)
    opt = nn.Adam(net.trainable_params(), lr)
    model = Model(net_with_loss, optimizer=opt)
    eval_param = {"model": net_with_loss, "valid_data": bsde.sample(config.valid_size)}
    cb = [LossMonitor(), TimeMonitor(), EvalCallBack(eval_param, config.ckpt_path, config.logging_frequency)]
    epoch = dataset.get_dataset_size() // config.logging_frequency
    model.train(epoch, dataset, callbacks=cb, sink_size=config.logging_frequency)
