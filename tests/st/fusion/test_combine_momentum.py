# Copyright 2023 Huawei Technologies Co., Ltd
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
import os
import numpy as np

from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.nn as nn
from mindspore import set_context
from mindspore import Tensor, FixedLossScaleManager
from mindspore import amp
from mindspore import ops
from mindspore.common import set_seed
from mindspore.nn import SoftmaxCrossEntropyWithLogits

set_seed(1)


class FusedMomentumNet(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=2,
                              stride=1,
                              has_bias=False,
                              weight_init='normal',
                              pad_mode='same')
        self.dense = nn.Dense(in_channels=in_channel,
                              out_channels=out_channel,
                              weight_init='normal',
                              bias_init='normal',
                              has_bias=True)
        self.mean = ops.ReduceMean(keep_dims=False)

    def construct(self, input_x):
        tmp_x = self.conv(input_x)
        tmp_y = self.relu(tmp_x)
        tmp_z = self.mean(tmp_y, (2, 3))
        output = self.dense(tmp_z)
        return output


def clear_files(path):
    os.system("rm " + path)


def find_files(para, file):
    output = os.popen("grep '%s' %s | wc -l" % (para, file))
    out = str(output.read().encode(), 'utf-8').strip()
    return out


SAVE_GRAPHS_PATH = "./ir_path"


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ir_fusion_combine_weight_decay_scale_momentum_weight():
    """"
    Feature: Test weight decay scale momentum fusion
    Description: Test weight decay scale momentum fusion
    Expectation: The results are as expected
    """
    set_context(mode=ms.GRAPH_MODE, device_target="GPU", save_graphs=True, save_graphs_path=SAVE_GRAPHS_PATH)
    clear_files(SAVE_GRAPHS_PATH + "/verbose_ir_files/*")
    batch_size = 1
    num_classes = 3

    input_np = np.random.uniform(0.0, 1.0, size=[batch_size, 3, 2, 2]).astype(np.float16)
    label_np = np.ones([batch_size, num_classes]).astype(np.float32)
    net = FusedMomentumNet(3, num_classes)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    group_params = [{
        'params': conv_params,
        'weight_decay': 0.3
    }, {
        'params': no_conv_params,
        'lr': 0.04
    }, {
        'order_params': net.trainable_params()
    }]
    opt = nn.Momentum(group_params, learning_rate=0.03, momentum=0.9, loss_scale=1.3, weight_decay=0.7)
    lsm = FixedLossScaleManager(loss_scale=1.3, drop_overflow_update=False)
    net = amp.build_train_network(net, opt, loss, level="O3", loss_scale_manager=lsm)
    net.set_train()
    net(Tensor(input_np), Tensor(label_np))
    result = find_files('CombineWeightDecayScaleMomentum', os.path.join(SAVE_GRAPHS_PATH + "/verbose_ir_files",
                                                                        'hwopt*combine_optimizer*ir'))
    assert result == '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ir_fusion_combine_momentum():
    """"
    Feature: Test momentum fusion
    Description: Test momentum fusion
    Expectation: The results are as expected
    """
    set_context(mode=ms.GRAPH_MODE, device_target="GPU", save_graphs=True, save_graphs_path=SAVE_GRAPHS_PATH)
    clear_files(SAVE_GRAPHS_PATH + "/verbose_ir_files/*")
    batch_size = 1
    num_classes = 3

    input_np = np.random.uniform(0.0, 1.0, size=[batch_size, 3, 2, 2]).astype(np.float32)
    label_np = np.ones([batch_size, num_classes]).astype(np.float32)
    net = FusedMomentumNet(3, num_classes)
    loss = SoftmaxCrossEntropyWithLogits(sparse=False)
    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    group_params = [{
        'params': conv_params,
        'lr': 0.2
    }, {
        'params': no_conv_params,
        'lr': 0.3
    }, {
        'order_params': net.trainable_params()
    }]
    opt = nn.Momentum(group_params, learning_rate=0.03, momentum=0.9)
    net = nn.WithLossCell(net, loss)
    net = nn.TrainOneStepCell(net, opt)
    net.set_train()
    net(Tensor(input_np), Tensor(label_np))
    result = find_files('CombineMomentum', os.path.join(SAVE_GRAPHS_PATH + "/verbose_ir_files",
                                                        'hwopt*combine_optimizer*ir'))
    assert result == '2'
