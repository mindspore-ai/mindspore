# Copyright 2019 Huawei Technologies Co., Ltd
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
""" test model train """
import os
import numpy as np

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore import Tensor, Parameter, Model
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim import Momentum
from mindspore.common.api import ms_function
import mindspore.nn as wrap
import mindspore.context as context
from apply_momentum import ApplyMomentum
from mindspore.train.summary.summary_record import SummaryRecord

CUR_DIR = os.getcwd()
SUMMARY_DIR = CUR_DIR + "/test_temp_summary_event_file/"

context.set_context(device_target="Ascend")


class MsWrapper(nn.Cell):
    def __init__(self, network):
        super(MsWrapper, self).__init__(auto_prefix=False)
        self._network = network

    @ms_function
    def construct(self, *args):
        return self._network(*args)


def me_train_tensor(net, input_np, label_np, epoch_size=2):
    context.set_context(mode=context.GRAPH_MODE)
    loss = SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
    opt = ApplyMomentum(Tensor(np.array([0.1])), Tensor(np.array([0.9])),
                        filter(lambda x: x.requires_grad, net.get_parameters()))
    Model(net, loss, opt)
    _network = wrap.WithLossCell(net, loss)
    _train_net = MsWrapper(wrap.TrainOneStepCell(_network, opt))
    _train_net.set_train()
    with SummaryRecord(SUMMARY_DIR, file_suffix="_MS_GRAPH", network=_train_net) as summary_writer:
        for epoch in range(0, epoch_size):
            print(f"epoch %d" % (epoch))
            output = _train_net(Tensor(input_np), Tensor(label_np))
            summary_writer.record(i)
            print("********output***********")
            print(output.asnumpy())


def me_infer_tensor(net, input_np):
    net.set_train()
    net = MsWrapper(net)
    output = net(Tensor(input_np))
    return output


def test_net():
    class Net(nn.Cell):
        def __init__(self, cin, cout):
            super(Net, self).__init__()
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
            self.conv = nn.Conv2d(cin, cin, kernel_size=1, stride=1, padding=0, has_bias=False, pad_mode="same")
            self.bn = nn.BatchNorm2d(cin, momentum=0.1, eps=0.0001)
            self.add = P.TensorAdd()
            self.relu = P.ReLU()
            self.mean = P.ReduceMean(keep_dims=True)
            self.reshape = P.Reshape()
            self.dense = nn.Dense(cin, cout)

        def construct(self, input_x):
            output = input_x
            output = self.maxpool(output)
            identity = output
            output = self.conv(output)
            output = self.bn(output)
            output = self.add(output, identity)
            output = self.relu(output)
            output = self.mean(output, (-2, -1))
            output = self.reshape(output, (32, -1))
            output = self.dense(output)
            return output

    net = Net(2048, 1001)
    input_np = np.ones([32, 2048, 14, 14]).astype(np.float32) * 0.01
    label_np = np.ones([32]).astype(np.int32)
    me_train_tensor(net, input_np, label_np)
    #me_infer_tensor(net, input_np)
