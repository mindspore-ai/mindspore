# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import mindspore.nn as nn
import mindspore.ops as ops
#from batch_renorm import BatchRenormalization

from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.ops import functional as F

class BRDNet(nn.Cell):
    """
    args:
        channel: 3 for color, 1 for gray
    """
    def __init__(self, channel):
        super(BRDNet, self).__init__()

        self.Conv2d_1 = nn.Conv2d(channel, 64, kernel_size=(3, 3), stride=(1, 1), pad_mode='same', has_bias=True)
        self.BRN_1 = nn.BatchNorm2d(64, eps=1e-3)
        self.layer1 = self.make_layer1(15)
        self.Conv2d_2 = nn.Conv2d(64, channel, kernel_size=(3, 3), stride=(1, 1), pad_mode='same', has_bias=True)
        self.Conv2d_3 = nn.Conv2d(channel, 64, kernel_size=(3, 3), stride=(1, 1), pad_mode='same', has_bias=True)
        self.BRN_2 = nn.BatchNorm2d(64, eps=1e-3)
        self.layer2 = self.make_layer2(7)
        self.Conv2d_4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), pad_mode='same', has_bias=True)
        self.BRN_3 = nn.BatchNorm2d(64, eps=1e-3)
        self.layer3 = self.make_layer2(6)
        self.Conv2d_5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), pad_mode='same', has_bias=True)
        self.BRN_4 = nn.BatchNorm2d(64, eps=1e-3)
        self.Conv2d_6 = nn.Conv2d(64, channel, kernel_size=(3, 3), stride=(1, 1), pad_mode='same', has_bias=True)
        self.Conv2d_7 = nn.Conv2d(channel*2, channel, kernel_size=(3, 3), stride=(1, 1), pad_mode='same', has_bias=True)

        self.relu = nn.ReLU()
        self.sub = ops.Sub()
        self.concat = ops.Concat(axis=1)#NCHW

    def make_layer1(self, nums):
        layers = []
        assert nums > 0
        for _ in range(nums):
            layers.append(nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), pad_mode='same', has_bias=True))
            layers.append(nn.BatchNorm2d(64, eps=1e-3))
            layers.append(nn.ReLU())
        return nn.SequentialCell(layers)

    def make_layer2(self, nums):
        layers = []
        assert nums > 0
        for _ in range(nums):
            layers.append(nn.Conv2d(64, 64, kernel_size=(3, 3), \
                stride=(1, 1), dilation=(2, 2), pad_mode='same', has_bias=True))
            layers.append(nn.ReLU())
        return nn.SequentialCell(layers)

    def construct(self, inpt):
        #inpt-----> 'NCHW'
        x = self.Conv2d_1(inpt)
        x = self.BRN_1(x)
        x = self.relu(x)
        # 15 layers, Conv+BN+relu
        x = self.layer1(x)

        # last layer, Conv
        x = self.Conv2d_2(x) #for output channel, gray is 1 color is 3
        x = self.sub(inpt, x)   # input - noise

        y = self.Conv2d_3(inpt)
        y = self.BRN_2(y)
        y = self.relu(y)

        # first Conv+relu's
        y = self.layer2(y)

        y = self.Conv2d_4(y)
        y = self.BRN_3(y)
        y = self.relu(y)

        # second Conv+relu's
        y = self.layer3(y)

        y = self.Conv2d_5(y)
        y = self.BRN_4(y)
        y = self.relu(y)

        y = self.Conv2d_6(y)#for output channel, gray is 1 color is 3
        y = self.sub(inpt, y)   # input - noise

        o = self.concat((x, y))
        z = self.Conv2d_7(o)#gray is 1 color is 3
        z = self.sub(inpt, z)

        return z

class BRDWithLossCell(nn.Cell):
    def __init__(self, network):
        super(BRDWithLossCell, self).__init__()
        self.network = network
        self.loss = nn.MSELoss(reduction='sum') #we use 'sum' instead of 'mean' to avoid the loss becoming too small
    def construct(self, images, targets):
        output = self.network(images)
        return self.loss(output, targets)

class TrainingWrapper(nn.Cell):
    """Training wrapper."""
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))
