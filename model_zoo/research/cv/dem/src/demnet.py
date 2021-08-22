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
"""
DEMNet, WithLossCell and TrainOneStepCell
"""
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context
from mindspore.common.initializer import Normal
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.parallel._utils import _get_gradients_mean, _get_parallel_mode, _get_device_num
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

class MyTanh(nn.Cell):
    def __init__(self):
        super(MyTanh, self).__init__()
        self.tanh = P.Tanh()

    def construct(self, x):
        return  1.7159 * self.tanh(2 * x / 3)

class DEMNet1(nn.Cell):
    """cub+att"""
    def __init__(self):
        super(DEMNet1, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Dense(312, 700, weight_init=Normal(0.0008))
        self.fc2 = nn.Dense(700, 1024, weight_init=Normal(0.0012))

    def construct(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

class DEMNet2(nn.Cell):
    """awa+att"""
    def __init__(self):
        super(DEMNet2, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Dense(85, 700, weight_init=Normal(0.0005))
        self.fc2 = nn.Dense(700, 1024, weight_init=Normal(0.0005))

    def construct(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

class DEMNet3(nn.Cell):
    """awa+word"""
    def __init__(self):
        super(DEMNet3, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Dense(1000, 1024, weight_init=Normal(0.0005))

    def construct(self, x):
        x = self.relu(self.fc1(x))
        return x

class DEMNet4(nn.Cell):
    """awa+fusion"""
    def __init__(self):
        super(DEMNet4, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = MyTanh()
        self.fc1 = nn.Dense(1000, 900, weight_init=Normal(0.0008))
        self.fc2 = nn.Dense(85, 900, weight_init=Normal(0.0012))
        self.fc3 = nn.Dense(900, 1024, weight_init=Normal(0.0012))

    def construct(self, att, word):
        word = self.tanh(self.fc1(word))
        att = self.tanh(self.fc2(att))
        fus = word + 3 * att
        fus = self.relu(self.fc3(fus))
        return fus

class MyWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data1, data2, label):
        out = self._backbone(data1, data2)
        return self._loss_fn(out, label)

class MyTrainOneStepCell(nn.Cell):
    """custom TrainOneStepCell"""
    def __init__(self, network, optimizer, sens=1.0):
        super(MyTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (context.ParallelMode.DATA_PARALLEL, context.ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, mean, degree)

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        grads = ops.clip_by_global_norm(grads, 0.2)
        self.optimizer(grads)
        return loss
