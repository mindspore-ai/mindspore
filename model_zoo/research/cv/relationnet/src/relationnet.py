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
"""relationnet"""

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import context
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.parallel._utils import _get_gradients_mean, _get_parallel_mode, _get_device_num
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from .config import relationnet_cfg as cfg


class Encoder_Relation(nn.Cell):
    """docstring for ClassName"""

    def __init__(self, input_size, hidden_size):
        super(Encoder_Relation, self).__init__()

        #init operations
        self.tile = ops.Tile()
        self.expand_dims = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.concat2dim = ops.Concat(axis=2)
        self.relu = ops.ReLU()
        self.sigmoid = ops.Sigmoid()
        self.reshape = ops.Reshape()
        self.stack = ops.Stack(0)
        self.concat0dim = ops.Concat(axis=0)
        self.class_num = cfg.class_num
        self.feature_dim = cfg.feature_dim

        #CNNEncoder-Network
        self.Encoderlayer1 = nn.SequentialCell(
            nn.Conv2d(1, 64, kernel_size=3, pad_mode='pad', padding=0, has_bias=True),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
        self.Encoderlayer2 = nn.SequentialCell(
            nn.Conv2d(64, 64, kernel_size=3, pad_mode='pad', padding=0, has_bias=True),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
        self.Encoderlayer3 = nn.SequentialCell(
            nn.Conv2d(64, 64, kernel_size=3, pad_mode='pad', padding=1, has_bias=True),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        self.Encoderlayer4 = nn.SequentialCell(
            nn.Conv2d(64, 64, kernel_size=3, pad_mode='pad', padding=1, has_bias=True),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())

        #Relation-Network
        self.Relationlayer1 = nn.SequentialCell(
            nn.Conv2d(128, 64, kernel_size=3, pad_mode='pad', padding=1, has_bias=True),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
        self.Relationlayer2 = nn.SequentialCell(
            nn.Conv2d(64, 64, kernel_size=3, pad_mode='pad', padding=1, has_bias=True),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
        self.fc1 = nn.Dense(input_size, hidden_size)
        self.fc2 = nn.Dense(hidden_size, 1)

    def construct(self, x):  # modified forward->construct
        '''construct'''
        sample = x[:5, :, :, :]
        batch = x[5:, :, :, :]
        s_out = self.Encoderlayer1(sample)
        s_out = self.Encoderlayer2(s_out)
        s_out = self.Encoderlayer3(s_out)
        s_out = self.Encoderlayer4(s_out)
        b_out = self.Encoderlayer1(batch)
        b_out = self.Encoderlayer2(b_out)
        b_out = self.Encoderlayer3(b_out)
        b_out = self.Encoderlayer4(b_out)
        sample_features, batch_features = s_out, b_out

        if batch_features.shape[0] == 95:
            sample_features_ext_list1 = []
            sample_features_ext_list2 = []
            for _ in range(45):
                sample_features_ext_list1.append(sample_features)
            sample_features_ext1 = self.stack(sample_features_ext_list1)
            for _ in range(50):
                sample_features_ext_list2.append(sample_features)
            sample_features_ext2 = self.stack(sample_features_ext_list2)
            sample_features_ext = self.concat0dim((sample_features_ext1, sample_features_ext2))
            batch_features_ext_list = []
            for _ in range(5):
                batch_features_ext_list.append(batch_features)
            batch_features_ext = self.stack(batch_features_ext_list)
            batch_features_ext = self.transpose(batch_features_ext, (1, 0, 2, 3, 4))

        else:
            sample_features_ext_list = []
            batch_features_ext_list = []
            for _ in range(5):
                sample_features_ext_list.append(sample_features)
                batch_features_ext_list.append(batch_features)
            sample_features_ext = self.stack(sample_features_ext_list)
            batch_features_ext = self.stack(batch_features_ext_list)
            batch_features_ext = self.transpose(batch_features_ext, (1, 0, 2, 3, 4))

        relation_pairs = self.concat2dim((sample_features_ext, batch_features_ext))
        relation_pairs = self.reshape(relation_pairs, (-1, self.feature_dim*2, 5, 5))


        #put relation pairs into relation network
        x = relation_pairs
        out = self.Relationlayer1(x)
        out = self.Relationlayer2(out)
        out = self.reshape(out, (out.shape[0], -1))
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        out = self.reshape(out, (-1, self.class_num))
        return out

def weight_init(custom_cell):
    '''weight_init'''
    for _, m in custom_cell.cells_and_names():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.set_data(Tensor(np.random.normal(0, np.sqrt(2. / n),
                                                      m.weight.data.shape).astype("float32")))
            if m.bias is not None:
                m.bias.set_data(
                    Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
        elif isinstance(m, nn.BatchNorm2d):
            m.gamma.set_data(
                Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
            m.beta.set_data(
                Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
        elif isinstance(m, nn.Dense):
            m.weight.set_data(Tensor(np.random.normal(
                0, 0.01, m.weight.data.shape).astype("float32")))
            if m.bias is not None:
                m.bias.set_data(
                    Tensor(np.ones(m.bias.data.shape, dtype="float32")))


class TrainOneStepCell(nn.Cell):
    '''TrainOneStep'''
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=False)
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
        grads = self.grad(self.network, weights)(*inputs)
        grads = self.grad_reducer(grads)
        grads = ops.clip_by_global_norm(grads, 0.5)
        return F.depend(loss, self.optimizer(grads))
