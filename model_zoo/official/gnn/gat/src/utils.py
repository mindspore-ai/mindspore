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
"""Utils for training gat"""
from mindspore import nn
from mindspore.common.parameter import ParameterTuple
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P


class MaskedSoftMaxLoss(nn.Cell):
    """Calculate masked softmax loss with l2 loss"""
    def __init__(self, num_class, label, mask, l2_coeff, params):
        super(MaskedSoftMaxLoss, self).__init__()
        self.num_class = num_class
        self.label = label
        self.mask = mask
        self.softmax = P.SoftmaxCrossEntropyWithLogits()
        self.reduce_mean = P.ReduceMean()
        self.cast = P.Cast()
        self.l2_coeff = l2_coeff
        self.params = ParameterTuple(list(param for param in params if param.name[-4:] != 'bias'))
        self.reduce_sum = P.ReduceSum()
        self.num_params = len(self.params)

    def construct(self, logits):
        """calc l2 loss"""
        l2_loss = 0
        for i in range(self.num_params):
            l2_loss = l2_loss + self.l2_coeff * P.L2Loss()(self.params[i])

        logits = P.Reshape()(logits, (-1, self.num_class))
        label = P.Reshape()(self.label, (-1, self.num_class))
        mask = P.Reshape()(self.mask, (-1,))

        logits = self.cast(logits, mstype.float32)
        loss = self.softmax(logits, label)[0]
        mask /= self.reduce_mean(mask)
        loss *= mask
        loss = self.reduce_mean(loss)
        l2_loss = P.Cast()(l2_loss, mstype.float32)
        return loss+l2_loss


class MaskedAccuracy(nn.Cell):
    """Calculate accuracy with mask"""
    def __init__(self, num_class, label, mask):
        super(MaskedAccuracy, self).__init__()
        self.argmax = P.Argmax(axis=1)
        self.cast = P.Cast()
        self.reduce_mean = P.ReduceMean()
        self.equal = P.Equal()
        self.num_class = num_class
        self.label = Tensor(label, dtype=mstype.float32)
        self.mask = Tensor(mask, dtype=mstype.float32)

    def construct(self, logits):
        """Calculate accuracy"""
        logits = P.Reshape()(logits, (-1, self.num_class))
        labels = P.Reshape()(self.label, (-1, self.num_class))
        mask = P.Reshape()(self.mask, (-1,))

        labels = self.cast(labels, mstype.float32)

        correct_prediction = self.equal(self.argmax(logits), self.argmax(labels))
        accuracy_all = self.cast(correct_prediction, mstype.float32)
        mask = self.cast(mask, mstype.float32)
        mask /= self.reduce_mean(mask)
        accuracy_all *= mask
        return self.reduce_mean(accuracy_all)


class LossAccuracyWrapper(nn.Cell):
    """
    Warp GAT model with loss calculation and accuracy calculation, loss is calculated with l2 loss.

    Args:
        network (Cell): GAT network with logits calculation as output.
        num_class (int): num of class for classification.
        label (numpy.ndarray): Train Dataset label.
        mask (numpy.ndarray): Train Dataset mask.
        l2_coeff (float): l2 loss discount rate.
    """
    def __init__(self, network, num_class, label, mask, l2_coeff):
        super(LossAccuracyWrapper, self).__init__()
        self.network = network
        label = Tensor(label, dtype=mstype.float32)
        mask = Tensor(mask, dtype=mstype.float32)
        self.loss_func = MaskedSoftMaxLoss(num_class, label, mask, l2_coeff, self.network.trainable_params())
        self.acc_func = MaskedAccuracy(num_class, label, mask)

    def construct(self, feature, biases):
        logits = self.network(feature, biases, training=False)
        loss = self.loss_func(logits)
        accuracy = self.acc_func(logits)
        return loss, accuracy


class LossNetWrapper(nn.Cell):
    """Wrap GAT model with loss calculation"""
    def __init__(self, network, num_class, label, mask, l2_coeff):
        super(LossNetWrapper, self).__init__()
        self.network = network
        label = Tensor(label, dtype=mstype.float32)
        mask = Tensor(mask, dtype=mstype.float32)
        params = list(param for param in self.network.trainable_params() if param.name[-4:] != 'bias')
        self.loss_func = MaskedSoftMaxLoss(num_class, label, mask, l2_coeff, params)

    def construct(self, feature, biases):
        logits = self.network(feature, biases)
        loss = self.loss_func(logits)
        return loss


class TrainOneStepCell(nn.Cell):
    """
    For network training. Warp the loss net with optimizer.

    Args:
        network (Cell): GAT network with loss calculation as the output.
        optimizer (Cell): Optimizer for minimize the loss.
        sens (Float): Backpropagation input number, default 1.0.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=True)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def construct(self, feature, biases):
        weights = self.weights
        loss = self.network(feature, biases)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(feature, biases, sens)
        return F.depend(loss, self.optimizer(grads))


class TrainGAT(nn.Cell):
    """
    Warp GAT model with everything needed for training, include loss, optimizer ,etc.

    Args:
        network (Cell): GAT network.
        num_class (int): num of class for classification.
        label (numpy.ndarray): Train Dataset label.
        mask (numpy.ndarray): Train Dataset mask.
        learning_rate (float): Learning rate.
        l2_coeff (float): l2 loss discount rate.
    """
    def __init__(self, network, num_class, label, mask, learning_rate, l2_coeff):
        super(TrainGAT, self).__init__(auto_prefix=False)
        self.network = network
        loss_net = LossNetWrapper(network, num_class, label, mask, l2_coeff)
        optimizer = nn.Adam(loss_net.trainable_params(),
                            learning_rate=learning_rate)
        self.loss_train_net = TrainOneStepCell(loss_net, optimizer)
        self.accuracy_func = MaskedAccuracy(num_class, label, mask)

    def construct(self, feature, biases):
        loss = self.loss_train_net(feature, biases)
        accuracy = self.accuracy_func(self.network(feature, biases))
        return loss, accuracy
