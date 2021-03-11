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
"""CTPN training network wrapper."""

import time
import numpy as np
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore import ParameterTuple
from mindspore.train.callback import Callback
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

time_stamp_init = False
time_stamp_first = 0
class LossCallBack(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF terminating training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1, rank_id=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.count = 0
        self.rpn_loss_sum = 0
        self.rpn_cls_loss_sum = 0
        self.rpn_reg_loss_sum = 0
        self.rank_id = rank_id

        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = time.time()
            time_stamp_init = True

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        rpn_loss = cb_params.net_outputs[0].asnumpy()
        rpn_cls_loss = cb_params.net_outputs[1].asnumpy()
        rpn_reg_loss = cb_params.net_outputs[2].asnumpy()

        self.count += 1
        self.rpn_loss_sum += float(rpn_loss)
        self.rpn_cls_loss_sum += float(rpn_cls_loss)
        self.rpn_reg_loss_sum += float(rpn_reg_loss)

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if self.count >= 1:
            global time_stamp_first
            time_stamp_current = time.time()
            rpn_loss = self.rpn_loss_sum / self.count
            rpn_cls_loss = self.rpn_cls_loss_sum / self.count
            rpn_reg_loss = self.rpn_reg_loss_sum / self.count
            loss_file = open("./loss_{}.log".format(self.rank_id), "a+")
            loss_file.write("%lu epoch: %s step: %s ,rpn_loss: %.5f, rpn_cls_loss: %.5f, rpn_reg_loss: %.5f"%
                            (time_stamp_current - time_stamp_first, cb_params.cur_epoch_num, cur_step_in_epoch,
                             rpn_loss, rpn_cls_loss, rpn_reg_loss))
            loss_file.write("\n")
            loss_file.close()

class LossNet(nn.Cell):
    """CTPN loss method"""
    def construct(self, x1, x2, x3):
        return x1

class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
    """
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, x, gt_bbox, gt_label, gt_num, img_shape=None):
        rpn_loss, _, _, rpn_cls_loss, rpn_reg_loss = self._backbone(x, gt_bbox, gt_label, gt_num, img_shape)
        return self._loss_fn(rpn_loss, rpn_cls_loss, rpn_reg_loss)

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, return backbone network.
        """
        return self._backbone


class TrainOneStepCell(nn.Cell):
    """
    Network training package class.

    Append an optimizer to the training network after that the construct function
    can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        network_backbone (Cell): The forward network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default value is 1.0.
        reduce_flag (bool): The reduce flag. Default value is False.
        mean (bool): Allreduce method. Default value is False.
        degree (int): Device number. Default value is None.
    """
    def __init__(self, network, network_backbone, optimizer, sens=1.0, reduce_flag=False, mean=True, degree=None):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.backbone = network_backbone
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.sens = Tensor((np.ones((1,)) * sens).astype(np.float32))
        self.reduce_flag = reduce_flag
        if reduce_flag:
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, x, gt_bbox, gt_label, gt_num, img_shape=None):
        weights = self.weights
        rpn_loss, _, _, rpn_cls_loss, rpn_reg_loss = self.backbone(x, gt_bbox, gt_label, gt_num, img_shape)
        grads = self.grad(self.network, weights)(x, gt_bbox, gt_label, gt_num, img_shape, self.sens)
        if self.reduce_flag:
            grads = self.grad_reducer(grads)
        return F.depend(rpn_loss, self.optimizer(grads)), rpn_cls_loss, rpn_reg_loss
