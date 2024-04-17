# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindspore as ms
from mindspore import ops, nn


def train_one(network, optimizer, loss_scaler, grad_reducer):
    def forward_func(imgs, gt_class, gt_bbox):
        loss, rois_loss, bbox_losses = network(imgs, gt_class, gt_bbox)
        return loss_scaler.scale(loss), rois_loss, bbox_losses

    grad_fn = ops.value_and_grad(forward_func, grad_position=None, weights=optimizer.parameters, has_aux=True)

    def train_step_func(imgs, gt_class, gt_bbox):
        (loss, loss_rpn, loss_rcnn), grads = grad_fn(imgs, gt_class, gt_bbox)
        grads = loss_scaler.unscale(grads)
        loss = loss_scaler.unscale(loss)
        grads = grad_reducer(grads)
        loss = ops.depend(loss, optimizer(grads))
        return loss, loss_rpn, loss_rcnn

    return train_step_func


class TrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    def __init__(self, network, optimizer, loss_scaler, grad_reducer, clip_grads=False):
        scale_sense = ms.Tensor(256.0, ms.float32)
        super(TrainOneStepCell, self).__init__(network, optimizer, scale_sense)
        self.loss_scaler = loss_scaler
        self.grad_reducer = grad_reducer
        self.clip_grads = clip_grads

    def construct(self, *inputs):
        weights = self.weights
        outputs = self.network(*inputs)
        loss, loss_rpn, loss_rcnn = outputs
        status, scaling_sens = self.start_overflow_check(loss, self.loss_scaler.scale_value)
        sens_tuple = (ops.ones_like(loss) * scaling_sens,)
        for i in range(1, len(outputs)):
            sens_tuple += (ops.zeros_like(outputs[i]),)
        grads = self.grad(self.network, weights)(*inputs, sens_tuple)
        grads = self.loss_scaler.unscale(grads)
        if self.clip_grads:
            grads = ops.clip_by_global_norm(grads)
        grads = self.grad_reducer(grads)
        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            loss = ops.depend(loss, self.optimizer(grads))
        return loss, loss_rpn, loss_rcnn
