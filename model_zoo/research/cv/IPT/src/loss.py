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
"""loss"""
import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0


clip_grad = C.MultitypeFuncGraph("clip_grad")

@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type(int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value(float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class ClipGradients(nn.Cell):
    """
    Clip gradients.

    Args:
        grads (list): List of gradient tuples.
        clip_type (Tensor): The way to clip, 'value' or 'norm'.
        clip_value (Tensor): Specifies how much to clip.

    Returns:
        List, a list of clipped_grad tuples.
    """
    def __init__(self):
        super(ClipGradients, self).__init__()
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = P.Cast()
        self.dtype = P.DType()

    def construct(self, grads, clip_type, clip_value):
        """ClipGradients"""
        if clip_type not in (0, 1):
            return grads
        new_grads = ()
        for grad in grads:
            dt = self.dtype(grad)
            if clip_type == 0:
                t = C.clip_by_value(grad, self.cast(F.tuple_to_array((-clip_value,)), dt),
                                    self.cast(F.tuple_to_array((clip_value,)), dt))
            else:
                t = self.clip_by_norm(grad, self.cast(F.tuple_to_array((clip_value,)), dt))
            t = self.cast(t, dt)
            new_grads = new_grads + (t,)
        return new_grads

grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()

@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))

_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()

@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


class IPTTrainOneStepWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of bert network training.
     Append an optimizer to the training network after that the construct
     function can be called to create the backward graph.

     Args:
         network (Cell): The training network. Note that loss function should have been added.
         optimizer (Optimizer): Optimizer for updating the weights.
         scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(IPTTrainOneStepWithLossScaleCell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()
        self.degree = 1
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    def construct(self, lr, hr, idx, sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(lr, hr, idx)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(lr, hr, idx, self.cast(scaling_sens, mstype.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        cond = self.get_overflow_status(status, grads)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond, scaling_sens)


class SupConLoss(nn.Cell):
    """SupConLoss for contrastive learning."""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.normalize = P.L2Normalize(axis=2)
        self.eye = P.Eye()
        self.unbind = P.Unstack(axis=1)
        self.cat = P.Concat(axis=0)
        self.matmul = P.MatMul()
        self.div = P.Div()
        self.transpose = P.Transpose()
        self.maxes = P.ArgMaxWithValue(axis=1, keep_dims=True)
        self.tile = P.Tile()
        self.scatter = P.ScatterNd()
        self.oneslike = P.OnesLike()
        self.exp = P.Exp()
        self.sum = P.ReduceSum(keep_dims=True)
        self.log = P.Log()
        self.reshape = P.Reshape()
        self.mean = P.ReduceMean()

    def construct(self, features):
        """SupConLoss"""
        features = self.normalize(features)
        batch_size = features.shape[0]
        mask = self.eye(batch_size, batch_size, mstype.float32)
        contrast_count = features.shape[1]
        split_num = contrast_count //190
        contrast_feature = ()
        temp_feature = self.cat(self.unbind(features[:, :190, :]))
        contrast_feature += (temp_feature,)
        for num in range(split_num):
            temp_feature = self.cat(self.unbind(features[:, 190*(num+1):190*(num+2), :]))
            contrast_feature += (temp_feature,)
        contrast_feature = self.cat(contrast_feature)
        if self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            anchor_feature = features[:, 0]
            anchor_count = 1
        anchor_dot_contrast = self.div(self.matmul(anchor_feature, self.transpose(contrast_feature, (1, 0))), \
            self.temperature)
        _, logits_max = self.maxes(anchor_dot_contrast)
        logits = anchor_dot_contrast - logits_max
        mask = self.tile(mask, (anchor_count, contrast_count))
        logits_mask = 1 - self.eye(mask.shape[0], mask.shape[1], mstype.float32)
        mask = mask * logits_mask
        exp_logits = self.exp(logits) * logits_mask
        log_prob = logits - self.log(self.sum(exp_logits, 1) + 1e-8)
        mean_log_prob_pos = self.sum((mask * log_prob), 1) / self.sum(mask, 1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = self.mean(self.reshape(loss, (anchor_count, batch_size)))
        return loss
