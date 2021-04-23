"""loss"""
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

import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

class SupConLoss(nn.Cell):
    """SupConLoss"""
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
        contrast_feature = self.cat(self.unbind(features))
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
        return loss, anchor_count


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
