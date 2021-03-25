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
"""GPT training wrapper"""

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops.operations.comm_ops import _VirtualDataset
from mindspore.nn.wrap.loss_scale import TrainOneStepWithLossScaleCell
from src.utils import ClipByGlobalNorm

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in [0, 1]:
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(
            grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
            F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad,
                                   F.cast(F.tuple_to_array((clip_value,)),
                                          dt))
    return new_grad


grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


class VirtualDatasetOneInputCell(nn.Cell):
    def __init__(self, backbone):
        super(VirtualDatasetOneInputCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._virtual_dataset = _VirtualDataset()

    def construct(self, *data):
        data_ = self._virtual_dataset(*data)
        return self._backbone(*data_)

class PanGu1TrainOneStepWithLossScaleCell(TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of PanGu1 network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """
    def __init__(self,
                 network,
                 optimizer,
                 scale_update_cell=None,
                 enable_global_norm=False,
                 config=None):
        super(PanGu1TrainOneStepWithLossScaleCell,
              self).__init__(network, optimizer, scale_update_cell)
        self.network = network
        self.config = config
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.default_lr = Tensor([0.0], dtype=mstype.float32)
        self.enable_global_norm = enable_global_norm
        self.clip = ClipByGlobalNorm(self.weights)
        self.cast = P.Cast()

    def construct(self, input_ids, input_position=None, attention_mask=None, layer_past=None, sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids, input_position, attention_mask)
        scaling_sens = self.scale_sense

        # alloc status and clear should be right before gradoperation
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network,
                          weights)(input_ids,
                                   input_position, attention_mask,
                                   scaling_sens_filled)

        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(
            F.partial(grad_scale, scaling_sens), grads)

        if self.enable_global_norm:
            grads, _ = self.clip(grads)
        else:
            grads = self.hyper_map(
                F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE),
                grads)
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        return F.depend(loss, succ), cond, scaling_sens
