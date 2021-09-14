# Copyright 2020 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.nn.cell import Cell
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.context import ParallelMode, get_auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore import context
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from src.model_utils.config import config

context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
time_stamp_init = False
time_stamp_first = 0
grad_scale = C.MultitypeFuncGraph("grad_scale")
_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
reciprocal = P.Reciprocal()

GRADIENT_CLIP_TYPE = config.GRADIENT_CLIP_TYPE
GRADIENT_CLIP_VALUE = config.GRADIENT_CLIP_VALUE

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
        tuple[Tensor]: clipped gradients.
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

class MyLoss(Cell):
    """
    Base class for other losses.
    """
    def __init__(self, reduction='mean'):
        super(MyLoss, self).__init__()
        if reduction is None:
            reduction = 'none'

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction method for {reduction.lower()} is not supported")

        self.average = True
        self.reduce = True
        if reduction == 'sum':
            self.average = False
        if reduction == 'none':
            self.reduce = False

        self.reduce_mean = P.ReduceMean()
        self.reduce_sum = P.ReduceSum()
        self.mul = P.Mul()
        self.cast = P.Cast()

    def get_axis(self, x):
        shape = F.shape(x)
        length = F.tuple_len(shape)
        perm = F.make_range(0, length)
        return perm

    def get_loss(self, x, weights=1.0):
        """
        Computes the weighted loss
        Args:
            weights: Optional `Tensor` whose rank is either 0, or the same rank as inputs, and must be broadcastable to
                inputs (i.e., all dimensions must be either `1`, or the same as the corresponding inputs dimension).
        """
        input_dtype = x.dtype
        x = self.cast(x, mstype.float32)
        weights = self.cast(weights, mstype.float32)
        x = self.mul(weights, x)
        if self.reduce and self.average:
            x = self.reduce_mean(x, self.get_axis(x))
        if self.reduce and not self.average:
            x = self.reduce_sum(x, self.get_axis(x))
        x = self.cast(x, input_dtype)
        return x

    def construct(self, base, target):
        raise NotImplementedError

class openpose_loss(MyLoss):
    def __init__(self):
        super(openpose_loss, self).__init__()
        self.expand_dims = P.ExpandDims()
        self.tile = P.Tile()
        self.mul = P.Mul()
        self.l2_loss = P.L2Loss()
        self.square = P.Square()
        self.reduceMean = P.ReduceMean()
        self.reduceSum = P.ReduceSum()
        self.shape = P.Shape()
        self.maxoftensor = P.ArgMaxWithValue(-1)

    def mean_square_error(self, map1, map2, mask=None):
        if mask is None:
            mse = self.reduceMean((map1 - map2) ** 2)
            return mse

        squareMap = self.square(map1 - map2)
        squareMap_mask = self.mul(squareMap, mask)
        mse = self.reduceMean(squareMap_mask)
        return mse

    def construct(self, logit_paf, logit_heatmap, gt_paf, gt_heatmap, ignore_mask):
        # Input
        # ignore_mask, make sure the ignore_mask the 0-1 array instead of the bool-false array
        heatmaps_loss = []
        pafs_loss = []
        total_loss = 0

        paf_masks = self.tile(self.expand_dims(ignore_mask, 1), (1, self.shape(gt_paf)[1], 1, 1))
        heatmap_masks = self.tile(self.expand_dims(ignore_mask, 1), (1, self.shape(gt_heatmap)[1], 1, 1))

        paf_masks = F.stop_gradient(paf_masks)
        heatmap_masks = F.stop_gradient(heatmap_masks)
        for logit_paf_t, logit_heatmap_t in zip(logit_paf, logit_heatmap):
            pafs_loss_t = self.mean_square_error(logit_paf_t, gt_paf, paf_masks)
            heatmaps_loss_t = self.mean_square_error(logit_heatmap_t, gt_heatmap, heatmap_masks)

            total_loss += pafs_loss_t + heatmaps_loss_t
            heatmaps_loss.append(heatmaps_loss_t)
            pafs_loss.append(pafs_loss_t)

        return total_loss, heatmaps_loss, pafs_loss

class BuildTrainNetwork(nn.Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, gt_paf, gt_heatmap, mask):
        logit_pafs, logit_heatmap = self.network(input_data)
        loss, _, _ = self.criterion(logit_pafs, logit_heatmap, gt_paf, gt_heatmap, mask)
        return loss

class TrainOneStepWithClipGradientCell(nn.Cell):
    '''TrainOneStepWithClipGradientCell'''
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepWithClipGradientCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.hyper_map = C.HyperMap()
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = get_auto_parallel_context("parallel_mode")
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *inputs):

        weights = self.weights
        loss = self.network(*inputs)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss
