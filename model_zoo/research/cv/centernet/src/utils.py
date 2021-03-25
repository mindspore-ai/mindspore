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
"""
Functional Cells to be used.
"""

import math
import time
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.common import dtype as mstype
from mindspore.train.callback import Callback
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR


clip_grad = ops.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Tensor")
def _clip_grad(clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    dt = ops.dtype(grad)
    new_grad = nn.ClipByNorm()(grad, ops.cast(ops.tuple_to_array((clip_value,)), dt))
    return new_grad


class ClipByNorm(nn.Cell):
    """
    Clip grads by gradient norm

    Args:
        clip_norm(float): The target norm of graident clip. Default: 1.0

    Returns:
        Tuple of Tensors, gradients after clip.
    """
    def __init__(self, clip_norm=1.0):
        super(ClipByNorm, self).__init__()
        self.hyper_map = ops.HyperMap()
        self.clip_norm = clip_norm

    def construct(self, grads):
        grads = self.hyper_map(ops.partial(clip_grad, self.clip_norm), grads)
        return grads


reciprocal = ops.Reciprocal()
grad_scale = ops.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


class GradScale(nn.Cell):
    """
    Gradients scale

    Args: None

    Returns:
        Tuple of Tensors, gradients after rescale.
    """
    def __init__(self):
        super(GradScale, self).__init__()
        self.hyper_map = ops.HyperMap()

    def construct(self, scale, grads):
        grads = self.hyper_map(ops.partial(grad_scale, scale), grads)
        return grads


class ClipByValue(nn.Cell):
    """
    Clip tensor by value

    Args: None

    Returns:
        Tensor, output after clip.
    """
    def __init__(self):
        super(ClipByValue, self).__init__()
        self.min = ops.Minimum()
        self.max = ops.Maximum()

    def construct(self, x, clip_value_min, clip_value_max):
        x_min = self.min(x, clip_value_max)
        x_max = self.max(x_min, clip_value_min)
        return x_max


class GatherFeature(nn.Cell):
    """
    Gather feature at specified position

    Args:
        enable_cpu_gather (bool): Use cpu operator GatherD to gather feature or not, adaption for CPU. Default: True.

    Returns:
        Tensor, feature at spectified position
    """
    def __init__(self, enable_cpu_gather=True):
        super(GatherFeature, self).__init__()
        self.tile = ops.Tile()
        self.shape = ops.Shape()
        self.concat = ops.Concat(axis=1)
        self.reshape = ops.Reshape()
        self.enable_cpu_gather = enable_cpu_gather
        if self.enable_cpu_gather:
            self.gather_nd = ops.GatherD()
            self.expand_dims = ops.ExpandDims()
        else:
            self.gather_nd = ops.GatherNd()

    def construct(self, feat, ind):
        """gather by specified index"""
        if self.enable_cpu_gather:
            _, _, c = self.shape(feat)
            # (b, N, c)
            index = self.expand_dims(ind, -1)
            index = self.tile(index, (1, 1, c))
            feat = self.gather_nd(feat, 1, index)
        else:
            # (b, N)->(b*N, 1)
            b, N = self.shape(ind)
            ind = self.reshape(ind, (-1, 1))
            ind_b = nn.Range(0, b, 1)()
            ind_b = self.reshape(ind_b, (-1, 1))
            ind_b = self.tile(ind_b, (1, N))
            ind_b = self.reshape(ind_b, (-1, 1))
            index = self.concat((ind_b, ind))
            # (b, N, 2)
            index = self.reshape(index, (b, N, -1))
            # (b, N, c)
            feat = self.gather_nd(feat, index)
        return feat


class TransposeGatherFeature(nn.Cell):
    """
    Transpose and gather feature at specified position

    Args: None

    Returns:
        Tensor, feature at spectified position
    """
    def __init__(self):
        super(TransposeGatherFeature, self).__init__()
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.perm_list = (0, 2, 3, 1)
        self.gather_feat = GatherFeature()

    def construct(self, feat, ind):
        # (b, c, h, w)->(b, h*w, c)
        feat = self.transpose(feat, self.perm_list)
        b, _, _, c = self.shape(feat)
        feat = self.reshape(feat, (b, -1, c))
        # (b, N, c)
        feat = self.gather_feat(feat, ind)
        return feat


class Sigmoid(nn.Cell):
    """
    Sigmoid and then Clip by value

    Args: None

    Returns:
        Tensor, feature after sigmoid and clip.
    """
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.cast = ops.Cast()
        self.dtype = ops.DType()
        self.sigmoid = nn.Sigmoid()
        self.clip_by_value = ops.clip_by_value

    def construct(self, x, min_value=1e-4, max_value=1-1e-4):
        x = self.sigmoid(x)
        dt = self.dtype(x)
        x = self.clip_by_value(x, self.cast(ops.tuple_to_array((min_value,)), dt),
                               self.cast(ops.tuple_to_array((max_value,)), dt))
        return x


class FocalLoss(nn.Cell):
    """
    Warpper for focal loss.

    Args:
        alpha(int): Super parameter in focal loss to mimic loss weight. Default: 2.
        beta(int): Super parameter in focal loss to mimic imbalance between positive and negative samples. Default: 4.

    Returns:
        Tensor, focal loss.
    """
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.pow = ops.Pow()
        self.log = ops.Log()
        self.select = ops.Select()
        self.equal = ops.Equal()
        self.less = ops.Less()
        self.cast = ops.Cast()
        self.fill = ops.Fill()
        self.dtype = ops.DType()
        self.shape = ops.Shape()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, out, target):
        """focal loss"""
        pos_inds = self.cast(self.equal(target, 1.0), mstype.float32)
        neg_inds = self.cast(self.less(target, 1.0), mstype.float32)
        neg_weights = self.pow(1 - target, self.beta)

        pos_loss = self.log(out) * self.pow(1 - out, self.alpha) * pos_inds
        neg_loss = self.log(1 - out) * self.pow(out, self.alpha) * neg_weights * neg_inds

        num_pos = self.reduce_sum(pos_inds, ())
        num_pos = self.select(self.equal(num_pos, 0.0),
                              self.fill(self.dtype(num_pos), self.shape(num_pos), 1.0), num_pos)
        pos_loss = self.reduce_sum(pos_loss, ())
        neg_loss = self.reduce_sum(neg_loss, ())
        loss = - (pos_loss + neg_loss) / num_pos
        return loss


class GHMCLoss(nn.Cell):
    """
    Warpper for gradient harmonizing loss for classification.

    Args:
        bins(int): Number of bins. Default: 10.
        momentum(float): Momentum for moving gradient density. Default: 0.0.

    Returns:
        Tensor, GHM loss for classification.
    """
    def __init__(self, bins=10, momentum=0.0):
        super(GHMCLoss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges_left = np.array([float(x) / bins for x in range(bins)], dtype=np.float32)
        self.edges_left = Tensor(edges_left.reshape((bins, 1, 1, 1, 1)))
        edges_right = np.array([float(x) / bins for x in range(1, bins + 1)], dtype=np.float32)
        edges_right[-1] += 1e-4
        self.edges_right = Tensor(edges_right.reshape((bins, 1, 1, 1, 1)))

        if momentum >= 0:
            self.acc_sum = Parameter(initializer(0, [bins], mstype.float32))

        self.abs = ops.Abs()
        self.log = ops.Log()
        self.cast = ops.Cast()
        self.select = ops.Select()
        self.reshape = ops.Reshape()
        self.reduce_sum = ops.ReduceSum()
        self.max = ops.Maximum()
        self.less = ops.Less()
        self.equal = ops.Equal()
        self.greater = ops.Greater()
        self.logical_and = ops.LogicalAnd()
        self.greater_equal = ops.GreaterEqual()
        self.zeros_like = ops.ZerosLike()
        self.expand_dims = ops.ExpandDims()

    def construct(self, out, target):
        """GHM loss for classification"""
        g = self.abs(out - target)
        g = self.expand_dims(g, 0) # (1, b, c, h, w)

        pos_inds = self.cast(self.equal(target, 1.0), mstype.float32)
        tot = self.max(self.reduce_sum(pos_inds, ()), 1.0)

        # (bin, b, c, h, w)
        inds_mask = self.logical_and(self.greater_equal(g, self.edges_left), self.less(g, self.edges_right))
        zero_matrix = self.cast(self.zeros_like(inds_mask), mstype.float32)
        inds = self.cast(inds_mask, mstype.float32)
        # (bins,)
        num_in_bin = self.reduce_sum(inds, (1, 2, 3, 4))
        valid_bins = self.greater(num_in_bin, 0)
        num_valid_bin = self.reduce_sum(self.cast(valid_bins, mstype.float32), ())

        if self.momentum > 0:
            self.acc_sum = self.select(valid_bins,
                                       self.momentum * self.acc_sum + (1 - self.momentum) * num_in_bin,
                                       self.acc_sum)
            acc_sum = self.acc_sum
            acc_sum = self.reshape(acc_sum, (self.bins, 1, 1, 1, 1))
            acc_sum = acc_sum + zero_matrix
            weights = self.select(self.equal(inds, 1), tot / acc_sum, zero_matrix)
            # (b, c, h, w)
            weights = self.reduce_sum(weights, 0)
        else:
            num_in_bin = self.reshape(num_in_bin, (self.bins, 1, 1, 1, 1))
            num_in_bin = num_in_bin + zero_matrix
            weights = self.select(self.equal(inds, 1), tot / num_in_bin, zero_matrix)
            # (b, c, h, w)
            weights = self.reduce_sum(weights, 0)

        weights = weights / num_valid_bin

        ghmc_loss = (target - 1.0) * self.log(1.0 - out) - target * self.log(out)
        ghmc_loss = self.reduce_sum(ghmc_loss * weights, ()) / tot
        return ghmc_loss


class GHMRLoss(nn.Cell):
    """
    Warpper for gradient harmonizing loss for regression.

    Args:
        bins(int): Number of bins. Default: 10.
        momentum(float): Momentum for moving gradient density. Default: 0.0.
        mu(float): Super parameter for smoothed l1 loss. Default: 0.02.

    Returns:
        Tensor, GHM loss for regression.
    """
    def __init__(self, bins=10, momentum=0.0, mu=0.02):
        super(GHMRLoss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.mu = mu
        edges_left = np.array([float(x) / bins for x in range(bins)], dtype=np.float32)
        self.edges_left = Tensor(edges_left.reshape((bins, 1, 1, 1, 1)))
        edges_right = np.array([float(x) / bins for x in range(1, bins + 1)], dtype=np.float32)
        edges_right[-1] += 1e-4
        self.edges_right = Tensor(edges_right.reshape((bins, 1, 1, 1, 1)))

        if momentum >= 0:
            self.acc_sum = Parameter(initializer(0, [bins], mstype.float32))

        self.abs = ops.Abs()
        self.sqrt = ops.Sqrt()
        self.cast = ops.Cast()
        self.select = ops.Select()
        self.reshape = ops.Reshape()
        self.reduce_sum = ops.ReduceSum()
        self.max = ops.Maximum()
        self.less = ops.Less()
        self.equal = ops.Equal()
        self.greater = ops.Greater()
        self.logical_and = ops.LogicalAnd()
        self.greater_equal = ops.GreaterEqual()
        self.zeros_like = ops.ZerosLike()
        self.expand_dims = ops.ExpandDims()

    def construct(self, out, target):
        """GHM loss for regression"""
        # ASL1 loss
        diff = out - target
        # gradient length
        g = self.abs(diff / self.sqrt(self.mu * self.mu + diff * diff))
        g = self.expand_dims(g, 0) # (1, b, c, h, w)

        pos_inds = self.cast(self.equal(target, 1.0), mstype.float32)
        tot = self.max(self.reduce_sum(pos_inds, ()), 1.0)

        # (bin, b, c, h, w)
        inds_mask = self.logical_and(self.greater_equal(g, self.edges_left), self.less(g, self.edges_right))
        zero_matrix = self.cast(self.zeros_like(inds_mask), mstype.float32)
        inds = self.cast(inds_mask, mstype.float32)
        # (bins,)
        num_in_bin = self.reduce_sum(inds, (1, 2, 3, 4))
        valid_bins = self.greater(num_in_bin, 0)
        num_valid_bin = self.reduce_sum(self.cast(valid_bins, mstype.float32), ())

        if self.momentum > 0:
            self.acc_sum = self.select(valid_bins,
                                       self.momentum * self.acc_sum + (1 - self.momentum) * num_in_bin,
                                       self.acc_sum)
            acc_sum = self.acc_sum
            acc_sum = self.reshape(acc_sum, (self.bins, 1, 1, 1, 1))
            acc_sum = acc_sum + zero_matrix
            weights = self.select(self.equal(inds, 1), tot / acc_sum, zero_matrix)
            # (b, c, h, w)
            weights = self.reduce_sum(weights, 0)
        else:
            num_in_bin = self.reshape(num_in_bin, (self.bins, 1, 1, 1, 1))
            num_in_bin = num_in_bin + zero_matrix
            weights = self.select(self.equal(inds, 1), tot / num_in_bin, zero_matrix)
            # (b, c, h, w)
            weights = self.reduce_sum(weights, 0)

        weights = weights / num_valid_bin

        ghmr_loss = self.sqrt(diff * diff + self.mu * self.mu) - self.mu
        ghmr_loss = self.reduce_sum(ghmr_loss * weights, ()) / tot
        return ghmr_loss


class RegLoss(nn.Cell):
    """
    Warpper for regression loss.

    Args:
        mode(str): L1 or Smoothed L1 loss. Default: "l1"

    Returns:
        Tensor, regression loss.
    """
    def __init__(self, mode='l1'):
        super(RegLoss, self).__init__()
        self.reduce_sum = ops.ReduceSum()
        self.cast = ops.Cast()
        self.expand_dims = ops.ExpandDims()
        self.reshape = ops.Reshape()
        self.gather_feature = TransposeGatherFeature()
        if mode == 'l1':
            self.loss = nn.L1Loss(reduction='sum')
        elif mode == 'sl1':
            self.loss = nn.SmoothL1Loss()
        else:
            self.loss = None

    def construct(self, output, mask, ind, target):
        pred = self.gather_feature(output, ind)
        mask = self.cast(mask, mstype.float32)
        num = self.reduce_sum(mask, ())
        mask = self.expand_dims(mask, 2)
        target = target * mask
        pred = pred * mask
        regr_loss = self.loss(pred, target)
        regr_loss = regr_loss / (num + 1e-4)
        return regr_loss


class RegWeightedL1Loss(nn.Cell):
    """
    Warpper for weighted regression loss.

    Args: None

    Returns:
        Tensor, regression loss.
    """
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()
        self.reduce_sum = ops.ReduceSum()
        self.gather_feature = TransposeGatherFeature()
        self.cast = ops.Cast()
        self.l1_loss = nn.L1Loss(reduction='sum')

    def construct(self, output, mask, ind, target):
        pred = self.gather_feature(output, ind)
        mask = self.cast(mask, mstype.float32)
        num = self.reduce_sum(mask, ())
        loss = self.l1_loss(pred * mask, target * mask)
        loss = loss / (num + 1e-4)
        return loss


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.

    Args:
        dataset_size (int): Dataset size. Default: -1.
        enable_static_time (bool): enable static time cost, adaption for CPU. Default: False.
    """

    def __init__(self, dataset_size=-1, enable_static_time=False):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size
        self._enable_static_time = enable_static_time

    def step_begin(self, run_context):
        """
        Get beginning time of each step
        """
        self._begin_time = time.time()

    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num / self._dataset_size)
            if percent == 0:
                percent = 1
                epoch_num -= 1
            if self._enable_static_time:
                cur_time = time.time()
                time_per_step = cur_time - self._begin_time
                print("epoch: {}, current epoch percent: {}, step: {}, time per step: {} s, outputs are {}"
                      .format(int(epoch_num), "%.3f" % percent, cb_params.cur_step_num, "%.3f" % time_per_step,
                              str(cb_params.net_outputs)), flush=True)
            else:
                print("epoch: {}, current epoch percent: {}, step: {}, outputs are {}"
                      .format(int(epoch_num), "%.3f" % percent, cb_params.cur_step_num,
                              str(cb_params.net_outputs)), flush=True)
        else:
            print("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                               str(cb_params.net_outputs)), flush=True)


class CenterNetPolynomialDecayLR(LearningRateSchedule):
    """
    Warmup and polynomial decay learning rate for CenterNet network.

    Args:
        learning_rate(float): Initial learning rate.
        end_learning_rate(float): Final learning rate after decay.
        warmup_steps(int): Warmup steps.
        decay_steps(int): Decay steps.
        power(int): Learning rate decay factor.

    Returns:
        Tensor, learning rate in time.
    """
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super(CenterNetPolynomialDecayLR, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = ops.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = ops.Cast()

    def construct(self, global_step):
        decay_lr = self.decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


class CenterNetMultiEpochsDecayLR(LearningRateSchedule):
    """
    Warmup and multi-steps decay learning rate for CenterNet network.

    Args:
        learning_rate(float): Initial learning rate.
        warmup_steps(int): Warmup steps.
        multi_steps(list int): The steps corresponding to decay learning rate.
        steps_per_epoch(int): How many steps for each epoch.
        factor(int): Learning rate decay factor. Default: 10.

    Returns:
        Tensor, learning rate in time.
    """
    def __init__(self, learning_rate, warmup_steps, multi_epochs, steps_per_epoch, factor=10):
        super(CenterNetMultiEpochsDecayLR, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = MultiEpochsDecayLR(learning_rate, multi_epochs, steps_per_epoch, factor)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = ops.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = ops.Cast()

    def construct(self, global_step):
        decay_lr = self.decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


class MultiEpochsDecayLR(LearningRateSchedule):
    """
    Calculate learning rate base on multi epochs decay function.

    Args:
        learning_rate(float): Initial learning rate.
        multi_steps(list int): The steps corresponding to decay learning rate.
        steps_per_epoch(int): How many steps for each epoch.
        factor(int): Learning rate decay factor. Default: 10.

    Returns:
        Tensor, learning rate.
    """
    def __init__(self, learning_rate, multi_epochs, steps_per_epoch, factor=10):
        super(MultiEpochsDecayLR, self).__init__()
        if not isinstance(multi_epochs, (list, tuple)):
            raise TypeError("multi_epochs must be list or tuple.")
        self.multi_epochs = Tensor(np.array(multi_epochs, dtype=np.float32) * steps_per_epoch)
        self.num = len(multi_epochs)
        self.start_learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.factor = factor
        self.pow = ops.Pow()
        self.cast = ops.Cast()
        self.less_equal = ops.LessEqual()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, global_step):
        cur_step = self.cast(global_step, mstype.float32)
        epochs = self.cast(self.less_equal(self.multi_epochs, cur_step), mstype.float32)
        lr = self.start_learning_rate / self.pow(self.factor, self.reduce_sum(epochs, ()))
        return lr
