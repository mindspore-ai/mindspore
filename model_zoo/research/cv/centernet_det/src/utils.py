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
Functional Cells to be used.
"""

import math
import time
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR
from mindspore.train.callback import Callback

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


class GatherFeature(nn.Cell):
    """
    Gather feature at specified position

    Args:
        enable_cpu_gather (bool): Use cpu operator GatherD to gather feature or not, adaption for CPU. Default: False.

    Returns:
        Tensor, feature at spectified position
    """
    def __init__(self, enable_cpu_gather=False):
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


class RegLoss(nn.Cell): #reg_l1_loss
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
        # print('CenterNetMultiEpochsDecayLR:',lr.dtype)
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
        multi_epochs = self.cast(self.multi_epochs, mstype.float32)
        epochs = self.cast(self.less_equal(multi_epochs, cur_step), mstype.float32)
        lr = self.start_learning_rate / self.pow(self.factor, self.reduce_sum(epochs, ()))
        return lr
