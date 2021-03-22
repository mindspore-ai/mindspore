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
CenterNet for traininig and evaluation
"""

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import Constant
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size
from mindspore import context
from .backbone_dla import DLASeg
from .utils import Sigmoid, GradScale
from .utils import FocalLoss, RegLoss, RegWeightedL1Loss
from .decode import MultiPoseDecode
from .config import dataset_config as data_cfg


def _generate_feature(cin, cout, kernel_size, head_name, head_conv=0):
    """
    Generate feature extraction function of each target head
    """
    fc = None
    if head_conv > 0:
        if 'hm' in head_name:
            conv2d = nn.Conv2d(head_conv, cout, kernel_size=kernel_size, has_bias=True, bias_init=Constant(-2.19))
        else:
            conv2d = nn.Conv2d(head_conv, cout, kernel_size=kernel_size, has_bias=True)
        fc = nn.SequentialCell([nn.Conv2d(cin, head_conv, kernel_size=3, has_bias=True), nn.ReLU(), conv2d])
    else:
        if 'hm' in head_name:
            fc = nn.Conv2d(cin, cout, kernel_size=kernel_size, has_bias=True, bias_init=Constant(-2.19))
        else:
            fc = nn.Conv2d(cin, cout, kernel_size=kernel_size, has_bias=True)
    return fc


class GatherMultiPoseFeatureCell(nn.Cell):
    """
    Gather features of multi-pose estimation.

    Args:
        net_config: The config info of CenterNet network.

    Returns:
        Tuple of Tensors, the target head of multi-person pose.
    """
    def __init__(self, net_config):
        super(GatherMultiPoseFeatureCell, self).__init__()
        head_conv = net_config.head_conv
        self.fc_heads = {}
        first_level = int(np.log2(net_config.down_ratio))
        self.dla_seg = DLASeg(net_config.down_ratio, net_config.last_level,
                              net_config.stage_levels, net_config.stage_channels)
        heads = {'hm': data_cfg.num_classes, 'wh': 2, 'hps': 2 * data_cfg.num_joints}
        if net_config.reg_offset:
            heads.update({'reg': 2})
        if net_config.hm_hp:
            heads.update({'hm_hp': data_cfg.num_joints})
        if net_config.reg_hp_offset:
            heads.update({'hp_offset': 2})

        in_channel = net_config.stage_channels[first_level]
        final_kernel = net_config.final_kernel
        self.hm_fn = _generate_feature(in_channel, heads['hm'], final_kernel, 'hm', head_conv)
        self.wh_fn = _generate_feature(in_channel, heads['wh'], final_kernel, 'wh', head_conv)
        self.hps_fn = _generate_feature(in_channel, heads['hps'], final_kernel, 'hps', head_conv)
        if net_config.reg_offset:
            self.reg_fn = _generate_feature(in_channel, heads['reg'], final_kernel, 'reg', head_conv)
        if net_config.hm_hp:
            self.hm_hp_fn = _generate_feature(in_channel, heads['hm_hp'], final_kernel, 'hm_hp', head_conv)
        if net_config.reg_hp_offset:
            self.reg_hp_fn = _generate_feature(in_channel, heads['hp_offset'], final_kernel, 'hp_offset', head_conv)
        self.sigmoid = Sigmoid()
        self.hm_hp = net_config.hm_hp
        self.reg_offset = net_config.reg_offset
        self.reg_hp_offset = net_config.reg_hp_offset
        self.not_enable_mse_loss = not net_config.mse_loss

    def construct(self, image):
        """Defines the computation performed."""
        output = self.dla_seg(image)

        output_hm = self.hm_fn(output)
        output_hm = self.sigmoid(output_hm)

        output_hps = self.hps_fn(output)
        output_wh = self.wh_fn(output)

        feature = (output_hm, output_hps, output_wh)

        if self.hm_hp:
            output_hm_hp = self.hm_hp_fn(output)
            if self.not_enable_mse_loss:
                output_hm_hp = self.sigmoid(output_hm_hp)
            feature += (output_hm_hp,)

        if self.reg_offset:
            output_reg = self.reg_fn(output)
            feature += (output_reg,)

        if self.reg_hp_offset:
            output_hp_offset = self.reg_hp_fn(output)
            feature += (output_hp_offset,)

        return feature


class CenterNetMultiPoseLossCell(nn.Cell):
    """
    Provide pose estimation network losses.

    Args:
        net_config: The config info of CenterNet network.

    Returns:
        Tensor, total loss.
    """
    def __init__(self, net_config):
        super(CenterNetMultiPoseLossCell, self).__init__()
        self.network = GatherMultiPoseFeatureCell(net_config)
        self.reduce_sum = ops.ReduceSum()
        self.crit = FocalLoss()
        self.crit_hm_hp = nn.MSELoss() if net_config.mse_loss else self.crit
        self.crit_kp = RegWeightedL1Loss() if not net_config.dense_hp else nn.L1Loss(reduction='sum')
        self.crit_reg = RegLoss(net_config.reg_loss)
        self.hm_weight = net_config.hm_weight
        self.hm_hp_weight = net_config.hm_hp_weight
        self.hp_weight = net_config.hp_weight
        self.wh_weight = net_config.wh_weight
        self.off_weight = net_config.off_weight
        self.hm_hp = net_config.hm_hp
        self.dense_hp = net_config.dense_hp
        self.reg_offset = net_config.reg_offset
        self.reg_hp_offset = net_config.reg_hp_offset
        self.hm_hp_ind = 3 if self.hm_hp else 2
        self.reg_ind = self.hm_hp_ind + 1 if self.reg_offset else self.hm_hp_ind
        self.reg_hp_ind = self.reg_ind + 1 if self.reg_hp_offset else self.reg_ind
        # just used for check
        self.print = ops.Print()
        self.concat = ops.Concat(axis=1)
        self.reshape = ops.Reshape()

    def construct(self, image, hm, reg_mask, ind, wh, kps, kps_mask, reg,
                  hm_hp, hp_offset, hp_ind, hp_mask):
        """Defines the computation performed."""
        feature = self.network(image)

        output_hm = feature[0]
        hm_loss = self.crit(output_hm, hm)

        output_hps = feature[1]
        if self.dense_hp:
            mask_weight = self.reduce_sum(kps_mask, ()) + 1e-4
            hp_loss = self.crit_kp(output_hps * kps_mask, kps * kps_mask) / mask_weight
        else:
            hp_loss = self.crit_kp(output_hps, kps_mask, ind, kps)

        output_wh = feature[2]
        wh_loss = self.crit_reg(output_wh, reg_mask, ind, wh)

        hm_hp_loss = 0
        if self.hm_hp and self.hm_hp_weight > 0:
            output_hm_hp = feature[self.hm_hp_ind]
            hm_hp_loss = self.crit_hm_hp(output_hm_hp, hm_hp)

        off_loss = 0
        if self.reg_offset and self.off_weight > 0:
            output_reg = feature[self.reg_ind]
            off_loss = self.crit_reg(output_reg, reg_mask, ind, reg)

        hp_offset_loss = 0
        if self.reg_hp_offset and self.off_weight > 0:
            output_hp_offset = feature[self.reg_hp_ind]
            hp_offset_loss = self.crit_reg(output_hp_offset, hp_mask, hp_ind, hp_offset)

        total_loss = (self.hm_weight * hm_loss + self.wh_weight * wh_loss +
                      self.off_weight * off_loss + self.hp_weight * hp_loss +
                      self.hm_hp_weight * hm_hp_loss + self.off_weight * hp_offset_loss)
        return total_loss


class CenterNetWithoutLossScaleCell(nn.Cell):
    """
    Encapsulation class of centernet training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.

    Returns:
        Tuple of Tensors, the loss, overflow flag and scaling sens of the network.
    """
    def __init__(self, network, optimizer):
        super(CenterNetWithoutLossScaleCell, self).__init__(auto_prefix=False)
        self.image = ImagePreProcess()
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=False)

    @ops.add_flags(has_effect=True)
    def construct(self, image, hm, reg_mask, ind, wh, kps, kps_mask, reg,
                  hm_hp, hp_offset, hp_ind, hp_mask):
        """Defines the computation performed."""
        image = self.image(image)
        weights = self.weights
        loss = self.network(image, hm, reg_mask, ind, wh, kps, kps_mask, reg,
                            hm_hp, hp_offset, hp_ind, hp_mask)

        grads = self.grad(self.network, weights)(image, hm, reg_mask, ind, wh, kps,
                                                 kps_mask, reg, hm_hp, hp_offset,
                                                 hp_ind, hp_mask)
        succ = self.optimizer(grads)
        ret = loss
        return ops.depend(ret, succ)


class CenterNetWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of centernet training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (number): Static loss scale. Default: 1.

    Returns:
        Tuple of Tensors, the loss, overflow flag and scaling sens of the network.
    """
    def __init__(self, network, optimizer, sens=1):
        super(CenterNetWithLossScaleCell, self).__init__(auto_prefix=False)
        self.image = ImagePreProcess()
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.reducer_flag = False
        self.allreduce = ops.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = ops.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = ops.Cast()
        self.alloc_status = ops.NPUAllocFloatStatus()
        self.get_status = ops.NPUGetFloatStatus()
        self.clear_status = ops.NPUClearFloatStatus()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = ops.LessEqual()
        self.grad_scale = GradScale()
        self.loss_scale = sens

    def construct(self, image, hm, reg_mask, ind, wh, kps, kps_mask, reg,
                  hm_hp, hp_offset, hp_ind, hp_mask):
        """Defines the computation performed."""
        image = self.image(image)
        weights = self.weights
        loss = self.network(image, hm, reg_mask, ind, wh, kps, kps_mask, reg,
                            hm_hp, hp_offset, hp_ind, hp_mask)
        scaling_sens = self.cast(self.loss_scale, mstype.float32) * 2.0 / 2.0
        # alloc status and clear should be right before gradoperation
        init = self.alloc_status()
        init = ops.depend(init, scaling_sens)
        clear_status = self.clear_status(init)
        scaling_sens = ops.depend(scaling_sens, clear_status)
        grads = self.grad(self.network, weights)(image, hm, reg_mask, ind, wh, kps,
                                                 kps_mask, reg, hm_hp, hp_offset,
                                                 hp_ind, hp_mask, scaling_sens)
        grads = self.grad_reducer(grads)
        grads = self.grad_scale(scaling_sens * self.degree, grads)
        init = ops.depend(init, grads)
        get_status = self.get_status(init)
        init = ops.depend(init, get_status)
        flag_sum = self.reduce_sum(init, (0,))
        if self.is_distributed:
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)

        succ = self.optimizer(grads)
        ret = (loss, cond, scaling_sens)
        return ops.depend(ret, succ)

class CenterNetMultiPoseEval(nn.Cell):
    """
    Encapsulation class of centernet testing.

    Args:
        net_config: The config info of CenterNet network.
        K(number): Max number of output objects. Default: 100.
        enable_nms_fp16(bool): Use float16 data for max_pool, adaption for CPU. Default: True.

    Returns:
        Tensor, detection of images(bboxes, score, keypoints and category id of each objects)
    """
    def __init__(self, net_config, K=100, enable_nms_fp16=True):
        super(CenterNetMultiPoseEval, self).__init__()
        self.network = GatherMultiPoseFeatureCell(net_config)
        self.decode = MultiPoseDecode(net_config, K, enable_nms_fp16)
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()

    def construct(self, image):
        """Calculate prediction scores"""
        features = self.network(image)
        detections = self.decode(features)
        return detections


class ImagePreProcess(nn.Cell):
    """
    Preprocess of image on device inplace of on host to improve performance.

    Args: None

    Returns:
        Tensor, normlized images and the format were converted to be NCHW
    """
    def __init__(self):
        super(ImagePreProcess, self).__init__()
        self.transpose = ops.Transpose()
        self.perm_list = (0, 3, 1, 2)
        self.mean = Tensor(data_cfg.mean.reshape((1, 1, 1, 3)))
        self.std = Tensor(data_cfg.std.reshape((1, 1, 1, 3)))
        self.cast = ops.Cast()

    def construct(self, image):
        image = self.cast(image, mstype.float32)
        image = (image / 255.0 - self.mean) / self.std
        image = self.transpose(image, self.perm_list)
        return image
