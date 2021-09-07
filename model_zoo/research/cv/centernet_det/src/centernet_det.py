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
CenterNet for training and evaluation
"""


import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from src.utils import Sigmoid, GradScale
from src.utils import FocalLoss, RegLoss
from src.decode import DetectionDecode
from src.hourglass import Convolution, Residual, Kp_module
from .model_utils.config import dataset_config as data_cfg
BN_MOMENTUM = 0.9


def _generate_feature(cin, cout, kernel_size, head, num_stacks, with_bn=True):
    """
    Generate hourglass network feature extraction function of each target head
    """
    module = None
    module = nn.CellList([
        nn.SequentialCell(
            Convolution(cin, cout, kernel_size, with_bn=with_bn),
            nn.Conv2d(cout, head, kernel_size=1, has_bias=True, pad_mode='pad')
        ) for _ in range(num_stacks)
    ])
    return module


class GatherDetectionFeatureCell(nn.Cell):
    """
    Gather hourglass features of object detection.

    Args:
        net_config: The config info of CenterNet network.

    Returns:
        Tuple of Tensors, the target head of object detection.
    """

    def __init__(self, net_config):
        super(GatherDetectionFeatureCell, self).__init__()
        self.nstack = net_config.num_stacks
        self.n = net_config.n
        self.cnv_dim = net_config.cnv_dim
        self.dims = net_config.dims
        self.modules = net_config.modules
        curr_dim = self.dims[0]
        self.heads = {'hm': data_cfg.num_classes, 'wh': 2}
        if net_config.reg_offset:
            self.heads.update({'reg': 2})

        self.pre = nn.SequentialCell(
            Convolution(3, 128, 7, stride=2),
            Residual(128, 256, 3, stride=2)
        )

        self.kps = nn.CellList([
            Kp_module(
                self.n, self.dims, self.modules
            ) for _ in range(self.nstack)
        ])

        self.cnvs = nn.CellList([
            Convolution(curr_dim, self.cnv_dim, 3) for _ in range(self.nstack)
        ])

        self.inters = nn.CellList([
            Residual(curr_dim, curr_dim, 3) for _ in range(self.nstack - 1)
        ])

        self.inters_ = nn.CellList([
            nn.SequentialCell(
                nn.Conv2d(curr_dim, curr_dim, kernel_size=1, has_bias=False),
                nn.BatchNorm2d(curr_dim, momentum=BN_MOMENTUM)
            ) for _ in range(self.nstack - 1)
        ])

        self.cnvs_ = nn.CellList([
            nn.SequentialCell(
                nn.Conv2d(self.cnv_dim, curr_dim, kernel_size=1, has_bias=False),
                nn.BatchNorm2d(curr_dim, momentum=BN_MOMENTUM)
            ) for _ in range(self.nstack - 1)
        ])

        self.relu = nn.ReLU()

        self.hm_fn = _generate_feature(cin=self.cnv_dim, cout=curr_dim, kernel_size=3, head=self.heads['hm'],
                                       num_stacks=self.nstack, with_bn=False)
        self.wh_fn = _generate_feature(cin=self.cnv_dim, cout=curr_dim, kernel_size=3, head=self.heads['wh'],
                                       num_stacks=self.nstack, with_bn=False)
        if net_config.reg_offset:
            self.reg_fn = _generate_feature(cin=self.cnv_dim, cout=curr_dim, kernel_size=3, head=self.heads['reg'],
                                            num_stacks=self.nstack, with_bn=False)

    def construct(self, image):
        """Defines the computation performed."""
        inter = self.pre(image)
        outs = ()
        for ind in range(self.nstack):
            kp = self.kps[ind](inter)
            cnv = self.cnvs[ind](kp)
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

            out = {}
            out['hm'] = self.hm_fn[ind](cnv)
            out['wh'] = self.wh_fn[ind](cnv)
            out['reg'] = self.reg_fn[ind](cnv)
            outs += (out,)
        return outs


class CenterNetLossCell(nn.Cell):
    """
    Provide object detection network losses.

    Args:
        net_config: The config info of CenterNet network.

    Returns:
        Tensor, total loss.
    """
    def __init__(self, net_config):
        super(CenterNetLossCell, self).__init__()
        self.network = GatherDetectionFeatureCell(net_config)
        self.num_stacks = net_config.num_stacks
        self.reduce_sum = ops.ReduceSum()
        self.Sigmoid = Sigmoid()
        self.FocalLoss = FocalLoss()
        self.crit = nn.MSELoss() if net_config.mse_loss else self.FocalLoss
        self.crit_reg = RegLoss(net_config.reg_loss)
        self.crit_wh = RegLoss(net_config.reg_loss)
        self.wh_weight = net_config.wh_weight
        self.hm_weight = net_config.hm_weight
        self.off_weight = net_config.off_weight
        self.reg_offset = net_config.reg_offset
        self.not_enable_mse_loss = not net_config.mse_loss

    def construct(self, image, hm, reg_mask, ind, wh, reg):
        """Defines the computation performed."""
        hm_loss, wh_loss, off_loss = 0, 0, 0
        feature = self.network(image)

        for s in range(self.num_stacks):
            output = feature[s]
            if self.not_enable_mse_loss:
                output_hm = self.Sigmoid(output['hm'])
            else:
                output_hm = output['hm']
            hm_loss += self.crit(output_hm, hm) / self.num_stacks

            output_wh = output['wh']
            wh_loss += self.crit_reg(output_wh, reg_mask, ind, wh) / self.num_stacks

            if self.reg_offset and self.off_weight > 0:
                output_reg = output['reg']
                off_loss += self.crit_reg(output_reg, reg_mask, ind, reg) / self.num_stacks
        total_loss = (self.hm_weight * hm_loss + self.wh_weight * wh_loss + self.off_weight * off_loss)
        return total_loss


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
        image = (image - self.mean) / self.std
        image = self.transpose(image, self.perm_list)
        return image


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
    def construct(self, image, hm, reg_mask, ind, wh, reg):
        """Defines the computation performed."""
        image = self.image(image)
        weights = self.weights
        loss = self.network(image, hm, reg_mask, ind, wh, reg)
        grads = self.grad(self.network, weights)(image, hm, reg_mask, ind, wh, reg)
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
        self.clear_before_grad = ops.NPUClearFloatStatus()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = ops.LessEqual()
        self.grad_scale = GradScale()
        self.loss_scale = sens

    @ops.add_flags(has_effect=True)
    def construct(self, image, hm, reg_mask, ind, wh, reg):
        """Defines the computation performed."""
        image = self.image(image)
        weights = self.weights
        loss = self.network(image, hm, reg_mask, ind, wh, reg)
        scaling_sens = self.cast(self.loss_scale, mstype.float32) * 2.0 / 2.0
        # alloc status and clear should be right before gradoperation
        init = self.alloc_status()
        self.clear_before_grad(init)
        grads = self.grad(self.network, weights)(image, hm, reg_mask, ind, wh, reg, scaling_sens)
        grads = self.grad_reducer(grads)
        grads = self.grad_scale(scaling_sens * self.degree, grads)
        self.get_status(init)
        flag_sum = self.reduce_sum(init, (0,))
        if self.is_distributed:
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, cond, scaling_sens)
        return ops.depend(ret, succ)


class CenterNetDetEval(nn.Cell):
    """
    Encapsulation class of centernet testing.

    Args:
        net_config: The config info of CenterNet network.
        K(number): Max number of output objects. Default: 100.
        enable_nms_fp16(bool): Use float16 data for max_pool, adaption for CPU. Default: False.

    Returns:
        Tensor, detection of images(bboxes, score, keypoints and category id of each objects)
    """
    def __init__(self, net_config, K=100, enable_nms_fp16=False):
        super(CenterNetDetEval, self).__init__()
        self.network = GatherDetectionFeatureCell(net_config)
        self.decode = DetectionDecode(net_config, K, enable_nms_fp16)

    def construct(self, image):
        """Calculate prediction scores"""
        output = self.network(image)
        features = output[-1]
        detections = self.decode(features)
        return detections
