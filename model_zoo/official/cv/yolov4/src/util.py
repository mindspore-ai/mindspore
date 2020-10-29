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
"""Util class or function."""
from mindspore.train.serialization import load_checkpoint
import mindspore.nn as nn
import mindspore.common.dtype as mstype

from .yolo import YoloLossBlock


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', tb_writer=None):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.tb_writer = tb_writer
        self.cur_step = 1
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(self.name, self.val, self.cur_step)
        self.cur_step += 1

    def __str__(self):
        fmtstr = '{name}:{avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def load_backbone(net, ckpt_path, args):
    """Load cspdarknet53 backbone checkpoint."""
    param_dict = load_checkpoint(ckpt_path)
    yolo_backbone_prefix = 'feature_map.backbone'
    darknet_backbone_prefix = 'backbone'
    find_param = []
    not_found_param = []
    net.init_parameters_data()
    for name, cell in net.cells_and_names():
        if name.startswith(yolo_backbone_prefix):
            name = name.replace(yolo_backbone_prefix, darknet_backbone_prefix)
            if isinstance(cell, (nn.Conv2d, nn.Dense)):
                darknet_weight = '{}.weight'.format(name)
                darknet_bias = '{}.bias'.format(name)
                if darknet_weight in param_dict:
                    cell.weight.set_data(param_dict[darknet_weight].data)
                    find_param.append(darknet_weight)
                else:
                    not_found_param.append(darknet_weight)
                if darknet_bias in param_dict:
                    cell.bias.set_data(param_dict[darknet_bias].data)
                    find_param.append(darknet_bias)
                else:
                    not_found_param.append(darknet_bias)
            elif isinstance(cell, (nn.BatchNorm2d, nn.BatchNorm1d)):
                darknet_moving_mean = '{}.moving_mean'.format(name)
                darknet_moving_variance = '{}.moving_variance'.format(name)
                darknet_gamma = '{}.gamma'.format(name)
                darknet_beta = '{}.beta'.format(name)
                if darknet_moving_mean in param_dict:
                    cell.moving_mean.set_data(param_dict[darknet_moving_mean].data)
                    find_param.append(darknet_moving_mean)
                else:
                    not_found_param.append(darknet_moving_mean)
                if darknet_moving_variance in param_dict:
                    cell.moving_variance.set_data(param_dict[darknet_moving_variance].data)
                    find_param.append(darknet_moving_variance)
                else:
                    not_found_param.append(darknet_moving_variance)
                if darknet_gamma in param_dict:
                    cell.gamma.set_data(param_dict[darknet_gamma].data)
                    find_param.append(darknet_gamma)
                else:
                    not_found_param.append(darknet_gamma)
                if darknet_beta in param_dict:
                    cell.beta.set_data(param_dict[darknet_beta].data)
                    find_param.append(darknet_beta)
                else:
                    not_found_param.append(darknet_beta)

    args.logger.info('================found_param {}========='.format(len(find_param)))
    args.logger.info(find_param)
    args.logger.info('================not_found_param {}========='.format(len(not_found_param)))
    args.logger.info(not_found_param)
    args.logger.info('=====load {} successfully ====='.format(ckpt_path))

    return net


def default_wd_filter(x):
    """default weight decay filter."""
    parameter_name = x.name
    if parameter_name.endswith('.bias'):
        # all bias not using weight decay
        return False
    if parameter_name.endswith('.gamma'):
        # bn weight bias not using weight decay, be carefully for now x not include BN
        return False
    if parameter_name.endswith('.beta'):
        # bn weight bias not using weight decay, be carefully for now x not include BN
        return False

    return True


def get_param_groups(network):
    """Param groups for optimizer."""
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]


class ShapeRecord:
    """Log image shape."""
    def __init__(self):
        self.shape_record = {
            416: 0,
            448: 0,
            480: 0,
            512: 0,
            544: 0,
            576: 0,
            608: 0,
            640: 0,
            672: 0,
            704: 0,
            736: 0,
            'total': 0
        }

    def set(self, shape):
        if len(shape) > 1:
            shape = shape[0]
        shape = int(shape)
        self.shape_record[shape] += 1
        self.shape_record['total'] += 1

    def show(self, logger):
        for key in self.shape_record:
            rate = self.shape_record[key] / float(self.shape_record['total'])
            logger.info('shape {}: {:.2f}%'.format(key, rate*100))


def keep_loss_fp32(network):
    """Keep loss of network with float32"""
    for _, cell in network.cells_and_names():
        if isinstance(cell, (YoloLossBlock,)):
            cell.to_float(mstype.float32)
