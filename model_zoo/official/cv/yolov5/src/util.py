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
"""Util class or function."""
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


def default_wd_filter(x):
    """default weight decay filter."""
    parameter_name = x.name
    if parameter_name.endswith('.bias'):
        # all bias not using weight decay
        return False
    if parameter_name.endswith('.gamma'):
        # bn weight bias not using weight decay, be carefully for now x not
        # include BN
        return False
    if parameter_name.endswith('.beta'):
        # bn weight bias not using weight decay, be carefully for now x not
        # include BN
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
            # bn weight bias not using weight decay, be carefully for now x not
            # include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not
            # include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0},
            {'params': decay_params}]


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
            logger.info('shape {}: {:.2f}%'.format(key, rate * 100))


def keep_loss_fp32(network):
    """Keep loss of network with float32"""
    for _, cell in network.cells_and_names():
        if isinstance(cell, (YoloLossBlock,)):
            cell.to_float(mstype.float32)
