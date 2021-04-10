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
"""auxiliary functions for train, to print and preload"""

import math
import logging
import os
import sys
from datetime import datetime
import numpy as np

from mindspore.train.serialization import load_checkpoint
import mindspore.nn as nn

def load_backbone(net, ckpt_path, args):
    """
    Load backbone
    """
    param_dict = load_checkpoint(ckpt_path)
    centerface_backbone_prefix = 'base'
    mobilev2_backbone_prefix = 'network.backbone'
    find_param = []
    not_found_param = []

    def replace_names(name, replace_name, replace_idx):
        names = name.split('.')
        if len(names) < 4:
            raise "centerface_backbone_prefix name too short"
        tmp = names[2] + '.' + names[3]
        if replace_name != tmp:
            replace_name = tmp
            replace_idx += 1
        name = name.replace(replace_name, 'features' + '.' + str(replace_idx))
        return name, replace_name, replace_idx

    replace_name = 'need_fp1.0'
    replace_idx = 0
    for name, cell in net.cells_and_names():
        if name.startswith(centerface_backbone_prefix):
            name = name.replace(centerface_backbone_prefix, mobilev2_backbone_prefix)
            if isinstance(cell, (nn.Conv2d, nn.Dense)):
                name, replace_name, replace_idx = replace_names(name, replace_name, replace_idx)
                mobilev2_weight = '{}.weight'.format(name)
                mobilev2_bias = '{}.bias'.format(name)
                if mobilev2_weight in param_dict:
                    cell.weight.set_data(param_dict[mobilev2_weight].data)
                    find_param.append(mobilev2_weight)
                else:
                    not_found_param.append(mobilev2_weight)
                if mobilev2_bias in param_dict:
                    cell.bias.set_data(param_dict[mobilev2_bias].data)
                    find_param.append(mobilev2_bias)
                else:
                    not_found_param.append(mobilev2_bias)
            elif isinstance(cell, (nn.BatchNorm2d, nn.BatchNorm1d)):
                name, replace_name, replace_idx = replace_names(name, replace_name, replace_idx)
                mobilev2_moving_mean = '{}.moving_mean'.format(name)
                mobilev2_moving_variance = '{}.moving_variance'.format(name)
                mobilev2_gamma = '{}.gamma'.format(name)
                mobilev2_beta = '{}.beta'.format(name)
                if mobilev2_moving_mean in param_dict:
                    cell.moving_mean.set_data(param_dict[mobilev2_moving_mean].data)
                    find_param.append(mobilev2_moving_mean)
                else:
                    not_found_param.append(mobilev2_moving_mean)
                if mobilev2_moving_variance in param_dict:
                    cell.moving_variance.set_data(param_dict[mobilev2_moving_variance].data)
                    find_param.append(mobilev2_moving_variance)
                else:
                    not_found_param.append(mobilev2_moving_variance)
                if mobilev2_gamma in param_dict:
                    cell.gamma.set_data(param_dict[mobilev2_gamma].data)
                    find_param.append(mobilev2_gamma)
                else:
                    not_found_param.append(mobilev2_gamma)
                if mobilev2_beta in param_dict:
                    cell.beta.set_data(param_dict[mobilev2_beta].data)
                    find_param.append(mobilev2_beta)
                else:
                    not_found_param.append(mobilev2_beta)

    args.logger.info('================found_param {}========='.format(len(find_param)))
    args.logger.info(find_param)
    args.logger.info('================not_found_param {}========='.format(len(not_found_param)))
    args.logger.info(not_found_param)
    args.logger.info('=====load {} successfully ====='.format(ckpt_path))

    return net

def get_param_groups(network):
    """
    Get param groups
    """
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


class DistributedSampler():
    """
    Distributed sampler
    """
    def __init__(self, dataset, rank, group_size, shuffle=True, seed=0):
        self.dataset = dataset
        self.rank = rank
        self.group_size = group_size
        self.dataset_length = len(self.dataset)
        self.num_samples = int(math.ceil(self.dataset_length * 1.0 / self.group_size))
        self.total_size = self.num_samples * self.group_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xffffffff
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_length).tolist()
        else:
            indices = list(range(len(self.dataset.classes)))

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank::self.group_size]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', tb_writer=None):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.tb_writer = tb_writer
        self.cur_step = 1

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

class LOGGER(logging.Logger):
    """
    Logger class
    """
    def __init__(self, logger_name, rank=0):
        super(LOGGER, self).__init__(logger_name)
        if rank % 8 == 0:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
            console.setFormatter(formatter)
            self.addHandler(console)

    def setup_logging_file(self, log_dir, rank=0):
        """
        Setup logging file
        """
        self.rank = rank
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_name = datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S') + '_rank_{}.log'.format(rank)
        self.log_fn = os.path.join(log_dir, log_name)
        fh = logging.FileHandler(self.log_fn)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        fh.setFormatter(formatter)
        self.addHandler(fh)

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)

    def save_args(self, args):
        self.info('Args:')
        args_dict = vars(args)
        for key in args_dict.keys():
            self.info('--> %s', key)
        self.info('')

    def important_info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO) and self.rank == 0:
            line_width = 2
            important_msg = '\n'
            important_msg += ('*'*70 + '\n')*line_width
            important_msg += ('*'*line_width + '\n')*2
            important_msg += '*'*line_width + ' '*8 + msg + '\n'
            important_msg += ('*'*line_width + '\n')*2
            important_msg += ('*'*70 + '\n')*line_width
            self.info(important_msg, *args, **kwargs)

def get_logger(path, rank):
    logger = LOGGER("centerface", rank)
    logger.setup_logging_file(path, rank)
    return logger
