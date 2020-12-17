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
"""Face Recognition train."""
import os
import argparse

import mindspore
from mindspore.nn import Cell
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size, init, get_rank
from mindspore.nn.optim import Momentum
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import config_base, config_beta
from src.my_logging import get_logger
from src.init_network import init_net
from src.dataset_factory import get_de_dataset
from src.backbone.resnet import get_backbone
from src.metric_factory import get_metric_fc
from src.loss_factory import get_loss
from src.lrsche_factory import warmup_step_list, list_to_gen
from src.callback_factory import ProgressMonitor

mindspore.common.seed.set_seed(1)
devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False,
                    device_id=devid, reserve_class_name_in_scope=False, enable_auto_mixed_precision=False)

class DistributedHelper(Cell):
    '''DistributedHelper'''
    def __init__(self, backbone, margin_fc):
        super(DistributedHelper, self).__init__()
        self.backbone = backbone
        self.margin_fc = margin_fc
        if margin_fc is not None:
            self.has_margin_fc = 1
        else:
            self.has_margin_fc = 0

    def construct(self, x, label):
        embeddings = self.backbone(x)
        if self.has_margin_fc == 1:
            return embeddings, self.margin_fc(embeddings, label)
        return embeddings


class BuildTrainNetwork(Cell):
    '''BuildTrainNetwork'''
    def __init__(self, network, criterion, args_1):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion
        self.args = args_1

        if int(args_1.model_parallel) == 0:
            self.is_model_parallel = 0
        else:
            self.is_model_parallel = 1

    def construct(self, input_data, label):

        if self.is_model_parallel == 0:
            _, output = self.network(input_data, label)
            loss = self.criterion(output, label)
        else:
            _ = self.network(input_data, label)
            loss = self.criterion(None, label)

        return loss

def parse_args():
    parser = argparse.ArgumentParser('MindSpore Face Recognition')
    parser.add_argument('--train_stage', type=str, default='base', help='train stage, base or beta')
    parser.add_argument('--is_distributed', type=int, default=1, help='if multi device')

    args_opt_1, _ = parser.parse_known_args()
    return args_opt_1

if __name__ == "__main__":
    args_opt = parse_args()

    support_train_stage = ['base', 'beta']
    if args_opt.train_stage.lower() not in support_train_stage:
        args.logger.info('support train stage is:{}, while yours is:{}'.
                         format(support_train_stage, args_opt.train_stage))
        raise ValueError('train stage not support.')
    args = config_base if args_opt.train_stage.lower() == 'base' else config_beta
    args.is_distributed = args_opt.is_distributed
    if args_opt.is_distributed:
        init()
        args.local_rank = get_rank()
        args.world_size = get_group_size()
        parallel_mode = ParallelMode.HYBRID_PARALLEL
    else:
        parallel_mode = ParallelMode.STAND_ALONE

    context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                      device_num=args.world_size, gradients_mean=True)

    if not os.path.exists(args.data_dir):
        args.logger.info('ERROR, data_dir is not exists, please set data_dir in config.py')
        raise ValueError('ERROR, data_dir is not exists, please set data_dir in config.py')

    args.lr_epochs = list(map(int, args.lr_epochs.split(',')))


    log_path = os.path.join(args.ckpt_path, 'logs')
    args.logger = get_logger(log_path, args.local_rank)

    if args.local_rank % 8 == 0:
        if not os.path.exists(args.ckpt_path):
            os.makedirs(args.ckpt_path)

    args.logger.info('args.world_size:{}'.format(args.world_size))
    args.logger.info('args.local_rank:{}'.format(args.local_rank))
    args.logger.info('args.lr:{}'.format(args.lr))

    momentum = args.momentum
    weight_decay = args.weight_decay

    de_dataset, steps_per_epoch, num_classes = get_de_dataset(args)
    args.logger.info('de_dataset:{}'.format(de_dataset.get_dataset_size()))
    args.steps_per_epoch = steps_per_epoch
    args.num_classes = num_classes

    args.logger.info('loaded, nums: {}'.format(args.num_classes))
    if args.nc_16 == 1:
        if args.model_parallel == 0:
            if args.num_classes % 16 == 0:
                args.logger.info('data parallel aleardy 16, nums: {}'.format(args.num_classes))
            else:
                args.num_classes = (args.num_classes // 16 + 1) * 16
        else:
            if args.num_classes % (args.world_size * 16) == 0:
                args.logger.info('model parallel aleardy 16, nums: {}'.format(args.num_classes))
            else:
                args.num_classes = (args.num_classes // (args.world_size * 16) + 1) * args.world_size * 16

    args.logger.info('for D, loaded, class nums: {}'.format(args.num_classes))
    args.logger.info('steps_per_epoch:{}'.format(args.steps_per_epoch))
    args.logger.info('img_total_num:{}'.format(args.steps_per_epoch * args.per_batch_size))

    args.logger.info('get_backbone----in----')
    _backbone = get_backbone(args)
    args.logger.info('get_backbone----out----')

    args.logger.info('get_metric_fc----in----')
    margin_fc_1 = get_metric_fc(args)
    args.logger.info('get_metric_fc----out----')

    args.logger.info('DistributedHelper----in----')
    network_1 = DistributedHelper(_backbone, margin_fc_1)
    args.logger.info('DistributedHelper----out----')

    args.logger.info('network fp16----in----')
    if args.fp16 == 1:
        network_1.add_flags_recursive(fp16=True)
    args.logger.info('network fp16----out----')

    criterion_1 = get_loss(args)
    if args.fp16 == 1 and args.model_parallel == 0:
        criterion_1.add_flags_recursive(fp32=True)

    if os.path.isfile(args.pretrained):
        param_dict = load_checkpoint(args.pretrained)
        param_dict_new = {}
        if args_opt.train_stage.lower() == 'base':
            for key, value in param_dict.items():
                if key.startswith('moments.'):
                    continue
                elif key.startswith('network.'):
                    param_dict_new[key[8:]] = value
        else:
            for key, value in param_dict.items():
                if key.startswith('moments.'):
                    continue
                elif key.startswith('network.'):
                    if 'layers.' in key and 'bn1' in key:
                        continue
                    elif 'se' in key:
                        continue
                    elif 'head' in key:
                        continue
                    elif 'margin_fc.weight' in key:
                        continue
                    else:
                        param_dict_new[key[8:]] = value
        load_param_into_net(network_1, param_dict_new)
        args.logger.info('load model {} success'.format(args.pretrained))
    else:
        init_net(args, network_1)

    train_net = BuildTrainNetwork(network_1, criterion_1, args)

    args.logger.info('args:{}'.format(args))
    # call warmup_step should behind the args steps_per_epoch
    args.lrs = warmup_step_list(args, gamma=0.1)
    lrs_gen = list_to_gen(args.lrs)
    opt = Momentum(params=train_net.trainable_params(), learning_rate=lrs_gen, momentum=momentum,
                   weight_decay=weight_decay)
    scale_manager = DynamicLossScaleManager(init_loss_scale=args.dynamic_init_loss_scale, scale_factor=2,
                                            scale_window=2000)
    model = Model(train_net, optimizer=opt, metrics=None, loss_scale_manager=scale_manager)
    save_checkpoint_steps = args.ckpt_steps
    args.logger.info('save_checkpoint_steps:{}'.format(save_checkpoint_steps))
    if args.max_ckpts == -1:
        keep_checkpoint_max = int(args.steps_per_epoch * args.max_epoch / save_checkpoint_steps) + 5 # for more than 5
    else:
        keep_checkpoint_max = args.max_ckpts
    args.logger.info('keep_checkpoint_max:{}'.format(keep_checkpoint_max))

    ckpt_config = CheckpointConfig(save_checkpoint_steps=save_checkpoint_steps, keep_checkpoint_max=keep_checkpoint_max)
    max_epoch_train = args.max_epoch
    args.logger.info('max_epoch_train:{}'.format(max_epoch_train))
    ckpt_cb = ModelCheckpoint(config=ckpt_config, directory=args.ckpt_path, prefix='{}'.format(args.local_rank))
    args.epoch_cnt = 0
    progress_cb = ProgressMonitor(args)
    new_epoch_train = max_epoch_train * steps_per_epoch // args.log_interval
    model.train(new_epoch_train, de_dataset, callbacks=[progress_cb, ckpt_cb], sink_size=args.log_interval)
