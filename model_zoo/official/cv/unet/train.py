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

import os
import argparse
import logging
import ast

import mindspore
import mindspore.nn as nn
from mindspore import Model, context
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.unet_medical import UNetMedical
from src.unet_nested import NestedUNet, UNet
from src.data_loader import create_dataset, create_cell_nuclei_dataset
from src.loss import CrossEntropyWithLogits, MultiCrossEntropyWithLogits
from src.utils import StepLossTimeMonitor, UnetEval, TempLoss, apply_eval, filter_checkpoint_parameter_by_list, dice_coeff
from src.config import cfg_unet
from src.eval_callback import EvalCallBack

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=device_id)

mindspore.set_seed(1)

def train_net(args_opt,
              cross_valid_ind=1,
              epochs=400,
              batch_size=16,
              lr=0.0001,
              cfg=None):
    rank = 0
    group_size = 1
    data_dir = args_opt.data_url
    run_distribute = args_opt.run_distribute
    if run_distribute:
        init()
        group_size = get_group_size()
        rank = get_rank()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          device_num=group_size,
                                          gradients_mean=False)
    need_slice = False
    if cfg['model'] == 'unet_medical':
        net = UNetMedical(n_channels=cfg['num_channels'], n_classes=cfg['num_classes'])
    elif cfg['model'] == 'unet_nested':
        net = NestedUNet(in_channel=cfg['num_channels'], n_class=cfg['num_classes'], use_deconv=cfg['use_deconv'],
                         use_bn=cfg['use_bn'], use_ds=cfg['use_ds'])
        need_slice = cfg['use_ds']
    elif cfg['model'] == 'unet_simple':
        net = UNet(in_channel=cfg['num_channels'], n_class=cfg['num_classes'])
    else:
        raise ValueError("Unsupported model: {}".format(cfg['model']))

    if cfg['resume']:
        param_dict = load_checkpoint(cfg['resume_ckpt'])
        if cfg['transfer_training']:
            filter_checkpoint_parameter_by_list(param_dict, cfg['filter_weight'])
        load_param_into_net(net, param_dict)

    if 'use_ds' in cfg and cfg['use_ds']:
        criterion = MultiCrossEntropyWithLogits()
    else:
        criterion = CrossEntropyWithLogits()
    if 'dataset' in cfg and cfg['dataset'] == "Cell_nuclei":
        repeat = cfg['repeat']
        dataset_sink_mode = True
        per_print_times = 0
        train_dataset = create_cell_nuclei_dataset(data_dir, cfg['img_size'], repeat, batch_size,
                                                   is_train=True, augment=True, split=0.8, rank=rank,
                                                   group_size=group_size)
        valid_dataset = create_cell_nuclei_dataset(data_dir, cfg['img_size'], 1, 1, is_train=False,
                                                   eval_resize=cfg["eval_resize"], split=0.8,
                                                   python_multiprocessing=False)
    else:
        repeat = cfg['repeat']
        dataset_sink_mode = False
        per_print_times = 1
        train_dataset, valid_dataset = create_dataset(data_dir, repeat, batch_size, True, cross_valid_ind,
                                                      run_distribute, cfg["crop"], cfg['img_size'])
    train_data_size = train_dataset.get_dataset_size()
    print("dataset length is:", train_data_size)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=train_data_size,
                                   keep_checkpoint_max=cfg['keep_checkpoint_max'])
    ckpoint_cb = ModelCheckpoint(prefix='ckpt_{}_adam'.format(cfg['model']),
                                 directory='./ckpt_{}/'.format(device_id),
                                 config=ckpt_config)

    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=lr, weight_decay=cfg['weight_decay'],
                        loss_scale=cfg['loss_scale'])

    loss_scale_manager = mindspore.train.loss_scale_manager.FixedLossScaleManager(cfg['FixedLossScaleManager'], False)

    model = Model(net, loss_fn=criterion, loss_scale_manager=loss_scale_manager, optimizer=optimizer, amp_level="O3")

    print("============== Starting Training ==============")
    callbacks = [StepLossTimeMonitor(batch_size=batch_size, per_print_times=per_print_times), ckpoint_cb]
    if args_opt.run_eval:
        eval_model = Model(UnetEval(net, need_slice=need_slice), loss_fn=TempLoss(),
                           metrics={"dice_coeff": dice_coeff(cfg_unet, False)})
        eval_param_dict = {"model": eval_model, "dataset": valid_dataset, "metrics_name": args_opt.eval_metrics}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=args_opt.eval_interval,
                               eval_start_epoch=args_opt.eval_start_epoch, save_best_ckpt=True,
                               ckpt_directory='./ckpt_{}/'.format(device_id), besk_ckpt_name="best.ckpt",
                               metrics_name=args_opt.eval_metrics)
        callbacks.append(eval_cb)
    model.train(int(epochs / repeat), train_dataset, callbacks=callbacks, dataset_sink_mode=dataset_sink_mode)
    print("============== End Training ==============")


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_url', dest='data_url', type=str, default='data/',
                        help='data directory')
    parser.add_argument('-t', '--run_distribute', type=ast.literal_eval,
                        default=False, help='Run distribute, default: false.')
    parser.add_argument("--run_eval", type=ast.literal_eval, default=False,
                        help="Run evaluation when training, default is False.")
    parser.add_argument("--save_best_ckpt", type=ast.literal_eval, default=True,
                        help="Save best checkpoint when run_eval is True, default is True.")
    parser.add_argument("--eval_start_epoch", type=int, default=0,
                        help="Evaluation start epoch when run_eval is True, default is 0.")
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="Evaluation interval when run_eval is True, default is 1.")
    parser.add_argument("--eval_metrics", type=str, default="dice_coeff", choices=("dice_coeff", "iou"),
                        help="Evaluation metrics when run_eval is True, support [dice_coeff, iou], "
                             "default is dice_coeff.")

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    print("Training setting:", args)

    epoch_size = cfg_unet['epochs'] if not args.run_distribute else cfg_unet['distribute_epochs']
    train_net(args_opt=args,
              cross_valid_ind=cfg_unet['cross_valid_ind'],
              epochs=epoch_size,
              batch_size=cfg_unet['batchsize'],
              lr=cfg_unet['lr'],
              cfg=cfg_unet)
