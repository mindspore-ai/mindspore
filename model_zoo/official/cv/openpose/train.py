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
import os
import argparse

import mindspore
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.nn.optim import Adam, Momentum
from mindspore.train.loss_scale_manager import FixedLossScaleManager

from src.dataset import create_dataset
from src.openposenet import OpenPoseNet
from src.loss import openpose_loss, BuildTrainNetwork, TrainOneStepWithClipGradientCell
from src.config import params
from src.utils import get_lr, load_model, MyLossMonitor

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)

parser = argparse.ArgumentParser('mindspore openpose training')
parser.add_argument('--train_dir', type=str, default='train2017', help='train data dir')
parser.add_argument('--train_ann', type=str, default='person_keypoints_train2017.json',
                    help='train annotations json')
parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')
args, _ = parser.parse_known_args()
args.jsonpath_train = os.path.join(params['data_dir'], 'annotations/' + args.train_ann)
args.imgpath_train = os.path.join(params['data_dir'], args.train_dir)
args.maskpath_train = os.path.join(params['data_dir'], 'ignore_mask_train')


def train():
    """Train function."""

    args.outputs_dir = params['save_model_path']

    if args.group_size > 1:
        init()
        context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        args.outputs_dir = os.path.join(args.outputs_dir, "ckpt_{}/".format(str(get_rank())))
        args.rank = get_rank()
    else:
        args.outputs_dir = os.path.join(args.outputs_dir, "ckpt_0/")
        args.rank = 0

    if args.group_size > 1:
        args.max_epoch = params["max_epoch_train_NP"]
        args.loss_scale = params['loss_scale'] / 2
        args.lr_steps = list(map(int, params["lr_steps_NP"].split(',')))
        params['train_type'] = params['train_type_NP']
        params['optimizer'] = params['optimizer_NP']
        params['group_params'] = params['group_params_NP']
    else:
        args.max_epoch = params["max_epoch_train"]
        args.loss_scale = params['loss_scale']
        args.lr_steps = list(map(int, params["lr_steps"].split(',')))

    # create network
    print('start create network')
    criterion = openpose_loss()
    criterion.add_flags_recursive(fp32=True)
    network = OpenPoseNet(vggpath=params['vgg_path'], vgg_with_bn=params['vgg_with_bn'])
    if params["load_pretrain"]:
        print("load pretrain model:", params["pretrained_model_path"])
        load_model(network, params["pretrained_model_path"])
    train_net = BuildTrainNetwork(network, criterion)

    # create dataset
    if os.path.exists(args.jsonpath_train) and os.path.exists(args.imgpath_train) \
            and os.path.exists(args.maskpath_train):
        print('start create dataset')
    else:
        print('Error: wrong data path')
        return 0

    num_worker = 20 if args.group_size > 1 else 48
    de_dataset_train = create_dataset(args.jsonpath_train, args.imgpath_train, args.maskpath_train,
                                      batch_size=params['batch_size'],
                                      rank=args.rank,
                                      group_size=args.group_size,
                                      num_worker=num_worker,
                                      multiprocessing=True,
                                      shuffle=True,
                                      repeat_num=1)
    steps_per_epoch = de_dataset_train.get_dataset_size()
    print("steps_per_epoch: ", steps_per_epoch)

    # lr scheduler
    lr_stage, lr_base, lr_vgg = get_lr(params['lr'] * args.group_size,
                                       params['lr_gamma'],
                                       steps_per_epoch,
                                       args.max_epoch,
                                       args.lr_steps,
                                       args.group_size,
                                       lr_type=params['lr_type'],
                                       warmup_epoch=params['warmup_epoch'])

    # optimizer
    if params['group_params']:
        vgg19_base_params = list(filter(lambda x: 'base.vgg_base' in x.name, train_net.trainable_params()))
        base_params = list(filter(lambda x: 'base.conv' in x.name, train_net.trainable_params()))
        stages_params = list(filter(lambda x: 'base' not in x.name, train_net.trainable_params()))

        group_params = [{'params': vgg19_base_params, 'lr': lr_vgg},
                        {'params': base_params, 'lr': lr_base},
                        {'params': stages_params, 'lr': lr_stage}]

        if params['optimizer'] == "Momentum":
            opt = Momentum(group_params, learning_rate=lr_stage, momentum=0.9)
        elif params['optimizer'] == "Adam":
            opt = Adam(group_params)
        else:
            raise ValueError("optimizer not support.")
    else:
        if params['optimizer'] == "Momentum":
            opt = Momentum(train_net.trainable_params(), learning_rate=lr_stage, momentum=0.9)
        elif params['optimizer'] == "Adam":
            opt = Adam(train_net.trainable_params(), learning_rate=lr_stage)
        else:
            raise ValueError("optimizer not support.")

    # callback
    config_ck = CheckpointConfig(save_checkpoint_steps=params['ckpt_interval'],
                                 keep_checkpoint_max=params["keep_checkpoint_max"])
    ckpoint_cb = ModelCheckpoint(prefix='{}'.format(args.rank), directory=args.outputs_dir, config=config_ck)
    time_cb = TimeMonitor(data_size=de_dataset_train.get_dataset_size())
    if args.rank == 0:
        callback_list = [MyLossMonitor(), time_cb, ckpoint_cb]
    else:
        callback_list = [MyLossMonitor(), time_cb]

    # train
    if params['train_type'] == 'clip_grad':
        train_net = TrainOneStepWithClipGradientCell(train_net, opt, sens=args.loss_scale)
        train_net.set_train()
        model = Model(train_net)
    elif params['train_type'] == 'fix_loss_scale':
        loss_scale_manager = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)
        train_net.set_train()
        model = Model(train_net, optimizer=opt, loss_scale_manager=loss_scale_manager)
    else:
        raise ValueError("Type {} is not support.".format(params['train_type']))

    print("============== Starting Training ==============")
    model.train(args.max_epoch, de_dataset_train, callbacks=callback_list,
                dataset_sink_mode=False)
    return 0

if __name__ == "__main__":
    mindspore.common.seed.set_seed(1)
    train()
