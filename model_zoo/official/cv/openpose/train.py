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

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.nn.optim import Adam

from src.dataset import create_dataset
from src.openposenet import OpenPoseNet
from src.loss import openpose_loss, BuildTrainNetwork
from src.config import params
from src.utils import parse_args, get_lr, load_model, MyLossMonitor

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)

def train():
    """Train function."""
    args = parse_args()

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

    # with out loss_scale
    if args.group_size > 1:
        args.loss_scale = params['loss_scale'] / 2
        args.lr_steps = list(map(int, params["lr_steps_NP"].split(',')))
    else:
        args.loss_scale = params['loss_scale']
        args.lr_steps = list(map(int, params["lr_steps"].split(',')))

    # create network
    print('start create network')
    criterion = openpose_loss()
    criterion.add_flags_recursive(fp32=True)
    network = OpenPoseNet(vggpath=params['vgg_path'])
    # network.add_flags_recursive(fp32=True)

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
                                       params["max_epoch_train"],
                                       args.lr_steps,
                                       args.group_size)
    vgg19_base_params = list(filter(lambda x: 'base.vgg_base' in x.name, train_net.trainable_params()))
    base_params = list(filter(lambda x: 'base.conv' in x.name, train_net.trainable_params()))
    stages_params = list(filter(lambda x: 'base' not in x.name, train_net.trainable_params()))

    group_params = [{'params': vgg19_base_params, 'lr': lr_vgg},
                    {'params': base_params, 'lr': lr_base},
                    {'params': stages_params, 'lr': lr_stage}]

    opt = Adam(group_params, loss_scale=args.loss_scale)

    train_net.set_train(True)
    loss_scale_manager = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)

    model = Model(train_net, optimizer=opt, loss_scale_manager=loss_scale_manager)

    params['ckpt_interval'] = max(steps_per_epoch, params['ckpt_interval'])
    config_ck = CheckpointConfig(save_checkpoint_steps=params['ckpt_interval'],
                                 keep_checkpoint_max=params["keep_checkpoint_max"])
    ckpoint_cb = ModelCheckpoint(prefix='{}'.format(args.rank), directory=args.outputs_dir, config=config_ck)
    time_cb = TimeMonitor(data_size=de_dataset_train.get_dataset_size())
    callback_list = [MyLossMonitor(), time_cb, ckpoint_cb]
    print("============== Starting Training ==============")
    model.train(params["max_epoch_train"], de_dataset_train, callbacks=callback_list,
                dataset_sink_mode=False)


if __name__ == "__main__":
    # mindspore.common.seed.set_seed(1)
    train()
