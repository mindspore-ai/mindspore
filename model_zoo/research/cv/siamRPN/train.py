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
""" train """
import os
import random
import time
import sys
import argparse
from mindspore import context
from mindspore.train.callback import Callback, ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.model import Model
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import numpy as np
from src.data_loader import TrainDataLoader
from src.net import SiameseRPN, BuildTrainNet, MyTrainOneStepCell
from src.config import config
from src.loss import MultiBoxLoss

sys.path.append('../')

parser = argparse.ArgumentParser(description='Mindspore SiameseRPN Training')

parser.add_argument('--is_parallel', default=False, type=bool, help='whether parallel or not parallel')

parser.add_argument('--is_cloudtrain', default=False, type=bool, help='whether cloud or not')

parser.add_argument('--train_url', default=None, help='Location of training outputs.')

parser.add_argument('--data_url', default=None, help='Location of data.')

parser.add_argument('--unzip_mode', default=0, type=int, metavar='N', help='unzip mode:0:no unzip,1:tar,2:unzip')

parser.add_argument('--device_id', default=2, type=int, metavar='N', help='number of total epochs to run')


#add random seed
random.seed(1)
np.random.seed(1)
ds.config.set_seed(1)


def main(args):
    """ Model"""
    net = SiameseRPN(groups=config.batch_size, is_train=True)
    criterion = MultiBoxLoss(config.batch_size)

    if config.check:
        checkpoint_path = os.path.join(config.checkpoint_path, config.pretrain_model)
        print("Load checkpoint Done ")
    print(config.checkpoint_path)
    if not checkpoint_path is None:
        param_dict = load_checkpoint(checkpoint_path)
        load_param_into_net(net, param_dict)
        cur_epoch = config.cur_epoch
        total_epoch = config.max_epoches - cur_epoch
    # dataloader
    data_loader = TrainDataLoader(config.train_path)
    if args.is_parallel:
        # get rank_id and rank_size
        rank_id = get_rank()
        rank_size = int(os.getenv('RANK_SIZE'))
        # create dataset
        dataset = ds.GeneratorDataset(data_loader, ["template", "detection", "label"], shuffle=True,
                                      num_parallel_workers=rank_size, num_shards=rank_size, shard_id=rank_id)
    else:
        dataset = ds.GeneratorDataset(data_loader, ["template", "detection", "label"], shuffle=True)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)

    # set training
    net.set_train()


    conv_params = list(filter(lambda layer: 'featureExtract.0.' not in layer.name and 'featureExtract.1.'
                              not in layer.name and 'featureExtract.4.' not in layer.name and 'featureExtract.5.'
                              not in layer.name and 'featureExtract.8.' not in layer.name and 'featureExtract.9.'
                              not in layer.name, net.trainable_params()))

    lr = adjust_learning_rate(config.start_lr, config.end_lr, config.max_epoches, dataset.get_dataset_size())
    #start fixed epoch,fix layers,select optimizer
    del lr[0:dataset.get_dataset_size() * cur_epoch]

    optimizer = nn.optim.SGD(learning_rate=lr, params=conv_params, momentum=config.momentum,
                             weight_decay=config.weight_decay)

    train_net = BuildTrainNet(net, criterion)
    train_network = MyTrainOneStepCell(train_net, optimizer)

    #define Model
    model = Model(train_network)
    loss_cb = LossMonitor()


    class Print_info(Callback):
        """ print callback function """
        def epoch_begin(self, run_context):
            self.epoch_time = time.time()
            self.tlosses = AverageMeter()

        def epoch_end(self, run_context):
            cb_params = run_context.original_args()
            epoch_seconds = (time.time() - self.epoch_time) * 1000
            print("epoch time: %s, per step time: %s"%(epoch_seconds, epoch_seconds/cb_params.batch_num))
        def step_begin(self, run_context):
            self.step_time = time.time()

        def step_end(self, run_context):
            step_mseconds = (time.time() - self.step_time) * 1000
            cb_params = run_context.original_args()
            loss = cb_params.net_outputs
            self.tlosses.update(loss)
            print("epoch: %s step: %s, loss is %s, "
                  "avg_loss is %s, step time is %s" % (cb_params.cur_epoch_num, cb_params.cur_step_num, loss,
                                                       self.tlosses.avg, step_mseconds), flush=True)
    print_cb = Print_info()
    #save checkpoint
    ckpt_cfg = CheckpointConfig(save_checkpoint_steps=dataset.get_dataset_size(), keep_checkpoint_max=51)
    if args.is_cloudtrain:
        ckpt_cb = ModelCheckpoint(prefix='siamRPN', directory=config.train_path+'/ckpt', config=ckpt_cfg)
    else:
        ckpt_cb = ModelCheckpoint(prefix='siamRPN', directory='./ckpt', config=ckpt_cfg)

    if config.checkpoint_path is not None and os.path.exists(config.checkpoint_path):
        model.train(total_epoch, dataset, callbacks=[loss_cb, ckpt_cb, print_cb], dataset_sink_mode=False)
    else:
        model.train(epoch=total_epoch, train_dataset=dataset, callbacks=[loss_cb, ckpt_cb, print_cb],
                    dataset_sink_mode=False)


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

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



def adjust_learning_rate(start_lr, end_lr, total_epochs, steps_pre_epoch):
    """ adjust lr """
    lr = np.logspace(np.log10(start_lr), np.log10(end_lr), num=total_epochs)[0]
    gamma = np.logspace(np.log10(start_lr), np.log10(end_lr), num=total_epochs)[1] / \
            np.logspace(np.log10(start_lr), np.log10(end_lr), num=total_epochs)[0]
    lr_each_step = []
    for _ in range(steps_pre_epoch):
        lr_each_step.append(lr)
    for _ in range(2, total_epochs + 1):
        lr = lr * gamma
        for _ in range(steps_pre_epoch):
            lr_each_step.append(lr)
    return lr_each_step

if __name__ == '__main__':
    Args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    if not Args.is_parallel:
        device_id = Args.device_id
        context.set_context(device_id=device_id, mode=context.GRAPH_MODE, device_target="Ascend")
    if Args.is_cloudtrain:
        import moxing as mox
        device_id = int(os.getenv('DEVICE_ID') if os.getenv('DEVICE_ID') is not None else 0)
        local_data_path = config.cloud_data_path
        # adapt to cloud: define distributed local data path
        local_data_path = os.path.join(local_data_path, str(device_id))
        # adapt to cloud: download data from obs to local location
        mox.file.copy_parallel(src_url=Args.data_url, dst_url=local_data_path)
        tar_command1 = "tar -zxf " + local_data_path + "/ytb_vid_filter.tar.gz -C " + local_data_path + '/train/'
        zip_command1 = "unzip -o -q " + local_data_path + "/train.zip -d " + local_data_path + '/train/'
        config.checkpoint_path = local_data_path
        if Args.unzip_mode == 2:
            os.system(zip_command1)
            local_data_path = local_data_path + '/train'
        elif Args.unzip_mode == 1:
            os.system("mkdir " + local_data_path + '/train')
            os.system(tar_command1)
            local_data_path = local_data_path + '/train/ytb_vid_filter'
        config.train_path = local_data_path
    elif Args.is_parallel:
        config.train_path = os.getenv('DATA_PATH')
    if Args.is_parallel:
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        if device_num > 1:
            context.set_context(device_id=device_id, mode=context.GRAPH_MODE, device_target="Ascend")
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              parameter_broadcast=True, gradients_mean=True)
            init()
    main(Args)
    if Args.is_cloudtrain:
        mox.file.copy_parallel(src_url=local_data_path + '/ckpt', dst_url=Args.train_url + '/ckpt')
