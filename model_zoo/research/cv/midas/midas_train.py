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
"""train midas."""
import os
import json
from mindspore import dtype as mstype
from mindspore import context
from mindspore import nn
from mindspore import Tensor
from mindspore.context import ParallelMode
import mindspore.dataset as ds
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.communication.management import init
from src.midas_net import MidasNet, Loss, NetwithCell
from src.utils import loadImgDepth
from src.config import config

set_seed(1)
ds.config.set_seed(1)

def dynamic_lr(num_epoch_per_decay, total_epochs, steps_per_epoch, lr, end_lr):
    """dynamic learning rate generator"""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    decay_steps = steps_per_epoch * num_epoch_per_decay
    lr = nn.PolynomialDecayLR(lr, end_lr, decay_steps, 0.5)
    for i in range(total_steps):
        if i < decay_steps:
            i = Tensor(i, mstype.int32)
            lr_each_step.append(lr(i).asnumpy())
        else:
            lr_each_step.append(end_lr)
    return lr_each_step


def train(mixdata_path):
    """train"""
    epoch_number_total = config.epoch_size
    batch_size = config.batch_size
    if config.is_modelarts:
        import moxing as mox
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        local_data_path = '/cache/data'
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, max_call_depth=10000)
        context.set_context(device_id=device_id)
        # define distributed local data path
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
        local_data_path = os.path.join(local_data_path, str(device_id))
        mixdata_path = os.path.join(local_data_path, mixdata_path)
        load_path = os.path.join(local_data_path, 'midas_resnext_101_WSL.ckpt')
        output_path = config.train_url
        print('local_data_path:', local_data_path)
        print('mixdata_path:', mixdata_path)
        print('load_path:', load_path)
        print('output_path:', output_path)
        # data download
        print('Download data.')
        mox.file.copy_parallel(src_url=config.data_url, dst_url=local_data_path)
    elif config.run_distribute:
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(device_id=device_id, mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False,
                            max_call_depth=10000)
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True,
                                          device_num=device_num
                                          )
        init()
        local_data_path = config.train_data_dir + '/data'
        mixdata_path = config.train_data_dir + '/data/mixdata.json'
        load_path = config.train_data_dir + '/midas/ckpt/midas_resnext_101_WSL.ckpt'
    else:
        local_data_path = config.train_data_dir + '/data'
        mixdata_path = config.train_data_dir + '/data/mixdata.json'
        load_path = config.train_data_dir + '/midas/ckpt/midas_resnext_101_WSL.ckpt'
        device_id = config.device_id
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=device_id,
                            max_call_depth=10000)
    # load data
    f = open(mixdata_path)
    data_config = json.load(f)
    img_paths = data_config['img']
    # depth_paths = data_config['depth']
    f.close()
    mix_dataset = loadImgDepth.LoadImagesDepth(local_path=local_data_path, img_paths=img_paths)
    if config.is_modelarts or config.run_distribute:
        mix_dataset = ds.GeneratorDataset(mix_dataset, ['img', 'mask', 'depth'], shuffle=True, num_parallel_workers=8,
                                          num_shards=device_num, shard_id=device_id)
    else:
        mix_dataset = ds.GeneratorDataset(mix_dataset, ['img', 'mask', 'depth'], shuffle=True)
    mix_dataset = mix_dataset.batch(8, drop_remainder=True)
    per_step_size = mix_dataset.get_dataset_size()
    # define net_loss_opt
    net = MidasNet()
    net = net.set_train()
    loss = Loss()
    param_dict = load_checkpoint(load_path)
    load_param_into_net(net, param_dict)
    backbone_params = list(filter(lambda x: 'backbone' in x.name, net.trainable_params()))
    no_backbone_params = list(filter(lambda x: 'backbone' not in x.name, net.trainable_params()))
    # no_backbone_params_lr = Tensor(dynamic_lr(5, epoch_number_total, per_step_size, 1e-4, 1e-6), mstype.float32)
    # backbone_params_lr = Tensor(dynamic_lr(5, epoch_number_total, per_step_size, 1e-5, 1e-7), mstype.float32)
    if config.lr_decay:
        group_params = [{'params': backbone_params,
                         'lr': nn.PolynomialDecayLR(config.backbone_params_lr
                                                    , config.backbone_params_end_lr,
                                                    epoch_number_total * per_step_size, config.power)},
                        {'params': no_backbone_params,
                         'lr': nn.PolynomialDecayLR(config.no_backbone_params_lr,
                                                    config.no_backbone_params_end_lr,
                                                    epoch_number_total * per_step_size, config.power)},
                        {'order_params': net.trainable_params()}]
    else:
        group_params = [{'params': backbone_params, 'lr': 1e-5},
                        {'params': no_backbone_params, 'lr': 1e-4},
                        {'order_params': net.trainable_params()}]
    optim = nn.Adam(group_params)
    netwithLoss = NetwithCell(net, loss)
    midas_net = nn.TrainOneStepCell(netwithLoss, optim)
    model = Model(midas_net)
    # define callback
    loss_cb = LossMonitor()
    time_cb = TimeMonitor()
    checkpointconfig = CheckpointConfig(saved_network=net)
    if config.is_modelarts:
        ckpoint_cb = ModelCheckpoint(prefix='Midas_{}'.format(device_id), directory=local_data_path + '/output/ckpt',
                                     config=checkpointconfig)
    else:
        ckpoint_cb = ModelCheckpoint(prefix='Midas_{}'.format(device_id), directory='./ckpt/', config=checkpointconfig)
    callbacks = [loss_cb, time_cb, ckpoint_cb]
    # train
    print("Starting Training:per_step_size={},batchsize={},epoch={}".format(per_step_size, batch_size,
                                                                            epoch_number_total))
    model.train(epoch_number_total, mix_dataset, callbacks=callbacks)
    if config.is_modelarts:
        mox.file.copy_parallel(local_data_path + "/output", output_path)


if __name__ == '__main__':
    train(mixdata_path="mixdata.json")
