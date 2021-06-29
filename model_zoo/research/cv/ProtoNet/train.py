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
ProtoNet train script.
"""
import os
import datetime
import mindspore.nn as nn
from mindspore.train import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore import dataset as ds
import mindspore.context as context
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from src.EvalCallBack import EvalCallBack
from src.protonet import WithLossCell
from src.PrototypicalLoss import PrototypicalLoss
from src.parser_util import get_parser
from src.protonet import ProtoNet
from model_init import init_dataloader

local_data_url = './cache/data'
local_train_url = './cache/out'


def train(opt, tr_dataloader, net, loss_fn, eval_loss_fn, optim, path, val_dataloader=None):
    '''
    train function
    '''

    inp = ds.GeneratorDataset(tr_dataloader, column_names=['data', 'label', 'classes'])
    my_loss_cell = WithLossCell(net, loss_fn)
    my_acc_cell = WithLossCell(net, eval_loss_fn)
    model = Model(my_loss_cell, optimizer=optim)

    eval_data = ds.GeneratorDataset(val_dataloader, column_names=['data', 'label', 'classes'])

    eval_cb = EvalCallBack(opt, my_acc_cell, eval_data, path)
    config = CheckpointConfig(save_checkpoint_steps=10,
                              keep_checkpoint_max=5,
                              saved_network=net)
    ckpoint_cb = ModelCheckpoint(prefix='protonet', directory=path, config=config)

    print('==========training test==========')
    starttime = datetime.datetime.now()
    model.train(opt.epochs, inp, callbacks=[ckpoint_cb, eval_cb, TimeMonitor()])
    endtime = datetime.datetime.now()
    print('epoch time: ', (endtime - starttime).seconds / 10, 'per step time:', (endtime - starttime).seconds / 1000)


def main():
    '''
    main function
    '''
    global local_data_url
    global local_train_url

    options = get_parser().parse_args()

    if options.run_offline:

        device_num = int(os.environ.get("DEVICE_NUM", 1))

        if device_num > 1:

            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        context.set_context(device_id=options.device_id)
        local_data_url = options.dataset_root
        local_train_url = options.experiment_root
        if not os.path.exists(options.experiment_root):
            os.makedirs(options.experiment_root)
    else:
        device_num = int(os.environ.get("DEVICE_NUM", 1))
        device_id = int(os.getenv("DEVICE_ID"))

        import moxing as mox
        if not os.path.exists(local_train_url):
            os.makedirs(local_train_url)

        context.set_context(device_id=device_id)

        if device_num > 1:

            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            local_data_url = os.path.join(local_data_url, str(device_id))
            local_train_url = os.path.join(local_train_url, str(device_id))

        mox.file.copy_parallel(src_url=options.data_url, dst_url=local_data_url)

    tr_dataloader = init_dataloader(options, 'train', local_data_url)
    val_dataloader = init_dataloader(options, 'val', local_data_url)

    loss_fn = PrototypicalLoss(options.num_support_tr, options.num_query_tr, options.classes_per_it_tr)
    eval_loss_fn = PrototypicalLoss(options.num_support_tr, options.num_query_tr, options.classes_per_it_val,
                                    is_train=False)

    Net = ProtoNet()
    optim = nn.Adam(params=Net.trainable_params(), learning_rate=0.001)
    train(options, tr_dataloader, Net, loss_fn, eval_loss_fn, optim, local_train_url, val_dataloader)
    if not options.run_offline:
        mox.file.copy_parallel(src_url='./cache/out', dst_url=options.train_url)


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE)
    main()
