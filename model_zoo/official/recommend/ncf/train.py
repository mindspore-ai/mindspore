# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Training entry file"""
import os
from absl import logging

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore import context, Model
from mindspore.context import ParallelMode
from mindspore.communication.management import init
from mindspore.common import set_seed

from src.dataset import create_dataset
from src.ncf import NCFModel, NetWithLossClass, TrainStepWrap

from model_utils.moxing_adapter import moxing_wrapper
from model_utils.config import config
from model_utils.device_adapter import get_device_id, get_device_num, get_rank_id, get_job_id

set_seed(1)

logging.set_verbosity(logging.INFO)

def modelarts_pre_process():
    config.checkpoint_path = os.path.join(config.output_path, str(get_rank_id()), config.checkpoint_path)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """train entry method"""
    print(config)
    print("device id: ", get_device_id())
    print("device num: ", get_device_num())
    print("rank id: ", get_rank_id())
    print("job id: ", get_job_id())

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    config.is_distributed = bool(get_device_num() > 1)
    if config.is_distributed:
        config.group_size = get_device_num()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=config.group_size, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          parameter_broadcast=True, gradients_mean=True)

        if config.device_target == "Ascend":
            context.set_context(device_id=get_device_id())
            init()
        elif config.device_target == "GPU":
            init()
    else:
        context.set_context(device_id=get_device_id())

    layers = config.layers
    num_factors = config.num_factors
    epochs = config.train_epochs

    ds_train, num_train_users, num_train_items = create_dataset(test_train=True, data_dir=config.data_path,
                                                                dataset=config.dataset, train_epochs=1,
                                                                batch_size=config.batch_size, num_neg=config.num_neg)
    print("ds_train.size: {}".format(ds_train.get_dataset_size()))

    ncf_net = NCFModel(num_users=num_train_users,
                       num_items=num_train_items,
                       num_factors=num_factors,
                       model_layers=layers,
                       mf_regularization=0,
                       mlp_reg_layers=[0.0, 0.0, 0.0, 0.0],
                       mf_dim=16)
    loss_net = NetWithLossClass(ncf_net)
    if config.device_target == "Ascend":
        loss_scale = 16384.0
    else:
        loss_scale = 1.0

    train_net = TrainStepWrap(loss_net, ds_train.get_dataset_size() * (epochs + 1), sens=loss_scale)
    train_net.set_train()

    model = Model(train_net)
    callback = LossMonitor(per_print_times=ds_train.get_dataset_size())
    ckpt_config = CheckpointConfig(save_checkpoint_steps=(4970845+config.batch_size-1)//(config.batch_size),
                                   keep_checkpoint_max=100)
    ckpoint_cb = ModelCheckpoint(prefix='NCF', directory=config.checkpoint_path, config=ckpt_config)
    model.train(epochs,
                ds_train,
                callbacks=[TimeMonitor(ds_train.get_dataset_size()), callback, ckpoint_cb],
                dataset_sink_mode=True)
    print("="*100 + "Training Finish!" + "="*100)

if __name__ == '__main__':
    run_train()
