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
"""train_criteo."""
import os
import sys

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.common import set_seed

from src.autodis import ModelBuilder, AUCMetric
from src.dataset import create_dataset, DataType
from src.callback import EvalCallBack, LossCallBack
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config, train_config, data_config, model_config
from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

set_seed(1)

def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.train_data_dir = config.data_path
    config.ckpt_path = os.path.join(config.output_path, config.ckpt_path)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    '''train function'''
    config.do_eval = config.do_eval == 'True'
    rank_size = get_device_num()
    if rank_size > 1:
        if config.device_target == "Ascend":
            device_id = get_device_id()
            context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=device_id)
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
            init()
            rank_id = get_rank_id()
        else:
            print("Unsupported device_target ", config.device_target)
            exit()
    else:
        if config.device_target == "Ascend":
            device_id = get_device_id()
            context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=device_id)
        else:
            print("Unsupported device_target ", config.device_target)
            exit()
        rank_size = None
        rank_id = None

    # Init Profiler

    ds_train = create_dataset(config.train_data_dir,
                              train_mode=True,
                              epochs=1,
                              batch_size=train_config.batch_size,
                              data_type=DataType(data_config.data_format),
                              rank_size=rank_size,
                              rank_id=rank_id)
    print("ds_train.size: {}".format(ds_train.get_dataset_size()))

    # steps_size = ds_train.get_dataset_size()

    model_builder = ModelBuilder(model_config, train_config)
    train_net, eval_net = model_builder.get_train_eval_net()
    auc_metric = AUCMetric()
    model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})

    time_callback = TimeMonitor(data_size=ds_train.get_dataset_size())
    loss_callback = LossCallBack(loss_file_path=config.loss_file_name)
    callback_list = [time_callback, loss_callback]

    if train_config.save_checkpoint:
        if rank_size:
            train_config.ckpt_file_name_prefix = train_config.ckpt_file_name_prefix + str(get_rank())
            config.ckpt_path = os.path.join(config.ckpt_path, 'ckpt_' + str(get_rank()) + '/')
        config_ck = CheckpointConfig(save_checkpoint_steps=train_config.save_checkpoint_steps,
                                     keep_checkpoint_max=train_config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix=train_config.ckpt_file_name_prefix,
                                  directory=config.ckpt_path,
                                  config=config_ck)
        callback_list.append(ckpt_cb)

    if config.do_eval:
        ds_eval = create_dataset(config.train_data_dir, train_mode=False,
                                 epochs=1,
                                 batch_size=train_config.batch_size,
                                 data_type=DataType(data_config.data_format))
        eval_callback = EvalCallBack(model, ds_eval, auc_metric,
                                     eval_file_path=config.eval_file_name)
        callback_list.append(eval_callback)
    model.train(train_config.train_epochs, ds_train, callbacks=callback_list)
if __name__ == '__main__':
    run_train()
