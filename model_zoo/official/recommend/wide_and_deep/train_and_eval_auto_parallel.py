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
"""train_multinpu."""


import os
import sys
import mindspore.dataset.engine as de
from mindspore import Model, context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train import ParallelMode
from mindspore.communication.management import get_rank, get_group_size, init
from mindspore.parallel import _cost_model_context as cost_model_context
from mindspore.nn.wrap.cell_wrapper import VirtualDatasetCellTriple

from src.wide_and_deep import PredictWithSigmoid, TrainStepWrap, NetWithLossClass, WideDeepModel
from src.callbacks import LossCallBack, EvalCallBack
from src.datasets import create_dataset, DataType
from src.metrics import AUCMetric
from src.config import WideDeepConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=True)
context.set_context(variable_memory_max_size="24GB")
context.set_context(enable_sparse=True)
cost_model_context.set_cost_model_context(multi_subgraphs=True)
init()



def get_WideDeep_net(config):
    """
    Get network of wide&deep model.
    """
    WideDeep_net = WideDeepModel(config)
    loss_net = NetWithLossClass(WideDeep_net, config)
    loss_net = VirtualDatasetCellTriple(loss_net)
    train_net = TrainStepWrap(loss_net, host_device_mix=bool(config.host_device_mix))
    eval_net = PredictWithSigmoid(WideDeep_net)
    eval_net = VirtualDatasetCellTriple(eval_net)
    return train_net, eval_net


class ModelBuilder():
    """
    ModelBuilder
    """
    def __init__(self):
        pass

    def get_hook(self):
        pass

    def get_train_hook(self):
        hooks = []
        callback = LossCallBack()
        hooks.append(callback)
        if int(os.getenv('DEVICE_ID')) == 0:
            pass
        return hooks

    def get_net(self, config):
        return get_WideDeep_net(config)


def train_and_eval(config):
    """
    test_train_eval
    """
    data_path = config.data_path
    batch_size = config.batch_size
    epochs = config.epochs
    if config.dataset_type == "tfrecord":
        dataset_type = DataType.TFRECORD
    elif config.dataset_type == "mindrecord":
        dataset_type = DataType.MINDRECORD
    else:
        dataset_type = DataType.H5
    host_device_mix = bool(config.host_device_mix)
    print("epochs is {}".format(epochs))
    if config.full_batch:
        context.set_auto_parallel_context(full_batch=True)
        de.config.set_seed(1)
        ds_train = create_dataset(data_path, train_mode=True, epochs=1,
                                  batch_size=batch_size*get_group_size(), data_type=dataset_type)
        ds_eval = create_dataset(data_path, train_mode=False, epochs=1,
                                 batch_size=batch_size*get_group_size(), data_type=dataset_type)
    else:
        ds_train = create_dataset(data_path, train_mode=True, epochs=1,
                                  batch_size=batch_size, rank_id=get_rank(),
                                  rank_size=get_group_size(), data_type=dataset_type)
        ds_eval = create_dataset(data_path, train_mode=False, epochs=1,
                                 batch_size=batch_size, rank_id=get_rank(),
                                 rank_size=get_group_size(), data_type=dataset_type)
    print("ds_train.size: {}".format(ds_train.get_dataset_size()))
    print("ds_eval.size: {}".format(ds_eval.get_dataset_size()))

    net_builder = ModelBuilder()

    train_net, eval_net = net_builder.get_net(config)
    train_net.set_train()
    auc_metric = AUCMetric()

    model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})

    eval_callback = EvalCallBack(model, ds_eval, auc_metric, config, host_device_mix=host_device_mix)

    callback = LossCallBack(config=config, per_print_times=20)
    ckptconfig = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size(), keep_checkpoint_max=5)
    ckpoint_cb = ModelCheckpoint(prefix='widedeep_train',
                                 directory=config.ckpt_path, config=ckptconfig)
    context.set_auto_parallel_context(strategy_ckpt_save_file="./strategy_train.ckpt")
    callback_list = [TimeMonitor(ds_train.get_dataset_size()), eval_callback, callback]
    if not host_device_mix:
        callback_list.append(ckpoint_cb)
    model.train(epochs, ds_train, callbacks=callback_list, dataset_sink_mode=(not host_device_mix))


if __name__ == "__main__":
    wide_deep_config = WideDeepConfig()
    wide_deep_config.argparse_init()
    if wide_deep_config.host_device_mix == 1:
        context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, mirror_mean=True)
    else:
        context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, mirror_mean=True)
    train_and_eval(wide_deep_config)
