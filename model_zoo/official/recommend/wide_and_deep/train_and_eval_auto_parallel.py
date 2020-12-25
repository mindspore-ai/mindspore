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
import mindspore.dataset as ds
from mindspore import Model, context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.context import ParallelMode
from mindspore.communication.management import get_rank, get_group_size, init
from mindspore.nn.wrap.cell_wrapper import VirtualDatasetCellTriple

from src.wide_and_deep import PredictWithSigmoid, TrainStepWrap, NetWithLossClass, WideDeepModel
from src.callbacks import LossCallBack, EvalCallBack
from src.datasets import create_dataset, DataType, compute_manual_shape
from src.metrics import AUCMetric
from src.config import WideDeepConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_WideDeep_net(config):
    """
    Get network of wide&deep model.
    """
    WideDeep_net = WideDeepModel(config)
    loss_net = NetWithLossClass(WideDeep_net, config)
    loss_net = VirtualDatasetCellTriple(loss_net)
    train_net = TrainStepWrap(loss_net, host_device_mix=bool(config.host_device_mix),
                              sparse=config.sparse)
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
    sparse = config.sparse
    print("epochs is {}".format(epochs))
    if config.full_batch:
        context.set_auto_parallel_context(full_batch=True)
        ds.config.set_seed(1)
        if config.field_slice:
            compute_manual_shape(config, get_group_size())
            ds_train = create_dataset(data_path, train_mode=True, epochs=1,
                                      batch_size=batch_size*get_group_size(), data_type=dataset_type,
                                      manual_shape=config.manual_shape, target_column=config.field_size)
            ds_eval = create_dataset(data_path, train_mode=False, epochs=1,
                                     batch_size=batch_size*get_group_size(), data_type=dataset_type,
                                     manual_shape=config.manual_shape, target_column=config.field_size)
        else:
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

    model = Model(train_net, eval_network=eval_net,
                  metrics={"auc": auc_metric})

    # Save strategy ckpts according to the rank id, this must be done before initializing the callbacks.
    config.stra_ckpt = os.path.join(config.stra_ckpt + "-{}".format(get_rank()), "strategy.ckpt")

    eval_callback = EvalCallBack(
        model, ds_eval, auc_metric, config)

    callback = LossCallBack(config=config, per_print_times=20)
    ckptconfig = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size()*epochs,
                                  keep_checkpoint_max=5, integrated_save=False)
    ckpoint_cb = ModelCheckpoint(prefix='widedeep_train',
                                 directory=os.path.join(config.ckpt_path, 'ckpt_' + str(get_rank())), config=ckptconfig)

    context.set_auto_parallel_context(strategy_ckpt_save_file=config.stra_ckpt)
    callback_list = [TimeMonitor(
        ds_train.get_dataset_size()), eval_callback, callback]
    if not host_device_mix:
        callback_list.append(ckpoint_cb)
    model.train(epochs, ds_train, callbacks=callback_list,
                dataset_sink_mode=(not sparse))


if __name__ == "__main__":
    wide_deep_config = WideDeepConfig()
    wide_deep_config.argparse_init()
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=wide_deep_config.device_target)
    context.set_context(variable_memory_max_size="24GB")
    context.set_context(enable_sparse=True)
    init()
    context.set_context(save_graphs_path='./graphs_of_device_id_' + str(get_rank()), save_graphs=True)
    if wide_deep_config.sparse:
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=True)
    else:
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
    train_and_eval(wide_deep_config)
