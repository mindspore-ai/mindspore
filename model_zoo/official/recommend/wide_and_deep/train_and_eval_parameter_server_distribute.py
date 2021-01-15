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
"""train distribute on parameter server."""


import os
import sys
import mindspore.dataset as ds
from mindspore import Model, context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.context import ParallelMode
from mindspore.communication.management import get_rank, get_group_size, init
from mindspore.nn.wrap.cell_wrapper import VirtualDatasetCellTriple
from mindspore.common import set_seed
from mindspore.parallel._ps_context import _is_role_worker

from src.wide_and_deep import PredictWithSigmoid, TrainStepWrap, NetWithLossClass, WideDeepModel
from src.callbacks import LossCallBack, EvalCallBack
from src.datasets import create_dataset, DataType
from src.metrics import AUCMetric
from src.config import WideDeepConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_wide_deep_net(config):
    """
    Get network of wide&deep model.
    """
    wide_deep_net = WideDeepModel(config)
    loss_net = NetWithLossClass(wide_deep_net, config)
    if cache_enable:
        loss_net = VirtualDatasetCellTriple(loss_net)
    train_net = TrainStepWrap(loss_net, parameter_server=bool(config.parameter_server),
                              sparse=config.sparse, cache_enable=(config.vocab_cache_size > 0))
    eval_net = PredictWithSigmoid(wide_deep_net)
    if cache_enable:
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
        return get_wide_deep_net(config)


def train_and_eval(config):
    """
    test_train_eval
    """
    set_seed(1000)
    data_path = config.data_path
    batch_size = config.batch_size
    epochs = config.epochs
    if config.dataset_type == "tfrecord":
        dataset_type = DataType.TFRECORD
    elif config.dataset_type == "mindrecord":
        dataset_type = DataType.MINDRECORD
    else:
        dataset_type = DataType.H5
    parameter_server = bool(config.parameter_server)
    if cache_enable:
        config.full_batch = True
    print("epochs is {}".format(epochs))
    if config.full_batch:
        context.set_auto_parallel_context(full_batch=True)
        ds.config.set_seed(1)
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

    if cache_enable:
        config.stra_ckpt = os.path.join(config.stra_ckpt + "-{}".format(get_rank()), "strategy.ckpt")
        context.set_auto_parallel_context(strategy_ckpt_save_file=config.stra_ckpt)

    eval_callback = EvalCallBack(model, ds_eval, auc_metric, config)

    callback = LossCallBack(config=config)
    if _is_role_worker():
        if cache_enable:
            ckptconfig = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size()*epochs,
                                          keep_checkpoint_max=1, integrated_save=False)
        else:
            ckptconfig = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size(), keep_checkpoint_max=5)
    else:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix='widedeep_train',
                                 directory=config.ckpt_path + '/ckpt_' + str(get_rank()) + '/',
                                 config=ckptconfig)
    callback_list = [TimeMonitor(ds_train.get_dataset_size()), eval_callback, callback]
    if get_rank() == 0:
        callback_list.append(ckpoint_cb)
    model.train(epochs, ds_train,
                callbacks=callback_list,
                dataset_sink_mode=(parameter_server and cache_enable))


if __name__ == "__main__":
    wide_deep_config = WideDeepConfig()
    wide_deep_config.argparse_init()
    context.set_context(mode=context.GRAPH_MODE, device_target=wide_deep_config.device_target, save_graphs=True)
    cache_enable = wide_deep_config.vocab_cache_size > 0
    if cache_enable and wide_deep_config.device_target != "GPU":
        context.set_context(variable_memory_max_size="24GB")
    context.set_ps_context(enable_ps=True)
    init()
    context.set_context(save_graphs_path='./graphs_of_device_id_'+str(get_rank()))

    if cache_enable:
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
    else:
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=get_group_size())
        wide_deep_config.sparse = True

    if wide_deep_config.sparse:
        context.set_context(enable_sparse=True)
    train_and_eval(wide_deep_config)
