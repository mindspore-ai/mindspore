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
import numpy as np
from mindspore import Model, context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train import ParallelMode
from mindspore.communication.management import get_rank, get_group_size, init

from src.wide_and_deep import PredictWithSigmoid, TrainStepWrap, NetWithLossClass, WideDeepModel
from src.callbacks import LossCallBack, EvalCallBack
from src.datasets import create_dataset
from src.metrics import AUCMetric
from src.config import WideDeepConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_WideDeep_net(config):
    """
    Get network of wide&deep model.
    """
    WideDeep_net = WideDeepModel(config)
    loss_net = NetWithLossClass(WideDeep_net, config)
    train_net = TrainStepWrap(loss_net)
    eval_net = PredictWithSigmoid(WideDeep_net)
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
    np.random.seed(1000)
    data_path = config.data_path
    batch_size = config.batch_size
    epochs = config.epochs
    print("epochs is {}".format(epochs))
    ds_train = create_dataset(data_path, train_mode=True, epochs=epochs,
                              batch_size=batch_size, rank_id=get_rank(), rank_size=get_group_size())
    ds_eval = create_dataset(data_path, train_mode=False, epochs=epochs + 1,
                             batch_size=batch_size, rank_id=get_rank(), rank_size=get_group_size())
    print("ds_train.size: {}".format(ds_train.get_dataset_size()))
    print("ds_eval.size: {}".format(ds_eval.get_dataset_size()))

    net_builder = ModelBuilder()

    train_net, eval_net = net_builder.get_net(config)
    train_net.set_train()
    auc_metric = AUCMetric()

    model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})

    eval_callback = EvalCallBack(model, ds_eval, auc_metric, config)

    callback = LossCallBack(config=config)
    ckptconfig = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size(), keep_checkpoint_max=5)
    if config.device_target == "Ascend":
        ckpoint_cb = ModelCheckpoint(prefix='widedeep_train',
                                     directory=config.ckpt_path, config=ckptconfig)
    elif config.device_target == "GPU":
        ckpoint_cb = ModelCheckpoint(prefix='widedeep_train_' + str(get_rank()),
                                     directory=config.ckpt_path, config=ckptconfig)
    out = model.eval(ds_eval)
    print("=====" * 5 + "model.eval() initialized: {}".format(out))
    model.train(epochs, ds_train,
                callbacks=[TimeMonitor(ds_train.get_dataset_size()), eval_callback, callback, ckpoint_cb])


if __name__ == "__main__":
    wide_deep_config = WideDeepConfig()
    wide_deep_config.argparse_init()

    context.set_context(mode=context.GRAPH_MODE, device_target=wide_deep_config.device_target, save_graphs=True)
    if wide_deep_config.device_target == "Ascend":
        init("hccl")
    elif wide_deep_config.device_target == "GPU":
        init("nccl")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, mirror_mean=True,
                                      device_num=get_group_size())

    train_and_eval(wide_deep_config)
