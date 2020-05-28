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
""" test_training """
import os

from mindspore import Model, context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from wide_deep.models.WideDeep import PredictWithSigmoid, TrainStepWrap, NetWithLossClass, WideDeepModel
from wide_deep.utils.callbacks import LossCallBack, EvalCallBack
from wide_deep.data.datasets import create_dataset
from wide_deep.utils.metrics import AUCMetric
from tools.config import Config_WideDeep

context.set_context(mode=context.GRAPH_MODE, device_target="Davinci")


def get_WideDeep_net(config):
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


def test_train_eval(config):
    """
    test_train_eval
    """
    data_path = config.data_path
    batch_size = config.batch_size
    epochs = config.epochs
    ds_train = create_dataset(data_path, train_mode=True, epochs=epochs, batch_size=batch_size)
    ds_eval = create_dataset(data_path, train_mode=False, epochs=epochs + 1, batch_size=batch_size)
    print("ds_train.size: {}".format(ds_train.get_dataset_size()))
    print("ds_eval.size: {}".format(ds_eval.get_dataset_size()))

    net_builder = ModelBuilder()

    train_net, eval_net = net_builder.get_net(config)
    train_net.set_train()
    auc_metric = AUCMetric()

    model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})

    eval_callback = EvalCallBack(model, ds_eval, auc_metric, config)

    callback = LossCallBack()
    ckptconfig = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=5)
    ckpoint_cb = ModelCheckpoint(prefix='widedeep_train', directory=config.ckpt_path, config=ckptconfig)

    out = model.eval(ds_eval)
    print("=====" * 5 + "model.eval() initialized: {}".format(out))
    model.train(epochs, ds_train, callbacks=[eval_callback, callback, ckpoint_cb])


if __name__ == "__main__":
    wide_deep_config = Config_WideDeep()
    wide_deep_config.argparse_init()

    test_train_eval(wide_deep_config)
