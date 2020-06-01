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

""" test_training """

import os

from mindspore import Model, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.wide_and_deep import PredictWithSigmoid, TrainStepWrap, NetWithLossClass, WideDeepModel
from src.callbacks import LossCallBack, EvalCallBack
from src.datasets import create_dataset
from src.metrics import AUCMetric
from src.config import WideDeepConfig

context.set_context(mode=context.GRAPH_MODE, device_target="Davinci",
                    save_graphs=True)


def get_WideDeep_net(config):
    WideDeep_net = WideDeepModel(config)

    loss_net = NetWithLossClass(WideDeep_net, config)
    train_net = TrainStepWrap(loss_net)
    eval_net = PredictWithSigmoid(WideDeep_net)

    return train_net, eval_net


class ModelBuilder():
    """
    Wide and deep model builder
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


def test_eval(config):
    """
    test evaluate
    """
    data_path = config.data_path
    batch_size = config.batch_size
    ds_eval = create_dataset(data_path, train_mode=False, epochs=2,
                             batch_size=batch_size)
    print("ds_eval.size: {}".format(ds_eval.get_dataset_size()))

    net_builder = ModelBuilder()
    train_net, eval_net = net_builder.get_net(config)

    param_dict = load_checkpoint(config.ckpt_path)
    load_param_into_net(eval_net, param_dict)

    auc_metric = AUCMetric()
    model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})

    eval_callback = EvalCallBack(model, ds_eval, auc_metric, config)

    model.eval(ds_eval, callbacks=eval_callback)


if __name__ == "__main__":
    widedeep_config = WideDeepConfig()
    widedeep_config.argparse_init()

    test_eval(widedeep_config)
