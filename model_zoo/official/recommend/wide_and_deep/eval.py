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
from mindspore.train.serialization import load_checkpoint, load_param_into_net,\
    build_searched_strategy, merge_sliced_parameter

from src.wide_and_deep import PredictWithSigmoid, TrainStepWrap, NetWithLossClass, WideDeepModel
from src.callbacks import LossCallBack, EvalCallBack
from src.datasets import create_dataset, DataType
from src.metrics import AUCMetric
from src.config import WideDeepConfig


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
    if config.dataset_type == "tfrecord":
        dataset_type = DataType.TFRECORD
    elif config.dataset_type == "mindrecord":
        dataset_type = DataType.MINDRECORD
    else:
        dataset_type = DataType.H5
    ds_eval = create_dataset(data_path, train_mode=False, epochs=1,
                             batch_size=batch_size, data_type=dataset_type)
    print("ds_eval.size: {}".format(ds_eval.get_dataset_size()))

    net_builder = ModelBuilder()
    train_net, eval_net = net_builder.get_net(config)
    ckpt_path = config.ckpt_path
    if ";" in ckpt_path:
        ckpt_paths = ckpt_path.split(';')
        param_list_dict = {}
        strategy = build_searched_strategy(config.stra_ckpt)
        for slice_path in ckpt_paths:
            param_slice_dict = load_checkpoint(slice_path)
            for key, value in param_slice_dict.items():
                if 'optimizer' in key:
                    continue
                if key not in param_list_dict:
                    param_list_dict[key] = []
                param_list_dict[key].append(value)
        param_dict = {}
        for key, value in param_list_dict.items():
            if key in strategy:
                merged_parameter = merge_sliced_parameter(value, strategy)
            else:
                merged_parameter = merge_sliced_parameter(value)
            param_dict[key] = merged_parameter
    else:
        param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(eval_net, param_dict)

    auc_metric = AUCMetric()
    model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})

    eval_callback = EvalCallBack(model, ds_eval, auc_metric, config)

    model.eval(ds_eval, callbacks=eval_callback)


if __name__ == "__main__":
    widedeep_config = WideDeepConfig()
    widedeep_config.argparse_init()

    context.set_context(mode=context.GRAPH_MODE, device_target=widedeep_config.device_target)
    test_eval(widedeep_config)
