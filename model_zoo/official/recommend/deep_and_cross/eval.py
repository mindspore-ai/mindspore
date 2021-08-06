# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

"""Parse arguments"""
from mindspore import Model, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.deep_and_cross import PredictWithSigmoid, TrainStepWrap, NetWithLossClass, DeepCrossModel
from src.callbacks import EvalCallBack
from src.datasets import create_dataset, DataType
from src.metrics import AUCMetric
from src.config import DeepCrossConfig


def get_DCN_net(configure):
    """
    Get network of deep&cross model.
    """
    DCN_net = DeepCrossModel(configure)

    loss_net = NetWithLossClass(DCN_net)
    train_net = TrainStepWrap(loss_net)
    eval_net = PredictWithSigmoid(DCN_net)

    return train_net, eval_net

class ModelBuilder():
    """
    Build the model.
    """
    def __init__(self):
        pass

    def get_hook(self):
        pass

    def get_net(self, configure):
        return get_DCN_net(configure)

def test_eval(configure):
    """
    test_eval
    """
    data_path = configure.data_path
    batch_size = configure.batch_size
    field_size = configure.field_size
    if configure.dataset_type == "tfrecord":
        dataset_type = DataType.TFRECORD
    elif configure.dataset_type == "mindrecord":
        dataset_type = DataType.MINDRECORD
    else:
        dataset_type = DataType.H5
    ds_eval = create_dataset(data_path, train_mode=False, epochs=1,
                             batch_size=batch_size, data_type=dataset_type, target_column=field_size+1)
    print("ds_eval.size: {}".format(ds_eval.get_dataset_size()))
    net_builder = ModelBuilder()
    train_net, eval_net = net_builder.get_net(configure)
    ckpt_path = configure.ckpt_path
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(eval_net, param_dict)
    auc_metric = AUCMetric()
    model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})
    eval_callback = EvalCallBack(model, ds_eval, auc_metric, configure)
    model.eval(ds_eval, callbacks=eval_callback)


if __name__ == "__main__":
    config = DeepCrossConfig()
    config.argparse_init()
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    test_eval(config)
