# Copyright 2022 Huawei Technologies Co., Ltd
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

import mindspore.context as context
from mindspore import set_seed
from mindspore.train.metrics import Accuracy
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.communication.management import init

from src.lenet import Net, get_optimizer, get_loss, get_dataset

from src.args import ClusterArgs
cluster_args = ClusterArgs(description="Run test_simple_dynamic_shape case.")
config_file_path = cluster_args.config_file_path
enable_ssl = cluster_args.enable_ssl
client_password = cluster_args.client_password
server_password = cluster_args.server_password
security_ctx = {
    "config_file_path": config_file_path,
    "enable_ssl": enable_ssl,
    "client_password": client_password,
    "server_password": server_password
}
device_target = cluster_args.device_target


set_seed(2)
context.set_context(mode=context.GRAPH_MODE, device_target=device_target, save_graphs=False)


def run():
    init()
    net = Net()
    opt = get_optimizer(net)
    criterion = get_loss()
    model = Model(net, criterion, opt, metrics={"Accuracy": Accuracy()})

    print("================= Start training =================", flush=True)
    ds_train = get_dataset("/home/workspace/mindspore_dataset/mnist/train")
    model.train(10, ds_train, callbacks=[LossMonitor(), TimeMonitor()], dataset_sink_mode=False)

    print("================= Start testing =================", flush=True)
    ds_eval = get_dataset("/home/workspace/mindspore_dataset/mnist/test")
    acc = model.eval(ds_eval, dataset_sink_mode=False)

    print("Accuracy is:", acc)
    assert acc.get('Accuracy') > 0.7

run()
