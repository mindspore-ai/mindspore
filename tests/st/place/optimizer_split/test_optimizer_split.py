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
from mindspore.train import Accuracy
from mindspore.train import Model
from mindspore.train import LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank

from src.net import SplitRefWithoutOptimNet, SplitOptimNet, get_optimizer, get_loss, get_dataset

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
net_name = cluster_args.net_name


context.set_context(mode=context.GRAPH_MODE, device_target=device_target, save_graphs=False)

net_name_map = {
    "split_ref_without_optim": SplitRefWithoutOptimNet,
    "split_optim": SplitOptimNet
}

net_need_split_opt_map = {
    "split_ref_without_optim": False,
    "split_optim": True
}


def run_dist():
    init()
    net = net_name_map.get(net_name)(dist=True)
    opt = get_optimizer(net, dist=net_need_split_opt_map.get(net_name))
    criterion = get_loss()
    model = Model(net, criterion, opt, metrics={"Accuracy": Accuracy()})

    print("================= Start training =================", flush=True)
    ds_train = get_dataset("/home/workspace/mindspore_dataset/mnist/train")
    model.train(5, ds_train, callbacks=[LossMonitor(), TimeMonitor()], dataset_sink_mode=False)

    print("================= Start testing distributed =================", flush=True)
    ds_eval = get_dataset("/home/workspace/mindspore_dataset/mnist/test")
    acc = model.eval(ds_eval, dataset_sink_mode=False)

    return acc.get('Accuracy')


def run_single():
    net = net_name_map.get(net_name)()
    opt = get_optimizer(net)
    criterion = get_loss()
    model = Model(net, criterion, opt, metrics={"Accuracy": Accuracy()})

    print("================= Start training =================", flush=True)
    ds_train = get_dataset("/home/workspace/mindspore_dataset/mnist/train")
    model.train(5, ds_train, callbacks=[LossMonitor(), TimeMonitor()], dataset_sink_mode=False)

    print("================= Start testing single =================", flush=True)
    ds_eval = get_dataset("/home/workspace/mindspore_dataset/mnist/test")
    acc = model.eval(ds_eval, dataset_sink_mode=False)

    return acc.get('Accuracy')

set_seed(2)
acc1 = run_dist()
print("Distributed accuracy is:", acc1, flush=True)

set_seed(2)
acc2 = run_single()
print("Single accuracy is:", acc2, flush=True)

# Only worker 0 has the meaning of precision.
if get_rank() == 0:
    assert abs(acc1 - acc2) < 1e-6
