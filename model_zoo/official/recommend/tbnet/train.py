# Copyright 2021 Huawei Technologies Co., Ltd
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
"""TB-Net training."""

import os
import argparse

import numpy as np
from mindspore import context, Model, Tensor
from mindspore.train.serialization import save_checkpoint
from mindspore.train.callback import Callback, TimeMonitor

from src import tbnet, config, metrics, dataset


class MyLossMonitor(Callback):
    """My loss monitor definition."""

    def epoch_end(self, run_context):
        """Print loss at each epoch end."""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())
        print('loss:' + str(loss))


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description='Train TBNet.')

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='steam',
        help="'steam' dataset is supported currently"
    )

    parser.add_argument(
        '--train_csv',
        type=str,
        required=False,
        default='train.csv',
        help="the train csv datafile inside the dataset folder"
    )

    parser.add_argument(
        '--test_csv',
        type=str,
        required=False,
        default='test.csv',
        help="the test csv datafile inside the dataset folder"
    )

    parser.add_argument(
        '--device_id',
        type=int,
        required=False,
        default=0,
        help="device id"
    )

    parser.add_argument(
        '--epochs',
        type=int,
        required=False,
        default=20,
        help="number of training epochs"
    )

    parser.add_argument(
        '--device_target',
        type=str,
        required=False,
        default='GPU',
        choices=['GPU'],
        help="run code on GPU"
    )

    parser.add_argument(
        '--run_mode',
        type=str,
        required=False,
        default='graph',
        choices=['graph', 'pynative'],
        help="run code by GRAPH mode or PYNATIVE mode"
    )

    return parser.parse_args()


def train_tbnet():
    """Training process."""
    args = get_args()

    home = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(home, 'data', args.dataset, 'config.json')
    train_csv_path = os.path.join(home, 'data', args.dataset, args.train_csv)
    test_csv_path = os.path.join(home, 'data', args.dataset, args.test_csv)
    ckpt_path = os.path.join(home, 'checkpoints')

    context.set_context(device_id=args.device_id)
    if args.run_mode == 'graph':
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    print(f"creating dataset from {train_csv_path}...")
    net_config = config.TBNetConfig(config_path)
    train_ds = dataset.create(train_csv_path, net_config.per_item_num_paths, train=True).batch(net_config.batch_size)
    test_ds = dataset.create(test_csv_path, net_config.per_item_num_paths, train=True).batch(net_config.batch_size)

    print("creating TBNet for training...")
    network = tbnet.TBNet(net_config)
    loss_net = tbnet.NetWithLossClass(network, net_config)
    train_net = tbnet.TrainStepWrap(loss_net, net_config.lr)
    train_net.set_train()
    eval_net = tbnet.PredictWithSigmoid(network)
    time_callback = TimeMonitor(data_size=train_ds.get_dataset_size())
    loss_callback = MyLossMonitor()
    model = Model(network=train_net, eval_network=eval_net, metrics={'auc': metrics.AUC(), 'acc': metrics.ACC()})
    print("training...")
    for i in range(args.epochs):
        print(f'===================== Epoch {i} =====================')
        model.train(epoch=1, train_dataset=train_ds, callbacks=[time_callback, loss_callback], dataset_sink_mode=False)
        train_out = model.eval(train_ds, dataset_sink_mode=False)
        test_out = model.eval(test_ds, dataset_sink_mode=False)
        print(f'Train AUC:{train_out["auc"]} ACC:{train_out["acc"]}  Test AUC:{test_out["auc"]} ACC:{test_out["acc"]}')

        save_checkpoint(network, os.path.join(ckpt_path, f'tbnet_epoch{i}.ckpt'))


if __name__ == '__main__':
    train_tbnet()
