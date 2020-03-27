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
"""
######################## train and test lenet example ########################
1. train lenet and get network model files(.ckpt) :
python main.py --data_path /home/workspace/mindspore_dataset/Tutorial_Network/Lenet/MNIST_Data

2. test lenet according to model file:
python main.py --data_path /home/workspace/mindspore_dataset/Tutorial_Network/Lenet/MNIST_Data
               --mode test --ckpt_path checkpoint_lenet_1-1_1875.ckpt
"""
import os
import argparse
from config import mnist_cfg as cfg

import mindspore.dataengine as de
import mindspore.nn as nn
from mindspore.model_zoo.lenet import LeNet5
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train import Model
import mindspore.ops.operations as P
import mindspore.transforms.c_transforms as C
from mindspore.transforms import Inter
from mindspore.nn.metrics import Accuracy
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


class CrossEntropyLoss(nn.Cell):
    """
    Define loss for network
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = P.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)

    def construct(self, logits, label):
        label = self.one_hot(label, F.shape(logits)[1], self.on_value, self.off_value)
        loss = self.cross_entropy(logits, label)[0]
        loss = self.mean(loss, (-1,))
        return loss

def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """
    create dataset for train or test
    """
    # define dataset
    ds1 = de.MnistDataset(data_path)

    # apply map operations on images
    ds1 = ds1.map(input_columns="label", operations=C.TypeCast(mstype.int32))
    ds1 = ds1.map(input_columns="image", operations=C.Resize((cfg.image_height, cfg.image_width),
                                                             interpolation=Inter.LINEAR),
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=C.Rescale(1 / 0.3081, -1 * 0.1307 / 0.3081),
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=C.Rescale(1.0 / 255.0, 0.0),
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=C.HWC2CHW(), num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    ds1 = ds1.shuffle(buffer_size=cfg.buffer_size)  # 10000 as in LeNet train script
    ds1 = ds1.batch(batch_size, drop_remainder=True)
    ds1 = ds1.repeat(repeat_size)

    return ds1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore MNIST Example')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'test'],
                        help='implement phase, set to train or test')
    parser.add_argument('--data_path', type=str, default="./MNIST_Data",
                        help='path where the dataset is saved')
    parser.add_argument('--ckpt_path', type=str, default="", help='if mode is test, must provide\
                        path where the trained ckpt file')

    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    network = LeNet5(cfg.num_classes)
    network.set_train()
    # net_loss = nn.SoftmaxCrossEntropyWithLogits()  # support this loss soon
    net_loss = CrossEntropyLoss()
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    if args.mode == 'train':  # train
        ds = create_dataset(os.path.join(args.data_path, args.mode), batch_size=cfg.batch_size,
                            repeat_size=cfg.epoch_size)
        print("============== Starting Training ==============")
        model.train(cfg['epoch_size'], ds, callbacks=[ckpoint_cb, LossMonitor()], dataset_sink_mode=False)
    elif args.mode == 'test':  # test
        print("============== Starting Testing ==============")
        param_dict = load_checkpoint(args.ckpt_path)
        load_param_into_net(network, param_dict)
        ds_eval = create_dataset(os.path.join(args.data_path, "test"), 32, 1)
        acc = model.eval(ds_eval, dataset_sink_mode=False)
        print("============== Accuracy:{} ==============".format(acc))
    else:
        raise RuntimeError('mode should be train or test, rather than {}'.format(args.mode))
