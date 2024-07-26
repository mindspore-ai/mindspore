# Copyright 2024 Huawei Technologies Co., Ltd
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

"""Operator Parallel Example"""

import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops
from mindspore.communication import init, get_rank
from mindspore.common.initializer import initializer
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_context(max_device_memory="28GB")
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
init()
ms.set_seed(1)
print("distribute network.", flush=True)


class Network(nn.Cell):
    """Network"""

    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.fc1_weight = ms.Parameter(initializer("normal", [28 * 28, 512], ms.float32))
        self.fc2_weight = ms.Parameter(initializer("normal", [512, 512], ms.float32))
        self.fc3_weight = ms.Parameter(initializer("normal", [512, 10], ms.float32))
        self.matmul1 = ops.MatMul()
        self.relu1 = ops.ReLU()
        self.matmul2 = ops.MatMul()
        self.relu2 = ops.ReLU()
        self.matmul3 = ops.MatMul()

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x)
        x = self.matmul2(x, self.fc2_weight)
        x = self.relu2(x)
        logits = self.matmul3(x, self.fc3_weight)
        return logits


def create_dataset(batch_size):
    """create dataset"""
    dataset_path = "/home/workspace/mindspore_dataset/mnist/train"
    data_set = ds.MnistDataset(dataset_path)
    image_transforms = [
        ds.vision.Rescale(1.0 / 255.0, 0),
        ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ds.vision.HWC2CHW()
    ]
    label_transform = ds.transforms.TypeCast(ms.int32)
    data_set = data_set.map(image_transforms, 'image')
    data_set = data_set.map(label_transform, 'label')
    data_set = data_set.batch(batch_size)
    return data_set


def test_remove_redundancy_save_True_load_True():
    '''
    Feature: remove_redundancy save ckpt and load ckpt.
    Description: Saving and loading checkpoints with redundancy elimination.
    Expectation: success.
    '''
    print("distribute network shard.", flush=True)
    net = Network()
    print("distribute network create dataset.", flush=True)

    dataset = create_dataset(32)
    optim = nn.SGD(net.trainable_params(), 1e-2)
    loss = nn.CrossEntropyLoss()
    rank_id = get_rank()
    config = CheckpointConfig(remove_redundancy=True)
    cbpoint_cb = ModelCheckpoint(prefix="redundancy", directory=f"./device{rank_id}_redundancy11", config=config)
    print("distribute network train.", flush=True)
    model = Model(net, loss_fn=loss, optimizer=optim)
    model.train(1, dataset, callbacks=cbpoint_cb)
    ckpt_path = f"./device{rank_id}_redundancy11/redundancy-1_1875.ckpt"

    print("distribute network loadcheckpoint.", flush=True)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(model.train_network, param_dict, remove_redundancy=True)
    print("distribute network parameter broadcast.", flush=True)
    model.train(1, dataset)


def test_remove_redundancy_save_False_load_True():
    '''
    Feature: save full ckpt and remove_redundancy load ckpt.
    Description: Full checkpoint saving and redundant-free checkpoint loading.
    Expectation: success.
    '''
    print("distribute network shard.", flush=True)
    net = Network()
    print("distribute network create dataset.", flush=True)

    dataset = create_dataset(32)
    optim = nn.SGD(net.trainable_params(), 1e-2)
    loss = nn.CrossEntropyLoss()
    rank_id = get_rank()
    config = CheckpointConfig(remove_redundancy=False)
    cbpoint_cb = ModelCheckpoint(prefix="redundancy", directory=f"./device{rank_id}_redundancy01", config=config)
    print("distribute network train.", flush=True)
    model = Model(net, loss_fn=loss, optimizer=optim)
    model.train(1, dataset, callbacks=cbpoint_cb)
    ckpt_path = f"./device{rank_id}_redundancy01/redundancy-1_1875.ckpt"

    print("distribute network loadcheckpoint.", flush=True)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(model.train_network, param_dict, remove_redundancy=True)
    print("distribute network parameter broadcast.", flush=True)
    model.train(1, dataset)
