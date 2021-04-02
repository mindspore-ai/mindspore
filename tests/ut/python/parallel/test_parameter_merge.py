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
import os
import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell, Momentum
from mindspore.ops import operations as P
from mindspore.train import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from tests.dataset_mock import MindData


class Dataset(MindData):
    def __init__(self, predict, label, length=3):
        super(Dataset, self).__init__(size=length)
        self.predict = predict
        self.label = label
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.predict, self.label

    def reset(self):
        self.index = 0


class Net(Cell):
    def __init__(self, weight, w2, begin, end, strides, strategy1=None, strategy2=None, mask=0):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.strided_slice = P.StridedSlice(begin_mask=mask).shard(strategy2)
        self.weight = Parameter(weight, "w1")
        self.mul2 = P.Mul()
        self.weight2 = Parameter(w2, "w2")
        self.begin = begin
        self.end = end
        self.strides = strides

    def construct(self, x, b):
        out = self.strided_slice(
            self.weight, self.begin, self.end, self.strides)
        out = self.mul(x, out)
        out = self.mul2(out, self.weight2)
        return out


_x = Tensor(np.ones([16, 64, 1]), dtype=ms.float32)
_b = Tensor(np.ones([16, 64, 32]), dtype=ms.float32)
_w1 = Tensor(np.ones([256, 64, 32]), dtype=ms.float32)
_w2 = Tensor(np.ones([128, 64, 1]), dtype=ms.float32)


def clean_all_ckpt_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ckpt') or file_name.endswith('.meta'):
                os.remove(os.path.join(folder_path, file_name))


def compile_net(net):
    context.set_context(save_graphs=False)
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2
    dataset = Dataset(_x, _b)
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, optimizer=opt)
    ckpt_config = CheckpointConfig(keep_checkpoint_max=1)
    ckpt_path = "./parallel_ckpt"
    ckpt_cb = ModelCheckpoint(prefix="parallel", directory=ckpt_path, config=ckpt_config)
    model.train(epoch_size, dataset, dataset_sink_mode=False, callbacks=[ckpt_cb])
    assert len(model._train_network.parallel_parameter_merge_net_dict) == 4
    clean_all_ckpt_files(ckpt_path)
    context.reset_auto_parallel_context()


def test_stridedslice_parameter():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 4, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), (1, 1, 1),
              strategy1, strategy2)
    compile_net(net)
