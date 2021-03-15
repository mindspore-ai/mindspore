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

import os
import argparse
import numpy as np
import mindspore.context as context
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.common import dtype as mstype
from mindspore.dataset.vision import Inter
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.nn import Flatten
from mindspore.nn import Conv2d
from mindspore.nn import BatchNorm2d
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import Adam
from mindspore.nn import EmbeddingLookup
from mindspore.nn import ReLU
import mindspore
import mindspore.ops.operations as op
from mindspore.common.parameter import Parameter
from mindspore.train import Model
from mindspore.common import set_seed

parser = argparse.ArgumentParser(description='test_ps_lenet')
parser.add_argument("--device_target", type=str, default="Ascend")
parser.add_argument("--dataset_path", type=str, default="/home/workspace/mindspore_dataset/mnist")
args, _ = parser.parse_known_args()
device_target = args.device_target
dataset_path = args.dataset_path
context.set_context(mode=context.GRAPH_MODE, device_target=device_target, enable_sparse=True)
context.set_ps_context(enable_ps=True)


class Menet(Cell):
    def __init__(self, in_channels, out_channels, kernel_size, vocab_size, embedding_size,
                 output_channels, target, sparse):
        super().__init__()
        set_seed(5)
        self.relu = ReLU()
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, has_bias=True, weight_init='normal')
        self.batchnorm = BatchNorm2d(num_features=out_channels)
        self.embedding_lookup = EmbeddingLookup(vocab_size=vocab_size,
                                                embedding_size=embedding_size,
                                                param_init='normal', target=target, sparse=sparse)
        self.flatten = Flatten()
        self.cast = op.Cast()
        self.bias = Parameter(Tensor(np.ones([output_channels]).astype(np.float32)), name='bias')
        self.biasadd = op.BiasAdd()
        self.type = mindspore.int32

    def construct(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.flatten(x)
        x = self.relu(x)
        x = self.cast(x, self.type)
        x = self.embedding_lookup(x)
        x = self.flatten(x)
        x = self.biasadd(x, self.bias)
        x = self.biasadd(x, self.bias)
        return x


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """
    create dataset for train or test
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width),
                          interpolation=Inter.LINEAR)  # Bilinear mode
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label",
                            num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image",
                            num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image",
                            num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image",
                            num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image",
                            num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


class NetFactory:
    def __init__(self, input_shape=(2, 1, 32, 32), in_channels=1, out_channels=3,
                 kernel_size=5, vocab_size=5, embedding_size=1, output_channels=3072,
                 epoch_size=1, target='CPU', sparse=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.output_channels = output_channels
        self.epoch_size = epoch_size
        self.target = target
        self.sparse = sparse
        self.input_np = np.random.randn(*input_shape).astype(np.float32)

    def no_ps_impl(self, dataset):
        context.set_ps_context(enable_ps=False)
        net = Menet(self.in_channels, self.out_channels, self.kernel_size, self.vocab_size,
                    self.embedding_size, self.output_channels, self.target, self.sparse)
        net.conv.conv2d.add_prim_attr('primitive_target', 'CPU')
        net.conv.bias_add.add_prim_attr('primitive_target', 'CPU')
        net.set_train()
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        opt = Adam(params=filter(lambda x: x.requires_grad, net.get_parameters()))
        opt.target = 'CPU'
        model = Model(net, loss, opt)
        model.train(self.epoch_size, dataset, dataset_sink_mode=False)
        input_me = Tensor(self.input_np)
        out_me = model.predict(input_me)
        context.set_ps_context(enable_ps=True)
        return out_me.asnumpy()

    def part_ps_impl(self, dataset):
        net = Menet(self.in_channels, self.out_channels, self.kernel_size, self.vocab_size,
                    self.embedding_size, self.output_channels, self.target, self.sparse)
        net.embedding_lookup.set_param_ps()
        net.conv.conv2d.add_prim_attr('primitive_target', 'CPU')
        net.conv.bias_add.add_prim_attr('primitive_target', 'CPU')
        net.set_train()
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        opt = Adam(params=filter(lambda x: x.requires_grad, net.get_parameters()))
        opt.target = 'CPU'
        model = Model(net, loss, opt)
        model.train(self.epoch_size, dataset, dataset_sink_mode=False)
        input_me = Tensor(self.input_np)
        out_me = model.predict(input_me)
        return out_me.asnumpy()

    def part_cmp(self):
        ds1 = create_dataset(os.path.join(dataset_path, "train"), 32, 1)
        ds2 = create_dataset(os.path.join(dataset_path, "train"), 32, 1)
        part_ps = self.part_ps_impl(ds1)
        no_ps = self.no_ps_impl(ds2)
        print(part_ps)
        print(no_ps)
        assert np.allclose(no_ps, part_ps, rtol=1.0e-4, atol=1.0e-4)


if __name__ == "__main__":
    fact = NetFactory()
    fact.part_cmp()
