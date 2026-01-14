# Copyright 2025 Huawei Technologies Co., Ltd
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
""" Test load and export """

import os
import stat
import shutil
import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore.nn import Flatten, Conv2d, EmbeddingLookup, GraphCell, ReLU, Momentum, SoftmaxCrossEntropyWithLogits
from mindspore.train import Model
from mindspore.train.serialization import export, load
from tests.st.auto_parallel.utils.dataset_utils import FakeData
from tests.mark_utils import arg_mark


data_path = os.path.split(os.path.abspath(__file__))[0] + "/data/net/"


class Menet(nn.Cell):
    def __init__(self, vocab_size, embedding_size, export_train):
        super().__init__()
        self.relu = ReLU()
        self.conv = Conv2d(in_channels=3, out_channels=3,
                           kernel_size=3, has_bias=True, weight_init='normal')
        self.embedding_lookup = EmbeddingLookup(vocab_size=vocab_size,
                                                embedding_size=embedding_size,
                                                param_init='normal', target='DEVICE', sparse=False)
        self.flatten = Flatten()
        self.cast = ops.Cast()
        self.split = ops.Split(0, 2)
        self.type = ms.int32
        self.export_train = export_train

    def construct(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.relu(x)
        x = self.cast(x, self.type)
        x = self.embedding_lookup(x)
        x = self.flatten(x)
        if not self.export_train:
            x = self.split(x)
        return x


def net_input(file_prefix_bin, input_shape=(8, 3, 32, 32), dtype=np.float32,
              label_shape=(8, 307200)):
    input_np = np.random.randn(*input_shape).astype(dtype)
    label = np.random.randint(0, 2, size=label_shape).astype(dtype)
    input_np.tofile(data_path + "{}_input.bin".format(file_prefix_bin))
    np.save(data_path + "{}_input.npy".format(file_prefix_bin), input_np)
    input_me = ms.Tensor(input_np)
    return input_me, label


def one_input_network(network, dataset, epoch_size):
    opt = Momentum(learning_rate=0.0001, momentum=0.0009,
                   params=filter(lambda x: x.requires_grad, network.get_parameters()))
    loss = SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
    model = Model(network, loss, opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)


def save_net_info_ret_subfix(file_format, pb_prefix, export_input_shape):
    infer_shape = str(export_input_shape).strip("(").strip(")").replace(" ", "")
    net_info = "{} {} FLOAT32 0,255 RANDOM input.1 170 mindir 0.01 0.01 #".format(pb_prefix,
                                                                                  infer_shape)
    if file_format == "MINDIR":
        net_info_path = os.path.join(data_path, "{}/".format(pb_prefix))
        if not os.path.isdir(os.path.dirname(net_info_path)):
            os.makedirs(os.path.dirname(net_info_path))
        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        fname = os.path.join(net_info_path, "new_net_info.txt")
        with os.fdopen(os.open(fname, flags, modes), 'w') as f:
            f.write(net_info)


def export_mindir(dataset, file_prefix, file_prefix_bin, export_train, net, input_me, label):
    epoch_size=2
    out = None
    if export_train:
        one_input_network(net, dataset, epoch_size)
    if not export_train:
        out = net(input_me)
    save_net_info_ret_subfix('MINDIR', file_prefix, input_me.asnumpy().shape)
    export(net, input_me, file_name=data_path + file_prefix, file_format='MINDIR')
    if export_train:
        out = net(input_me)
    if isinstance(out, tuple):
        num = len(out)
        for i in range(num):
            out[i].asnumpy().tofile(data_path + "{}_output_{}.bin".format(file_prefix_bin, i))
            np.save(data_path + "{}_output_{}.npy".format(file_prefix_bin, i),
                    out[i].asnumpy())
    else:
        out.asnumpy().tofile(data_path + "{}_output.bin".format(file_prefix_bin))
        np.save(data_path + "{}_output.npy".format(file_prefix_bin), out.asnumpy())
    return out


def check_files_authority(path, expr_auth):
    real_auth = oct(os.stat(path).st_mode)[-3:]
    assert int(real_auth) == expr_auth


def load_mindir(input_me, mindir_files, method="method1", loss_fn=None):
    mindir_file = data_path + mindir_files
    check_files_authority(mindir_file, 400)
    graph = load(mindir_file)
    graph_cell = GraphCell(graph)

    model = Model(network=graph_cell, loss_fn=loss_fn, optimizer=None)
    output = model.predict(input_me)
    return output


def cmp_mindir(out, output):
    if isinstance(out, tuple):
        num = len(out)
        for i in range(num):
            assert np.allclose(out[i].asnumpy(), output[i].asnumpy(), 0.0001, 0.0001)
    else:
        assert np.allclose(out.asnumpy(), output.asnumpy(), 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=230.00)
def test_export_model_train_load_model():
    """
    Feature: MindIR load and export
    Description: Test MindIR load and export
    Expectation: No exception.
    """
    if not os.path.isdir(os.path.dirname(data_path)):
        os.makedirs(os.path.dirname(data_path))

    ms.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    file_prefix = "test_export_model_train_load_model"
    file_prefix_bin = file_prefix
    mindir_files = file_prefix + ".mindir"
    export_train = True
    net = Menet(vocab_size=30, embedding_size=100, export_train=export_train)
    net_inputs = net_input(file_prefix_bin)
    dataset = FakeData(size=1024, batch_size=32, image_size=(3, 32, 32), num_classes=307200)
    export_me = export_mindir(dataset, file_prefix, file_prefix_bin, export_train, net,
                              net_inputs[0], net_inputs[1])
    load_me = load_mindir(net_inputs[0], mindir_files)
    cmp_mindir(export_me, load_me)

    if os.path.isdir(os.path.dirname(data_path)):
        shutil.rmtree(os.path.dirname(data_path))
