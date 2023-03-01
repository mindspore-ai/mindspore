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
from collections import namedtuple
import numpy as np
import pytest

from mindspore import ops, nn, Tensor, Parameter, ParameterTuple, context, set_seed
from mindspore.common.initializer import initializer, XavierUniform
import mindspore.dataset as ds
from mindspore.train import Callback, Model
from mindspore.common import dtype as mstype
import mindspore as ms


class CrossNet(nn.Cell):
    def __init__(self, hidden_size, num_layer, l2_reg=0):
        super(CrossNet, self).__init__()
        self.l2_reg = l2_reg
        self.num_layers = num_layer
        kernels = []
        bias_list = []
        for i in range(self.num_layers):
            kernel = Parameter(initializer(XavierUniform(0.02), (hidden_size, 1), mstype.float32),
                               requires_grad=True, name="kernerl" + str(i))
            kernels.append(kernel)
            bias = Parameter(Tensor(np.zeros((hidden_size, 1)), mstype.float32),
                             requires_grad=True, name="bias" + str(i))
            bias_list.append(bias)
        self.kernels = ParameterTuple(kernels)
        self.bias = ParameterTuple(bias_list)
        self.expand_dim = ops.ExpandDims()
        self.squeeze = ops.Squeeze(2)
        self.matmul = ops.MatMul()

    def construct(self, x):
        x_0 = self.expand_dim(x, 2)
        x_l = x_0
        for i in range(self.num_layers):
            xl_w = ops.tensor_dot(x_l, self.kernels[i], axes=(1, 0))
            dot = ops.matmul(x_0, xl_w)
            x_l = dot + self.bias[i] + x_l
        x_l = self.squeeze(x_l)
        return x_l


class DNN(nn.Cell):
    def __init__(self, input_size, hidden_units, activation='relu', l2_reg=0, dropout_rate=0):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.num_layers = len(hidden_units)
        self.hidden_units = [self.input_size] + list(hidden_units)
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        dense_layers = []
        drop_layers = []
        for i in range(self.num_layers):
            dense_layer = nn.Dense(in_channels=self.hidden_units[i], out_channels=self.hidden_units[i + 1],
                                   activation=self.activation, weight_init="heUniform")
            dense_layers.append(dense_layer)
            drop_layer = nn.Dropout(p=self.dropout_rate)
            drop_layers.append(drop_layer)
        self.dense_layers = nn.CellList(dense_layers)
        self.drop_layers = nn.CellList(drop_layers)

    def construct(self, x):
        for i in range(self.num_layers):
            x = self.dense_layers[i](x)
            x = self.drop_layers[i](x)
        return x


class DCN(nn.Cell):
    def __init__(self, numeric_size, sparse_list, hidden_units, cross_layer, output_num=1):
        super(DCN, self).__init__()
        self.embed_list = nn.CellList()
        for sparse_feature in sparse_list:
            embed = nn.Embedding(
                sparse_feature.voc_size, sparse_feature.embed_size, embedding_table='xavierUniform')
            self.embed_list.append(embed)
        self.input_size = sum(
            sparse_feature.embed_size for sparse_feature in sparse_list) + numeric_size
        self.hidden_units = hidden_units
        self.cross_layer = cross_layer
        self.cross_net = CrossNet(self.input_size, self.cross_layer)
        self.dense_net = DNN(self.input_size, self.hidden_units)
        self.output_num = output_num
        self.in_channels = self.input_size + self.hidden_units[-1]
        self.out_dense = nn.Dense(in_channels=self.in_channels, out_channels=self.output_num,
                                  has_bias=False, weight_init="xavierUniform")
        self.split = ops.Split(1, len(sparse_list))
        self.squeeze = ops.Squeeze(1)
        self.transpose = ops.Transpose()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.cast = ops.Cast()
        self.shape = ops.Shape()
        self.concat = ops.Concat(1)

    def construct(self, x, cellist):
        inputs = []
        cellist = self.transpose(cellist, (1, 0))
        for i, ele in enumerate(self.split(cellist)):
            embed = self.embed_list[i]
            inputs.append(embed(self.squeeze(ele)))
        inputs.append(x)
        concat_x = self.concat(inputs)
        cross_x = self.cross_net(concat_x)
        dense_x = self.dense_net(concat_x)
        concat_x = self.concat((cross_x, dense_x))
        x = self.out_dense(concat_x)
        return x


class PairWiseLoss(nn.Cell):
    def __init__(self):
        super(PairWiseLoss, self).__init__()
        self.sub = ops.Sub()
        self.mul = ops.Mul()
        self.relu = ops.ReLU()
        self.expandim = ops.ExpandDims()
        self.cast = ops.Cast()
        self.greater = ops.Greater()
        self.ones = Tensor(np.ones(1), mstype.float32)
        self.reduce_sum = ops.ReduceSum()

    def construct(self, y_pred, y_true):
        pairwise_label_diff = self.sub(self.expandim(y_true, 1), y_true)
        pairwise_logits = self.sub(self.expandim(y_pred, 1), y_pred)
        pairwise_labels = self.cast(self.greater(
            pairwise_label_diff, 0), mstype.float32)
        losses = self.mul(pairwise_labels, self.relu(
            self.ones - pairwise_logits))
        loss = self.reduce_sum(losses)
        return loss


class MyWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self._squeeze = ops.Squeeze(1)

    @property
    def backbone_network(self):
        return self._backbone

    def construct(self, x, y, label):
        out = self._backbone(x, y)
        out = self._squeeze(out)
        return self._loss_fn(out, label)


class LossCallback(Callback):
    def __init__(self):
        super(LossCallback, self).__init__()
        self.loss_list = []

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        result = cb_params.net_outputs
        self.loss_list.append(result.asnumpy().mean())


def gen_data(numeric_columns, sparse_columns, batch_size_list):
    np.random.seed(0)
    data_list = []
    for batch in batch_size_list:
        numeric_values = np.random.randn(
            batch, numeric_columns[0].size).astype(np.float32)
        sparse_values = []
        for sparse_column in sparse_columns:
            voc_size = sparse_column.voc_size
            sparse_value = np.random.randint(
                0, voc_size, (1, batch), dtype=np.int32)
            sparse_values.append(sparse_value)
        sparse_values = np.concatenate(sparse_values)
        label_values = np.random.randint(0, 10, batch).astype(np.float32)
        data_list.append((numeric_values, sparse_values, label_values))
    return data_list


def get_train_loss(numeric_columns, sparse_columns, data_list, mode):
    context.set_context(mode=mode, device_target="GPU")
    dataset = ds.GeneratorDataset(
        data_list, ["dense", "category", "label"], shuffle=False)
    numeric_size = numeric_columns[0].size
    net = DCN(numeric_size, sparse_columns, hidden_units=(32, 32), cross_layer=2, output_num=1)
    loss_fn = PairWiseLoss()
    loss_net = MyWithLossCell(net, loss_fn)
    train_net = nn.TrainOneStepCell(loss_net, nn.Adam(net.trainable_params(), learning_rate=1e-3, weight_decay=1e-5))
    train_net.set_train()
    train_net.set_inputs(Tensor(shape=[None, numeric_size], dtype=ms.float32),
                         Tensor(shape=[len(sparse_columns), None], dtype=ms.int32),
                         Tensor(shape=[None], dtype=ms.float32))
    loss_callback = LossCallback()
    model = Model(train_net)
    sink_step = dataset.get_dataset_size()
    model.train(sink_step, dataset, callbacks=loss_callback, sink_size=1, dataset_sink_mode=True)
    loss_list = loss_callback.loss_list
    return loss_list


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_train():
    """
    Feature: Test the dcn_dynamic network with small shape.
    Description:  The batch of inputs is dynamic.
    Expectation: Assert that results of GRAPH_MODE(static graph) are consistent with expected result.
    """
    batch_size_list = [6, 70, 123]
    DenseFeature = namedtuple("DenseFeature", ['name', 'size'])
    numeric_columns = [DenseFeature("dense", 32)]
    SparseFeature = namedtuple("SparseFeature", ['name', 'voc_size', 'embed_size'])
    sparse_columns = [SparseFeature('a', 7, 6), SparseFeature('b', 136, 18), SparseFeature('c', 3, 6)]
    data_list = gen_data(numeric_columns, sparse_columns, batch_size_list)
    # GRAPH_MODE is temporarily not supported due to some new features that are not completely complete
    set_seed(0)
    graph_loss = get_train_loss(numeric_columns, sparse_columns, data_list, context.PYNATIVE_MODE)
    expect_loss = [6.687461, 2928.5852, 8715.267]
    assert np.allclose(graph_loss, expect_loss, 1e-3, 1e-3)
