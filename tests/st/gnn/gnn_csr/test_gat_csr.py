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
import math
import time
import pytest
import numpy as np
import mindspore as ms
from mindspore.common.initializer import initializer, XavierUniform
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context

from gnngraph_dataset import GraphDataset


DATASET_PATH = "/home/workspace/mindspore_dataset/cora/cora_mr/cora_v2_with_mask.npz"
NUM_LAYERS = 1
NUM_HIDDEN = 8
NUM_HEADS = 8
NUM_OUT_HEADS = 1
IN_DROP = 0.6
ATTN_DROP = 0.6
NEGATIVE_SLOPE = 0.2
EPOCHS = 200
EPOCHS_PERF = 5
LR = 0.005
WEIGHT_DECAY = 5e-4
SEED = 20


class GATConv(ms.nn.Cell):
    def __init__(self,
                 in_feat_size: int,
                 out_size: int,
                 num_attn_head: int,
                 input_drop_out_rate: float = 1.0,
                 attn_drop_out_rate: float = 1.0,
                 leaky_relu_slope: float = 0.2,
                 activation=None,
                 add_norm=False) -> None:
        super().__init__()
        self.in_feat_size = in_feat_size
        self.out_size = out_size
        self.num_attn_head = num_attn_head
        input_drop_out_rate = input_drop_out_rate
        attn_drop_out_rate = attn_drop_out_rate
        leaky_relu_slope = leaky_relu_slope
        add_norm = add_norm
        self.reshape = ms.ops.Reshape()
        gain = math.sqrt(2)
        self.fc = ms.nn.Dense(in_feat_size, out_size * num_attn_head, weight_init=XavierUniform(gain), has_bias=False)
        self.attn_s = ms.Parameter(initializer(XavierUniform(gain), [num_attn_head, out_size], ms.float32),
                                   name="attn_s")
        self.attn_d = ms.Parameter(initializer(XavierUniform(gain), [num_attn_head, out_size], ms.float32),
                                   name="attn_d")
        self.bias = ms.Parameter(initializer('zero', [num_attn_head, out_size], ms.float32), name='bias')
        self.feat_drop = ms.nn.Dropout(p=1.0 - input_drop_out_rate)
        self.attn_drop = ms.nn.Dropout(p=1.0 - attn_drop_out_rate)
        self.leaky_relu = ms.nn.LeakyReLU(leaky_relu_slope)
        self.exp = ms.ops.Exp()
        if add_norm:
            self.norm_constant = ms.Tensor(100, ms.float32)
            self.norm_div = ms.ops.Div()
        else:
            self.norm_div = None
        self.activation = activation

    def construct(self, x, n_nodes, row_indices, indptr, indices):
        x = self.feat_drop(x)
        x = self.fc(x)
        feat_src = feat_dst = ms.ops.Reshape()(x, (-1, self.num_attn_head, self.out_size))
        ed = ms.ops.ReduceSum(True)(feat_dst * self.attn_d, -1)
        ed = ms.ops.gather(ed, row_indices, 0)
        es = ms.ops.ReduceSum(True)(feat_src * self.attn_s, -1)
        es = ms.ops.gather(es, indices, 0)
        if self.norm_div is not None:
            edge = self.leaky_relu(es + ed)
            edge = self.exp(self.norm_div(edge, self.norm_constant))
        else:
            edge = self.exp(self.leaky_relu(es + ed))
        csr_edge = ms.CSRTensor(indptr, indices, edge, (n_nodes, n_nodes, self.num_attn_head, 1))
        edge_sum = ms.ops.csr_reduce_sum(csr_edge, 1)
        edge_sum = ms.ops.maximum(1e-5, edge_sum)
        feat_avg = feat_src / edge_sum.reshape((n_nodes, self.num_attn_head, 1))
        feat_src = ms.ops.gather(feat_avg, indices, 0)
        csr_attn_feat = ms.CSRTensor(
            indptr, indices, feat_src * edge, (n_nodes, n_nodes, self.num_attn_head, self.out_size))
        v_h = ms.ops.csr_reduce_sum(csr_attn_feat, 1)
        v_h = v_h + self.bias
        if self.activation:
            v_h = self.activation(v_h)
        return ms.ops.Flatten()(v_h)


class GatNet(ms.nn.Cell):
    def __init__(self,
                 num_layers: int,
                 data_feat_size: int,
                 hidden_dim_size: int,
                 n_classes: int,
                 heads,
                 input_drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 leaky_relu_slope: float = 0.2,
                 add_norm: bool = True,
                 activation: ms.nn.Cell = None):
        super().__init__()
        self.layer0 = GATConv(data_feat_size, hidden_dim_size, heads[0], input_drop_out_rate, attn_drop_out_rate,
                              leaky_relu_slope, activation(), add_norm)
        self.mid_layers = []
        for i in range(0, num_layers):
            self.mid_layers.append(GATConv(hidden_dim_size * heads[i], n_classes, heads[i + 1], input_drop_out_rate,
                                           attn_drop_out_rate, leaky_relu_slope, None, add_norm))

    def construct(self, x, n_nodes, row_indices, indptr, indices):
        x = self.layer0(x, n_nodes, row_indices, indptr, indices)
        for layer in self.mid_layers:
            x = layer(x, n_nodes, row_indices, indptr, indices)
        return x


class LossNet(nn.Cell):
    """ LossNet definition """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, x, target, train_mask, n_nodes, row_indices, indptr, indices):
        predict = self.net(x, n_nodes, row_indices, indptr, indices)
        target = ops.Squeeze()(target)
        loss = self.loss_fn(predict, target)
        loss = loss * train_mask
        return ms.ops.ReduceSum()(loss) / ms.ops.ReduceSum()(train_mask)


class DataNet(ms.nn.Cell):
    def __init__(self, ds, net):
        super().__init__()
        self.x = ds.x
        self.in_deg = ds.in_deg
        self.out_deg = ds.out_deg
        self.train_mask = ms.Tensor(ds.train_mask, ms.float32)
        self.y = ds.y
        self.indptr = ds.indptr
        self.indices = ds.indices
        self.row_indices = ds.row_indices
        self.n_nodes = int(ds.n_nodes)
        self.net = net

    def construct(self):
        return self.net(self.x, self.y, self.train_mask, self.n_nodes, self.row_indices, self.indptr, self.indices)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gat_csr():
    """
    Feature: Test GAT model with CSR optimizations.
    Description: Test GAT model in graph mode with CSR-related cluster ops.
    Expectation: Success.
    """
    np.random.seed(SEED)
    context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True,
                        graph_kernel_flags="--enable_expand_ops=Gather "
                                           "--enable_cluster_ops=UnsortedSegmentSum,CSRReduceSum,CSRGather "
                                           "--enable_recompute_fusion=false "
                                           "--enable_parallel_fusion=false "
                                           "--recompute_increment_threshold=40000000 "
                                           "--recompute_peak_threshold=3000000000 "
                                           "--enable_csr_fusion=true ")
    # dataloader
    ds = GraphDataset(DATASET_PATH)
    feature_size = ds.x.shape[1]
    # model
    net = GatNet(num_layers=NUM_LAYERS,
                 data_feat_size=feature_size,
                 hidden_dim_size=NUM_HIDDEN,
                 n_classes=ds.n_classes,
                 heads=[NUM_HEADS for _ in range(NUM_LAYERS)] + [NUM_OUT_HEADS],
                 input_drop_out_rate=IN_DROP,
                 attn_drop_out_rate=ATTN_DROP,
                 leaky_relu_slope=NEGATIVE_SLOPE,
                 activation=ms.nn.ELU,
                 add_norm=True)
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=LR, weight_decay=WEIGHT_DECAY)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)
    train_net = DataNet(ds, train_net)
    total = 0.
    warm_up = 3
    for e in range(EPOCHS):
        beg = time.time()
        train_net.set_train()
        train_net.set_grad()
        train_loss = train_net()
        end = time.time()
        dur = end - beg
        if e >= warm_up:
            total = total + dur
    avg_dur = total * 1000 / (EPOCHS - warm_up)
    print("Model:{} Dataset:{} Avg epoch time:{} Loss: {}".format("GAT", DATASET_PATH, avg_dur,
                                                                  train_loss))
    assert train_loss < 1.2
    assert avg_dur < 2.0


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gat_csr_perf():
    """
    Feature: Test GAT model with CSR optimizations.
    Description: Test GAT model in graph mode with CSR-related cluster ops.
    Expectation: Success.
    """
    np.random.seed(SEED)
    context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True,
                        graph_kernel_flags="--enable_expand_ops=Gather "
                                           "--enable_cluster_ops=UnsortedSegmentSum,CSRReduceSum,CSRGather "
                                           "--enable_recompute_fusion=false "
                                           "--enable_parallel_fusion=false "
                                           "--recompute_increment_threshold=40000000 "
                                           "--recompute_peak_threshold=3000000000 "
                                           "--enable_csr_fusion=true ")
    # dataloader
    ds = GraphDataset()
    feature_size = ds.x.shape[1]
    # model
    net = GatNet(num_layers=NUM_LAYERS,
                 data_feat_size=feature_size,
                 hidden_dim_size=NUM_HIDDEN,
                 n_classes=ds.n_classes,
                 heads=[NUM_HEADS for _ in range(NUM_LAYERS)] + [NUM_OUT_HEADS],
                 input_drop_out_rate=IN_DROP,
                 attn_drop_out_rate=ATTN_DROP,
                 leaky_relu_slope=NEGATIVE_SLOPE,
                 activation=ms.nn.ELU,
                 add_norm=True)
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=LR, weight_decay=WEIGHT_DECAY)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)
    train_net = DataNet(ds, train_net)
    total = 0.
    warm_up = 3
    for e in range(EPOCHS_PERF):
        beg = time.time()
        train_net.set_train()
        train_net.set_grad()
        _ = train_net()
        end = time.time()
        dur = end - beg
        if e >= warm_up:
            total = total + dur
    avg_dur = total * 1000 / (EPOCHS_PERF - warm_up)
    print("Model:{} Dataset:{} Avg epoch time:{}".format("GAT", "PERF", avg_dur))
    assert avg_dur < 300
