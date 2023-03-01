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
import time
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore.common.initializer import XavierUniform
import mindspore.nn as nn
import mindspore.context as context

from gnngraph_dataset import GraphDataset, GatherNet, CSRReduceSumNet

DATASET_PATH = "/home/workspace/mindspore_dataset/cora/cora_mr/cora_v2_with_mask.npz"
FEAT_DROPOUT = 0.5
EDGE_DROPOUT = 0.5
EPOCHS = 200
EPOCHS_PERF = 5
NUM_HIDDEN = 64
LR = 1e-2
WEIGHT_DECAY = 5e-4
ALPHA = 0.1
K = 10
SEED = 20


class APPNPConv(ms.nn.Cell):
    def __init__(self,
                 k: int,
                 alpha: float,
                 edge_drop=1.0,
                 indptr_backward=None,
                 indices_backward=None) -> None:
        super().__init__()
        self.k_ = k
        self.alpha_ = alpha
        self.edge_drop = ms.nn.Dropout(p=1.0 - edge_drop)
        self.min_clip = Tensor(1, ms.int32)
        self.max_clip = Tensor(10000000, ms.int32)
        self.gather = GatherNet(indptr_backward, indices_backward)
        self.csr_reduce_sum = CSRReduceSumNet(indices_backward)

    def construct(self, x, in_deg, out_deg, n_nodes, indptr, indices):
        out_deg = ms.ops.clip_by_value(out_deg, self.min_clip, self.max_clip)
        out_deg = ms.ops.Reshape()(ms.ops.Pow()(out_deg, -0.5), ms.ops.Shape()(out_deg) + (1,))
        in_deg = ms.ops.clip_by_value(in_deg, self.min_clip, self.max_clip)
        in_deg = ms.ops.Reshape()(ms.ops.Pow()(in_deg, -0.5), ms.ops.Shape()(in_deg) + (1,))
        feat0 = x
        for _ in range(self.k_):
            u_x = self.gather(x, indices, 0)
            u_in_deg = self.gather(in_deg, indices, 0)
            edge = u_x * u_in_deg
            v_h = self.csr_reduce_sum(indptr, indices, edge, (n_nodes, n_nodes) + edge.shape[1:], 1)
            v_h = ms.ops.reshape(v_h, (n_nodes,) + v_h.shape[2:])
            v_h = v_h * out_deg
            x = (1 - self.alpha_) * v_h + self.alpha_ * feat0
        return x


class APPNPNet(nn.Cell):
    def __init__(self,
                 in_feats,
                 hidden_dim: int,
                 n_classes: int,
                 feat_dropout: float,
                 edge_dropout: float,
                 alpha,
                 k,
                 activation: ms.nn.Cell = None,
                 indptr_backward=None,
                 indices_backward=None):
        super().__init__()
        self.fc0 = nn.Dense(in_feats, hidden_dim, weight_init=XavierUniform())
        self.fc1 = nn.Dense(hidden_dim, n_classes, weight_init=XavierUniform())
        self.act = activation()
        self.feat_drop = nn.Dropout(p=1.0 - feat_dropout)
        self.propagate = APPNPConv(k, alpha, edge_dropout, indptr_backward, indices_backward)

    def construct(self, x, in_deg, out_deg, n_nodes, indptr, indices):
        x = self.feat_drop(x)
        x = self.act(self.fc0(x))
        x = self.fc1(self.feat_drop(x))
        x = self.propagate(x, in_deg, out_deg, n_nodes, indptr, indices)
        return x


class LossNet(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, x, in_deg, out_deg, train_mask, target, n_nodes, indptr, indices):
        predict = self.net(x, in_deg, out_deg, n_nodes, indptr, indices)
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
        self.n_nodes = int(ds.n_nodes)
        self.net = net

    def construct(self):
        return self.net(self.x, self.in_deg, self.out_deg, self.train_mask, self.y,
                        self.n_nodes, self.indptr, self.indices)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_appnp_csr():
    """
    Feature: Test APPNP model with CSR optimizations.
    Description: Test APPNP model in graph mode with CSR-related cluster ops.
    Expectation: Success.
    """
    np.random.seed(SEED)
    context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True,
                        graph_kernel_flags="--enable_expand_ops=Gather --enable_cluster_ops=CSRReduceSum,CSRDiv "
                                           "--enable_recompute_fusion=false "
                                           "--enable_parallel_fusion=false "
                                           "--recompute_increment_threshold=40000000 "
                                           "--recompute_peak_threshold=3000000000 "
                                           "--enable_csr_fusion=true ")
    # dataloader
    ds = GraphDataset(DATASET_PATH)
    feature_size = ds.x.shape[1]
    # model
    net = APPNPNet(in_feats=feature_size,
                   hidden_dim=NUM_HIDDEN,
                   n_classes=ds.n_classes,
                   activation=ms.nn.ReLU,
                   feat_dropout=FEAT_DROPOUT,
                   edge_dropout=EDGE_DROPOUT,
                   alpha=ALPHA,
                   k=K,
                   indptr_backward=ds.indptr_backward,
                   indices_backward=ds.indices_backward)

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
    print("Model:{} Dataset:{} Avg epoch time:{} Loss: {}".format("APPNP", DATASET_PATH, avg_dur,
                                                                  train_loss))
    assert train_loss < 0.35
    assert avg_dur < 3.0


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_appnp_csr_perf():
    """
    Feature: Test APPNP model with CSR optimizations.
    Description: Test APPNP model in graph mode with CSR-related cluster ops.
    Expectation: Success.
    """
    np.random.seed(SEED)
    context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True,
                        graph_kernel_flags="--enable_expand_ops=Gather --enable_cluster_ops=CSRReduceSum,CSRDiv "
                                           "--enable_recompute_fusion=false "
                                           "--enable_parallel_fusion=false "
                                           "--recompute_increment_threshold=40000000 "
                                           "--recompute_peak_threshold=3000000000 "
                                           "--enable_csr_fusion=true ")
    # dataloader
    ds = GraphDataset()
    feature_size = ds.x.shape[1]
    # model
    net = APPNPNet(in_feats=feature_size,
                   hidden_dim=NUM_HIDDEN,
                   n_classes=ds.n_classes,
                   activation=ms.nn.ReLU,
                   feat_dropout=FEAT_DROPOUT,
                   edge_dropout=EDGE_DROPOUT,
                   alpha=ALPHA,
                   k=K,
                   indptr_backward=ds.indptr_backward,
                   indices_backward=ds.indices_backward)

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
    print("Model:{} Dataset:{} Avg epoch time:{}".format("APPNP", "PERF", avg_dur))
    assert avg_dur < 1200
