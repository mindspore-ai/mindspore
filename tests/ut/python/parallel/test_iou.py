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

import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


_anchor_boxes = Tensor(np.ones([32, 4]), ms.float32)
_gt_boxes = Tensor(np.ones([64, 4]), ms.float32)


class Net(Cell):
    def __init__(self, strategy=None):
        super(Net, self).__init__()
        self.iou = P.IOU().shard(strategy)

    def construct(self, anchor_boxes, gt_boxes):
        x = self.iou(anchor_boxes, gt_boxes)
        return x


def compile_net(net: Cell):
    net.set_train()
    _cell_graph_executor.compile(net, _anchor_boxes, _gt_boxes)
    context.reset_auto_parallel_context()


def test_auto_parallel_iou():
    """
    Feature: test IOU auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net()
    compile_net(net)


def test_data_parallel_iou():
    """
    Feature: test IOU data parallel strategy
    Description: only shard the batch dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((2, 1), (4, 1))
    net = Net(strategy)
    compile_net(net)


def test_iou_strategy_error():
    """
    Feature: test IOU with illegal strategy
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((2, 2), (2, 1))
    net = Net(strategy)
    with pytest.raises(RuntimeError):
        compile_net(net)
    context.reset_auto_parallel_context()
