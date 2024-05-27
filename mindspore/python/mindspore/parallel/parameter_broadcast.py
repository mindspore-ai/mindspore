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
"""Parameter broadcast"""
from __future__ import absolute_import

__all__ = ["parameter_broadcast"]

import numpy as np


def parameter_broadcast(net, layout, cur_rank=0, initial_rank=0):
    """
    Broadcast parameter to other rank in data parallel dimension.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        net (Cell): The network where the parameters will be broadcasted.
        layout (Dict): Parameter layout dictionary. Come from
            :func:`mindspore.nn.Cell.parameter_layout_dict`
            or read from file(for example: "strategy.ckpt" saved by using the
            `strategy_ckpt_config` parameter of :func:`mindspore.set_auto_parallel_context`).
            The key is param name, the value is the layout of this parameter.
        cur_rank (int, optional): current rank id. Default: ``0``.
        initial_rank (int, optional): Start rank id for each pipeline. Default: ``0``.

    Raises:
        ValueError: `cur_rank` is not rank id of current rank.
        ValueError: `initial_rank` is not the start rank id of current pipeline stage.
        ValueError: Parameter name in `layout` can not be found in
            :func:`mindspore.nn.Cell.parameters_dict`.

    Examples:
        >>> import os
        >>> import mindspore as ms
        >>> import mindspore.dataset as ds
        >>> from mindspore import nn, ops
        >>> from mindspore.communication import init
        >>> from mindspore.common.initializer import initializer
        >>> from mindspore.train import Model
        >>> from mindspore.parallel.parameter_broadcast import parameter_broadcast
        >>> from mindspore.train.serialization import load_checkpoint, load_param_into_net
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> ms.set_context(max_device_memory="28GB")
        >>> ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
        >>> init()
        >>> ms.set_seed(1)
        >>> class Network(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.flatten = ops.Flatten()
        ...         self.fc1_weight = ms.Parameter(initializer("normal", [28*28, 512], ms.float32))
        ...         self.fc2_weight = ms.Parameter(initializer("normal", [512, 512], ms.float32))
        ...         self.fc3_weight = ms.Parameter(initializer("normal", [512, 10], ms.float32))
        ...         self.matmul1 = ops.MatMul()
        ...         self.relu1 = ops.ReLU()
        ...         self.matmul2 = ops.MatMul()
        ...         self.relu2 = ops.ReLU()
        ...         self.matmul3 = ops.MatMul()
        ...     def construct(self, x):
        ...         x = self.flatten(x)
        ...         x = self.matmul1(x, self.fc1_weight)
        ...         x = self.relu1(x)
        ...         x = self.matmul2(x, self.fc2_weight)
        ...         x = self.relu2(x)
        ...         logits = self.matmul3(x, self.fc3_weight)
        ...         return logits
        >>> net = Network()
        >>> net.matmul1.shard(((2, 4), (4, 1)))
        >>> net.relu1.shard(((4, 1),))
        >>> net.matmul2.shard(((1, 8), (8, 1)))
        >>> net.relu2.shard(((8, 1),))
        >>> # Create the dataset taking MNIST as an example. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/mnist.py
        >>> dataset = create_dataset()
        >>> optim = nn.SGD(net.trainable_params(), 1e-2)
        >>> loss = nn.CrossEntropyLoss()
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
        >>> model.train(1, dataset)
        >>> ms.save_checkpoint(net, "./simple.ckpt", False)
        >>> layout = model.train_network.parameter_layout_dict
        >>> param_dict = load_checkpoint("./simple.ckpt")
        >>> load_param_into_net(net, param_dict)
        >>> rank_id = os.environ["RANK_ID"]
        >>> parameter_broadcast(model.train_network, layout, int(rank_id), 0)
        >>> class LossCallBack(Callback):
        ...     def step_end(self, run_context):
        ...         cb_params = run_context.original_args()
        ...         print("step end, cur step num: ", cb_params.cur_step_num, flush=True)
        >>> model.train(1, dataset, callbacks=[LossCallBack()])
    """
    if not layout:
        return
    import mindspore as ms
    from mindspore import Tensor
    from mindspore.communication import get_rank, create_group, get_group_size
    from mindspore.train._utils import get_parameter_redundancy, remove_param_redundancy
    from mindspore.nn.wrap.cell_wrapper import AllreduceGraph
    origin_parallel_mode = ms.get_auto_parallel_context("parallel_mode")
    if origin_parallel_mode not in ("semi_auto_parallel", "auto_parallel"):
        return
    if cur_rank != get_rank():
        raise ValueError(f"For parameter broadcast, the cur_rank: {cur_rank} is wrong.")
    if initial_rank % (get_group_size() / ms.get_auto_parallel_context("pipeline_stages")) != 0:
        raise ValueError(f"For parameter broadcast, the initial_rank: {initial_rank} is wrong.")
    param_redundancy = get_parameter_redundancy(layout, initial_rank)
    if not param_redundancy:
        return
    single_params = remove_param_redundancy(param_redundancy)
    if not single_params:
        return
    param_redundancy_reversed = {}
    for key, redundancy in param_redundancy.items():
        for item in redundancy:
            if len(item) == 1:
                continue
            if cur_rank in item:
                param_redundancy_reversed.setdefault(item, []).append(key)
    if not param_redundancy_reversed:
        return
    if cur_rank not in single_params:
        return
    net_param_dict = net.parameters_dict()
    ms.set_auto_parallel_context(parallel_mode="hybrid_parallel")
    for group, params in param_redundancy_reversed.items():
        create_group(str(group), list(group))
        allreduce_input = []
        for param in params:
            if param not in net_param_dict:
                raise ValueError(f"For parameter broadcast, the param: {param} can not be found.")
            real_param = net_param_dict[param]
            if param not in single_params[cur_rank]:
                real_param.set_data(Tensor(np.zeros(real_param.shape), dtype=real_param.dtype))
            allreduce_input.append(real_param)
        if not allreduce_input:
            continue
        allreduce_graph = AllreduceGraph(allreduce_input, str(group))
        allreduce_graph()
    ms.set_auto_parallel_context(parallel_mode=origin_parallel_mode)
