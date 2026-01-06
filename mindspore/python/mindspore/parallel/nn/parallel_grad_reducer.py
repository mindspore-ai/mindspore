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
"""parallel serialization"""
from __future__ import absolute_import

__all__ = ['PipelineGradReducer']

from mindspore.nn.cell import Cell
from mindspore.ops import functional as F, composite as C, operations as P
import mindspore.common.dtype as mstype
from mindspore.common.sparse_tensor import Tensor
from mindspore.common.api import jit
from mindspore.common.parameter import Parameter
from mindspore.nn.layer import Identity
from mindspore.parallel._utils import _get_enable_parallel_optimizer


grad_scale = C.MultitypeFuncGraph("grad_scale")
shard_grad_scale = C.MultitypeFuncGraph("shard_grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor", "Tensor")
def tensor_grad_scale_pipeline(scale, grad, accu_grad):
    accu_grad = F.depend(accu_grad, grad)
    new_grad = accu_grad * reciprocal(scale)
    accu_grad = F.depend(accu_grad, new_grad)
    zeros = F.tensor_mul(accu_grad, 0.0)
    new_grad = F.depend(new_grad, F.assign(accu_grad, zeros))
    return new_grad


@shard_grad_scale.register("Tensor", "Tensor", "Tensor")
def tensor_shard_grad_scale_pipeline(scale, grad, accu_grad):
    new_grad = grad * reciprocal(scale)
    accu_grad = F.depend(accu_grad, new_grad)
    new_grad = F.depend(new_grad, F.assign(accu_grad, F.zeros_like(accu_grad)))
    return new_grad


class PipelineGradReducer(Cell):
    """
    Functional training scenarios for gradient statute and accumulation of pipeline parallel.

    Args:
        parameters (list): the parameters to be updated.
        scale_sense (float, optional): the scale sense of the gradient. Default: 1.0.
        opt_shard(bool, optional): if use parallel optimizer, set opt_shard True. Default: ``None``.

    Raise:
        RuntimeError: If the mode is not graph mode.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_

            This example should be run with multiple devices.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import nn, ops, Tensor
        >>> from mindspore.communication import init
        >>> from mindspore.parallel.auto_parallel import AutoParallel
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> ms.reset_auto_parallel_context()
        >>> init()
        >>> ms.set_seed(1)
        >>>
        >>> class Network(nn.Cell):
        ...     def __init__(self, in_features, out_features, sens=1.0):
        ...         super().__init__()
        ...         self.layer1 = nn.Dense(in_features, 16)
        ...         self.relu1 = nn.ReLU()
        ...         self.layer2 = nn.Dense(16, 16)
        ...         self.relu2 = nn.ReLU()
        ...         self.layer3 = nn.Dense(16, out_features)
        ...
        ...     def construct(self, x):
        ...         x = self.layer1(x)
        ...         x = self.relu1(x)
        ...         x = self.layer2(x)
        ...         x = self.relu2(x)
        ...         logits = self.layer3(x)
        ...         return logits
        >>>
        >>> size, in_features, out_features = 16, 32, 10
        >>> net = Network(in_features, out_features)
        >>> net.layer1.pipeline_stage = 0
        >>> net.relu1.pipeline_stage = 0
        >>> net.layer2.pipeline_stage = 0
        >>> net.relu2.pipeline_stage = 1
        >>> net.layer3.pipeline_stage = 1
        >>> loss_fn = nn.CrossEntropyLoss()
        >>> optimizer = nn.SGD(net.trainable_params(), 1e-2)
        >>> net_with_loss = nn.PipelineCell(nn.WithLossCell(net, loss_fn), 2)
        >>> net_with_loss.set_train()
        >>> def forward_fn(inputs, target):
        ...     loss = net_with_loss(inputs, target)
        ...     return loss
        >>>
        >>> grad_fn = ops.value_and_grad(forward_fn, None, net_with_loss.trainable_params())
        >>> pp_grad_reducer = nn.PipelineGradReducer(optimizer.parameters)
        >>>
        >>> @ms.jit
        >>> def train_one_step(inputs, target):
        ...     loss, grads = grad_fn(inputs, target)
        ...     grads = pp_grad_reducer(grads)
        ...     optimizer(grads)
        ...     return loss, grads
        >>>
        >>> parallel_net = AutoParallel(train_one_step, parallel_mode="semi_auto")
        >>> parallel_net.pipeline(stages=2)
        >>> inputs = Tensor(np.ones([size, in_features]).astype(np.float32))
        >>> label = Tensor(np.ones([size, out_features]).astype(np.float32))
        >>> loss, _ = train_one_step(inputs, label)
        >>> print(loss)
        46.304886
    """
    def __init__(self, parameters, scale_sense=1.0, opt_shard=None):
        super(PipelineGradReducer, self).__init__(auto_prefix=False) # pylint: disable=super-with-arguments
        self.accu_grads = parameters.clone(prefix="accu_grads", init="zeros")
        self.grad_reducer = Identity()
        self.degree = Tensor(1, mstype.float32)
        self.scale_sense = Parameter(scale_sense, name='scale_sense')
        self.hyper_map = C.HyperMap()
        if opt_shard is None:
            self.opt_shard = _get_enable_parallel_optimizer()
        else:
            self.opt_shard = opt_shard

    @jit
    def construct(self, *args, **kwargs):
        new_grads = None
        if self.opt_shard:
            grads = self.grad_reducer(*args, **kwargs)
            new_grads = self.hyper_map(F.partial(shard_grad_scale, self.scale_sense * self.degree),
                                       grads, self.accu_grads)
        else:
            accu_grads = self.grad_reducer(self.accu_grads)
            new_grads = self.hyper_map(F.partial(grad_scale, self.scale_sense * self.degree), grads, accu_grads)
        return new_grads
