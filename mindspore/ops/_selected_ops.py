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

""" resolve ops """
from mindspore.ops.op_selector import new_ops_selector

op_selector = new_ops_selector(
    "mindspore.ops.operations", "mindspore.nn._graph_kernels")
opt_selector = new_ops_selector(
    "mindspore.nn.optim", "mindspore.nn._graph_kernels")
nn_selector = new_ops_selector(
    "mindspore.nn", "mindspore.nn._graph_kernels")


@nn_selector
class BatchNorm2d:
    """Operator selector for BatchNorm2d."""
    def __call__(self, *args):
        pass


@op_selector
class ReLU:
    """Operator selector for ReLU."""
    def __call__(self, *args):
        pass


@op_selector
class ReduceMean:
    """Operator selector for ReduceMean."""
    def __call__(self, *args):
        pass


@op_selector
class BiasAdd:
    """Operator selector for BiasAdd."""
    def __call__(self, *args):
        pass


@op_selector
class ApplyMomentum:
    """Operator selector for ApplyMomentum."""
    def __call__(self, *args):
        pass


@op_selector
class SoftmaxCrossEntropyWithLogits:
    """Operator selector for SoftmaxCrossEntropyWithLogits."""
    def __call__(self, *args):
        pass


@op_selector
class LogSoftmax:
    """Operator selector for LogSoftmax."""
    def __call__(self, *args):
        pass


@op_selector
class Tanh:
    """Operator selector for Tanh."""
    def __call__(self, *args):
        pass


@op_selector
class GeLU:
    """Operator selector for GeLU."""
    def __call__(self, *args):
        pass


@op_selector
class FastGeLU:
    """Operator selector for FastGeLU."""
    def __call__(self, *args):
        pass


@op_selector
class LayerNorm:
    """Operator selector for LayerNorm."""
    def __call__(self, *args):
        pass


@op_selector
class Softmax:
    """Operator selector for Softmax."""
    def __call__(self, *args):
        pass


@op_selector
class LambUpdateWithLR:
    """Operator selector for LambUpdateWithLR."""
    def __call__(self, *args):
        pass


@op_selector
class LambNextMV:
    """Operator selector for LambNextMV."""
    def __call__(self, *args):
        pass


@op_selector
class LambApplyOptimizerAssign:
    """Operator selector for LambApplyOptimizerAssign."""
    def __call__(self, *args):
        pass


@op_selector
class LambApplyWeightAssign:
    """Operator selector for LambApplyWeightAssign."""
    def __call__(self, *args):
        pass
