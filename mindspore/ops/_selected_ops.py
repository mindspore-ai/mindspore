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
    def __call__(self, *args):
        pass


@op_selector
class ReLU:
    def __call__(self, *args):
        pass


@op_selector
class ReduceMean:
    def __call__(self, *args):
        pass


@op_selector
class BiasAdd:
    def __call__(self, *args):
        pass


@op_selector
class ApplyMomentum:
    def __call__(self, *args):
        pass


@op_selector
class SoftmaxCrossEntropyWithLogits:
    def __call__(self, *args):
        pass


@op_selector
class LogSoftmax:
    def __call__(self, *args):
        pass


@op_selector
class Tanh:
    def __call__(self, *args):
        pass


@op_selector
class GeLU:
    def __call__(self, *args):
        pass


@op_selector
class FastGeLU:
    def __call__(self, *args):
        pass


@op_selector
class LayerNorm:
    def __call__(self, *args):
        pass


@op_selector
class Softmax:
    def __call__(self, *args):
        pass


@op_selector
class LambUpdateWithLR:
    def __call__(self, *args):
        pass


@op_selector
class LambNextMV:
    def __call__(self, *args):
        pass


@op_selector
class LambApplyOptimizerAssign:
    def __call__(self, *args):
        pass


@op_selector
class LambApplyWeightAssign:
    def __call__(self, *args):
        pass
