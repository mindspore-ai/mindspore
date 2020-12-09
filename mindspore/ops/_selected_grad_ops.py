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

""" resolved grad ops """
from mindspore.ops.op_selector import new_ops_selector

op_selector = new_ops_selector(
    "mindspore.ops.operations._grad_ops", "mindspore.nn._graph_kernels")


@op_selector
class MaximumGrad:
    def __call__(self, *args):
        pass


@op_selector
class MinimumGrad:
    def __call__(self, *args):
        pass


@op_selector
class AbsGrad:
    def __call__(self, *args):
        pass


@op_selector
class BiasAddGrad:
    def __call__(self, *args):
        pass


@op_selector
class TanhGrad:
    def __call__(self, *args):
        pass
