# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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

"""Env related operations."""
from __future__ import absolute_import
from mindspore.ops.composite.base import MultitypeFuncGraph
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.primitive import Primitive
from mindspore.ops.operations import _grad_ops
from mindspore.ops import operations as P

env_get = MultitypeFuncGraph("env_get")
environ_get = Primitive('EnvironGet')
ref_to_embed = _grad_ops.RefToEmbed()
tensor_zeros_like = P.ZerosLike()


@env_get.register("EnvType", "Tensor")
def _tensor_env_get(env, parameter):
    """Used to get env."""
    return environ_get(env, ref_to_embed(parameter), tensor_zeros_like(parameter))


@env_get.register("EnvType", "MapTensor")
def _map_tensor_env_get(env, map_parameter):
    """Used to get env for map parameter."""
    return environ_get(env, ref_to_embed(map_parameter), zeros_like(map_parameter))
