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

"""Component that construct function that init params with random function and return gradients wrt inputs."""

from mindspore.ops.composite import GradOperation
from ...components.icomponent import IBuilderComponent
from ...utils.block_util import run_block, gen_grad_net, create_funcs, get_uniform_with_shape


class RunBackwardBlockWrtInputsWithRandParamBC(IBuilderComponent):
    def __call__(self):
        grad_op = GradOperation(get_all=True, sens_param=True)
        return create_funcs(self.verification_set, gen_grad_net, run_block, grad_op, get_uniform_with_shape)
