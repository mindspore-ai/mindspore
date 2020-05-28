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

"""Test bert ops check gradient."""

from mindspore import context
from mindspore.ops import operations as P
from ..mindspore_test import mindspore_test
from ..pipeline.gradient.compare_gradient import \
    pipeline_for_compare_inputs_grad_with_numerical_diff_for_group_by_group_config, \
    pipeline_for_compare_inputs_jacobian_with_numerical_diff_for_group_by_group_config

# from ...vm_impl import *


verification_set = {
    'inputs': [
        {
            'id': 'MatMul',
            'group': 'bert',
            'desc_inputs': [
                [3, 3],
                [3, 3]
            ]
        },
    ],
    'function': [
        {
            'id': 'MatMul',
            'group': 'bert',
            'block': P.MatMul(),
            'reduce_output': False
        }
    ],
    'ext': {}
}


@mindspore_test(pipeline_for_compare_inputs_grad_with_numerical_diff_for_group_by_group_config)
def test_bert_ops_check_gradient_exec_1():
    context.set_context(mode=context.PYNATIVE_MODE)
    return verification_set


@mindspore_test(pipeline_for_compare_inputs_jacobian_with_numerical_diff_for_group_by_group_config)
def test_bert_ops_check_gradient_exec_2():
    context.set_context(mode=context.PYNATIVE_MODE)
    return verification_set
