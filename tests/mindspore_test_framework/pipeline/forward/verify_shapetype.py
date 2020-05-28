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

"""Pipelines for shape and type checking."""

from ...components.executor.exec_forward import IdentityEC
from ...components.expect_result_policy.cartesian_product_on_group_for_expect_result \
    import GroupCartesianProductERPC
from ...components.function.get_function_from_config import IdentityBC
from ...components.function_inputs_policy.cartesian_product_on_group_for_function_inputs \
    import GroupCartesianProductFIPC
from ...components.inputs.get_inputs_from_config import IdentityDC
from ...components.verifier.verify_shapetype import ShapeTypeVC

# pylint: disable=W0105
"""
Test if operator's result shape is correct. The pipeline will apply function with group id to any inputs
with same group id to generate test cases. The pipeline is suitable for config in a grouped style.

Example:
    verification_set = {
    'function': [
        {
            'id': 'Add',
            'group': 'op-test',
            'block': Add
        }
    ],
    'inputs': [
        {
            'id': '1',
            'group': 'op-test',
            'desc_inputs': [
                np.array([[1, 1], [1, 1]]).astype(np.float32),
                np.array([[2, 2], [2, 2]]).astype(np.float32)
            ]
        },
        {
            'id': '2',
            'group': 'op-test',
            'desc_inputs': [
                np.array([[3, 3], [3, 3]]).astype(np.float32),
                np.array([[4, 4], [4, 4]]).astype(np.float32)
            ]
        }
    ],
    'expect': [
        {
            'id': '1',
            'group': 'op-test-op-test',
            'desc_expect': {
                'shape_type': [
                    {
                        'type': np.float32,
                        'shape': (2, 2)
                    }
                ]
            }
        }
    ],
    'ext': {}
}
"""
pipeline_for_verify_shapetype_for_group_by_group_config = [IdentityDC, IdentityBC, GroupCartesianProductFIPC,
                                                           IdentityEC, GroupCartesianProductERPC, ShapeTypeVC]
