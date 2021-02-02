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

"""Pipelines for gradients comparison."""

from ...components.executor.check_gradient_wrt_inputs import CheckGradientWrtInputsEC
from ...components.executor.check_gradient_wrt_params import CheckGradientWrtParamsEC
from ...components.executor.check_jacobian_wrt_inputs import CheckJacobianWrtInputsEC
from ...components.executor.exec_gradient import IdentityBackwardEC
from ...components.expect_result_policy.cartesian_product_on_id_for_expect_result import IdCartesianProductERPC
from ...components.facade.me_facade import MeFacadeFC
from ...components.function.get_function_from_config import IdentityBC
from ...components.function.run_gradient_wrt_inputs import RunBackwardBlockWrtInputsBC
from ...components.function.run_gradient_wrt_params import RunBackwardBlockWrtParamsBC
from ...components.function_inputs_policy.cartesian_product_on_id_for_function_inputs import IdCartesianProductFIPC
from ...components.inputs.generate_inputs_from_shape import GenerateFromShapeDC
from ...components.inputs.load_inputs_from_npy import LoadFromNpyDC
from ...components.verifier.compare_gradient import CompareGradientWithVC
from ...components.verifier.verify_expect_from_npy import LoadFromNpyVC

# pylint: disable=W0105
"""
Compare inputs gradient with user defined operator's gradient. This pipeline is suitable for
case-by-case style config.

Example:
    verification_set = [
        ('Add', {
            'block': (P.Add(), {'reduce_output': False}),
            'desc_inputs': [[1, 3, 3, 4], [1, 3, 3, 4]],
            'desc_bprop': [[1, 3, 3, 4]],
            'desc_expect': {
                'compare_gradient_with': [
                    (run_user_defined_grad, user_defined.add)
                ],
            }
        })
    ]
"""
pipeline_for_compare_inputs_grad_with_user_defined_for_case_by_case_config = \
    [MeFacadeFC, GenerateFromShapeDC,
     RunBackwardBlockWrtInputsBC, IdCartesianProductFIPC,
     IdentityBackwardEC, IdCartesianProductERPC,
     CompareGradientWithVC]

"""
Compare inputs gradient with data stored in npy file. This pipeline is suitable for
case-by-case style config.

Example:
    verification_set = [
        ('MatMul', {
            'block': P.MatMul(),
            'desc_inputs': [
                ('tests/mindspore_test_framework/apps/data/input_0.npy', {
                    'dtype': np.float32
                }),
                ('tests/mindspore_test_framework/apps/data/input_1.npy', {
                    'dtype': np.float32
                }),
            ],
            'desc_bprop': [
                np.ones(shape=(2, 2)).astype(np.float32)
            ],
            'desc_expect': [
                ('tests/mindspore_test_framework/apps/data/grad_0.npy', {
                    'dtype': np.float32,
                    'check_tolerance': True,
                    'relative_tolerance': 1e-2,
                    'absolute_tolerance': 1e-2
                }),
                ('tests/mindspore_test_framework/apps/data/grad_1.npy', {
                    'dtype': np.float32,
                    'max_error': 1e-3
                }),
            ]
        })
    ]
"""
pipeline_for_compare_inputs_grad_with_npy_for_case_by_case_config = \
    [MeFacadeFC, LoadFromNpyDC, RunBackwardBlockWrtInputsBC,
     IdCartesianProductFIPC, IdentityBackwardEC,
     IdCartesianProductERPC, LoadFromNpyVC]

"""
Compare params gradient with data stored in npy file. This pipeline is suitable for
case-by-case style config.

Example:
    verification_set = [
        ('MatMul', {
            'block': P.MatMul(),
            'desc_inputs': [
                ('tests/mindspore_test_framework/apps/data/input_0.npy', {
                    'dtype': np.float32
                }),
                ('tests/mindspore_test_framework/apps/data/input_1.npy', {
                    'dtype': np.float32
                }),
            ],
            'desc_bprop': [
                np.ones(shape=(2, 2)).astype(np.float32)
            ],
            'desc_expect': [
                ('tests/mindspore_test_framework/apps/data/grad_0.npy', {
                    'dtype': np.float32,
                    'check_tolerance': True,
                    'relative_tolerance': 1e-2,
                    'absolute_tolerance': 1e-2
                }),
                ('tests/mindspore_test_framework/apps/data/grad_1.npy', {
                    'dtype': np.float32,
                    'max_error': 1e-3
                }),
            ]
        })
    ]
"""
pipeline_for_compare_params_grad_with_npy_for_case_by_case_config = \
    [MeFacadeFC, LoadFromNpyDC, RunBackwardBlockWrtParamsBC,
     IdCartesianProductFIPC, IdentityBackwardEC,
     IdCartesianProductERPC, LoadFromNpyVC]

"""
Compare inputs gradient with result of numerical differentiation. This pipeline is
suitable for case-by-case style config.

Example:
    verification_set = [
        ('Add', {
            'block': (P.Add(), {'reduce_output': False}),
            'desc_inputs': [[1, 3, 3, 4], [1, 3, 3, 4]],
            'desc_bprop': [[1, 3, 3, 4]]
        })
    ]
"""
pipeline_for_compare_inputs_grad_with_numerical_diff_for_case_by_case_config = \
    [MeFacadeFC, GenerateFromShapeDC, IdentityBC,
     IdCartesianProductFIPC,
     CheckGradientWrtInputsEC]

"""
Compare inputs gradient with result of numerical differentiation. This pipeline is
suitable for config in a grouped style.

Example:
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
"""
pipeline_for_compare_inputs_grad_with_numerical_diff_for_group_by_group_config = [GenerateFromShapeDC, IdentityBC,
                                                                                  IdCartesianProductFIPC,
                                                                                  CheckGradientWrtInputsEC]

"""
Compare params gradient with result of numerical differentiation. This pipeline is suitable for
config in a grouped style.

Example:
    verification_set = {
        'inputs': {
            'id': 'BertAttentionSoftmax',
            'group': 'bert',
            'desc_inputs': [
                [128, 1024],
                [1, 16, 128, 128]
            ],
            'desc_bprop': [
                [1, 16, 128, 64],
                [1, 16, 128, 64]
            ]
        },
        'function': [
            {
                'id': 'BertAttentionSoftmax',
                'group': 'BertAttentionSoftmax',
                'block': BertAttentionSoftmax(batch_size=1,
                                              to_tensor_width=1024,
                                              from_seq_length=128,
                                              to_seq_length=128,
                                              num_attention_heads=16,
                                              size_per_head=64,
                                              value_act=None,
                                              attention_probs_dropout_prob=0,
                                              initializer_range=0.02)
            }
        ],
        'ext': {}
    }
"""
pipeline_for_compare_params_grad_with_numerical_diff_for_group_by_group_config = [GenerateFromShapeDC, IdentityBC,
                                                                                  IdCartesianProductFIPC,
                                                                                  CheckGradientWrtParamsEC]

"""
Compare inputs jacobian with result of numerical differentiation. This pipeline is suitable for
config in a case-by-case style.

Example:
    verification_set = [
        ('Add', {
            'block': (P.Add(), {'reduce_output': False}),
            'desc_inputs': [[1, 3, 3, 4], [1, 3, 3, 4]],
            'desc_bprop': [[1, 3, 3, 4]],
            'desc_expect': {
                'compare_with': [
                    (run_np, np.add),
                    (run_user_defined, user_defined.add)
                ],
                'compare_gradient_with': [
                    (run_user_defined_grad, user_defined.add)
                ],
            }
        }),
    ]
"""
pipeline_for_compare_inputs_jacobian_with_numerical_diff_for_case_by_case_config = \
    [MeFacadeFC, GenerateFromShapeDC, IdentityBC,
     IdCartesianProductFIPC,
     CheckJacobianWrtInputsEC]

"""
Compare inputs jacobian with result of numerical differentiation. This pipeline is suitable for
config in a grouped style.

Example:
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
"""
pipeline_for_compare_inputs_jacobian_with_numerical_diff_for_group_by_group_config = [GenerateFromShapeDC, IdentityBC,
                                                                                      IdCartesianProductFIPC,
                                                                                      CheckJacobianWrtInputsEC]
