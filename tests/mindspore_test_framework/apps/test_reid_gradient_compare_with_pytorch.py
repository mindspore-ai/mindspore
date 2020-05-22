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

"""Compare with expectations loaded from npy file"""

import numpy as np

from mindspore import context
from mindspore.ops import operations as P
from ..mindspore_test import mindspore_test
from ..pipeline.gradient.compare_gradient import pipeline_for_compare_inputs_grad_with_npy_for_case_by_case_config

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


@mindspore_test(pipeline_for_compare_inputs_grad_with_npy_for_case_by_case_config)
def test_reid_check_gradient():
    context.set_context(mode=context.PYNATIVE_MODE)
    return verification_set
