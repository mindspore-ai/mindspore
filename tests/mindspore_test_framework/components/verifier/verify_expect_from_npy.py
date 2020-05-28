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

"""Component that comparing results with expectation serialized as npy file."""

import numpy as np

from ...components.icomponent import IVerifierComponent
from ...utils import keyword
from ...utils.npy_util import load_data_from_npy_or_shape
from ...utils.other_util import to_numpy_list, to_numpy
from ...utils.verifier_util import tolerance_assert


class LoadFromNpyVC(IVerifierComponent):
    """
    Verify if the results are like expects from npy data, expects could be shape/tensor/np.ndarray/file path.

    Examples:
        'desc_expect': [
            'apps/bert_data/bert_encoder_Reshape_1_output_0.npy',
        ]
        'desc_expect': [
            ('apps/bert_data/bert_encoder_Reshape_1_output_0.npy', np.float32, 1e-3), # (path, dtype, max_error)
        ]
        'desc_expect': [
            ([2, 2], np.float32, 6, 1e-3) # (shape, dtype, scale, max_error)
        ]
    """

    def __call__(self):
        dpaths = self.expect.get(keyword.desc_expect)
        expects = load_data_from_npy_or_shape(dpaths, False)
        results = self.func_result[keyword.result]
        if results:
            results = to_numpy_list(results)
            for i, e in enumerate(expects):
                expect, max_error, check_tolerance, relative_tolerance, absolute_tolerance = e
                expect = to_numpy(expect)
                if check_tolerance:
                    tolerance_assert(expect, results[i], relative_tolerance, absolute_tolerance)
                else:
                    if np.fabs(expect - results[i]).max() > max_error:
                        raise ValueError(f'Error: expect {i}th result {expect}, '
                                         f'but got {results[i]}, max_error {max_error}')
                    print(f'expect {i}th result {expect}, got: {results[i]}, max_error {max_error}')
