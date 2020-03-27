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

"""Component that verify shape and type."""

from ...components.icomponent import IVerifierComponent
from ...utils.other_util import to_numpy_list
from ...utils import keyword

class ShapeTypeVC(IVerifierComponent):
    """
    Verify if the result's shape and type are correct.

    Examples:
        'desc_expect': {
            'shape_type': [
                {
                    'type': np.float32,
                    'shape': (2, 2)
                }
            ]
        }
    """
    def verify(self, expect, func_result, verification_set):
        results = to_numpy_list(func_result[keyword.result])
        expects = expect[keyword.desc_expect][keyword.shape_type]
        for i, e in enumerate(expects):
            if results[i].shape != e[keyword.shape] or results[i].dtype != e[keyword.type]:
                raise TypeError(f'Error: expect {expect}, but got {func_result}')
