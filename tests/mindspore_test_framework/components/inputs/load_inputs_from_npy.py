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

"""Component that load inputs from npy file."""

from ...components.icomponent import IDataComponent
from ...utils import keyword
from ...utils.npy_util import load_data_from_npy_or_shape


class LoadFromNpyDC(IDataComponent):
    """
    Load inputs from npy data, inputs could be shape/tensor/np.ndarray/file path.

    Examples:
        'desc_inputs': [
            'apps/bert_data/bert_encoder_Reshape_1_output_0.npy',
        ]
        'desc_inputs': [
            ('apps/bert_data/bert_encoder_Reshape_1_output_0.npy', np.float32), # (path, dtype)
        ]
        'desc_inputs': [
            ([2, 2], np.float32, 6) # (shape, dtype, scale)
        ]
        'desc_bprop': [
            'apps/bert_data/bert_encoder_Reshape_1_output_0.npy',
        ]
        'desc_bprop': [
            ('apps/bert_data/bert_encoder_Reshape_1_output_0.npy', np.float32),
        ]
        'desc_bprop': [
            ([2, 2], np.float32, 6)
        ]
    """

    def __call__(self):
        result = []
        for config in self.verification_set[keyword.inputs]:
            config[keyword.desc_inputs] = load_data_from_npy_or_shape(config[keyword.desc_inputs])
            config[keyword.desc_bprop] = load_data_from_npy_or_shape(config.get(keyword.desc_bprop, []))
            result.append(config)
        return result
