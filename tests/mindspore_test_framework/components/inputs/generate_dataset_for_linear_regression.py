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

"""Component that generate dataset for linear regression."""

from ...components.icomponent import IDataComponent
from ...utils import keyword
from ...utils.dataset_util import generate_dataset_for_linear_regression


class GenerateDataSetForLRDC(IDataComponent):
    """
    Create dataset for linear regression, with salt from normal distribution.

    Examples:
        'inputs': {
            'true_params': ([2, -3.4], 4.2),
            'num_samples': 100,
            'batch_size': 20,
        }
    """

    def __call__(self):
        result = []
        for config in self.verification_set[keyword.inputs]:
            desc_inputs = config[keyword.desc_inputs]
            config[keyword.desc_inputs] = generate_dataset_for_linear_regression(desc_inputs[keyword.true_params][0],
                                                                                 desc_inputs[keyword.true_params][1],
                                                                                 desc_inputs[keyword.num_samples],
                                                                                 desc_inputs[keyword.batch_size])
            result.append(config)
        return result
