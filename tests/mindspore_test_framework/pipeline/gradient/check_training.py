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

"""Pipelines for loss checking."""

from ...components.executor.exec_and_verify_model_loss import LossVerifierEC
from ...components.facade.me_facade import MeFacadeFC
from ...components.function.get_function_from_config import IdentityBC
from ...components.function_inputs_policy.cartesian_product_on_id_for_function_inputs import IdCartesianProductFIPC
from ...components.inputs.generate_dataset_for_linear_regression import GenerateDataSetForLRDC

# pylint: disable=W0105
"""
Check if model loss converge to a given bound.

Example:
    verification_set = [
        ('Linreg', {
            'block': {
                'model': network,
                'loss': SquaredLoss(),
                'opt': Lamb(network.trainable_params(), lr=0.02, weight_decay=0.01),
                'num_epochs': num_epochs,
                'loss_upper_bound': 0.3,
            },
            'desc_inputs': {
                'true_params': ([2, -3.4], 4.2),
                'num_samples': 100,
                'batch_size': 20,
            }
        })
    ]
"""
pipeline_for_check_model_loss_for_case_by_case_config = [MeFacadeFC, GenerateDataSetForLRDC,
                                                         IdentityBC, IdCartesianProductFIPC,
                                                         LossVerifierEC]
