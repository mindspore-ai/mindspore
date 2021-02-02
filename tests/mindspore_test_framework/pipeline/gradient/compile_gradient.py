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

"""Pipelines for gradients."""

from ...components.executor.exec_gradient import IdentityBackwardEC
from ...components.facade.me_facade import MeFacadeFC
from ...components.function.compile_gradient_wrt_inputs import CompileBackwardBlockWrtInputsBC
from ...components.function.run_gradient_wrt_inputs import RunBackwardBlockWrtInputsBC
from ...components.function_inputs_policy.cartesian_product_on_id_for_function_inputs import IdCartesianProductFIPC
from ...components.inputs.generate_inputs_from_shape import GenerateFromShapeDC

# pylint: disable=W0105
"""
Check if compiling gradient anf graph is ok. This pipeline is suitable for case-by-case style config.

Example:
    verification_set = [
        ('Add', {
            'block': (P.Add(), {'reduce_output': False}),
            'desc_inputs': [[1, 3, 3, 4], [1, 3, 3, 4]],
            'desc_bprop': [[1, 3, 3, 4]]
        })
    ]
"""
pipeline_for_compile_grad_anf_graph_for_case_by_case_config = \
    [MeFacadeFC, GenerateFromShapeDC, CompileBackwardBlockWrtInputsBC,
     IdCartesianProductFIPC, IdentityBackwardEC]

"""
Check if compiling gradient ge graph is ok. This pipeline is suitable for case-by-case style config.

Example:
    verification_set = [
        ('Add', {
            'block': (P.Add(), {'reduce_output': False}),
            'desc_inputs': [[1, 3, 3, 4], [1, 3, 3, 4]],
            'desc_bprop': [[1, 3, 3, 4]]
        })
    ]
"""
pipeline_for_compile_grad_ge_graph_for_case_by_case_config = \
    [MeFacadeFC, GenerateFromShapeDC, RunBackwardBlockWrtInputsBC,
     IdCartesianProductFIPC, IdentityBackwardEC]
