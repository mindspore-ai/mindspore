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

"""Pipelines for forward computing."""

from ...components.executor.exec_forward import IdentityEC
from ...components.facade.me_facade import MeFacadeFC
from ...components.function.compile_block import CompileBlockBC
from ...components.function.run_block import RunBlockBC
from ...components.function_inputs_policy.cartesian_product_on_id_for_function_inputs import IdCartesianProductFIPC
from ...components.inputs.generate_inputs_from_shape import GenerateFromShapeDC

# pylint: disable=W0105
"""
Test if compile forward anf graph is ok.
The pipeline just try to run to compile forward anf graph process without comparing any result with any expect.
The pipeline is suitable for config in a case-by-case style.

Example:
    Examples:
    verification_set = [
        ('Add', {
            'block': (P.Add(), {'reduce_output': False}),
            'desc_inputs': [[1, 3, 3, 4], [1, 3, 3, 4]],
            'desc_bprop': [[1, 3, 3, 4]],
        })
    ]
"""
pipeline_for_compile_forward_anf_graph_for_case_by_case_config = [MeFacadeFC, GenerateFromShapeDC, CompileBlockBC,
                                                                  IdCartesianProductFIPC, IdentityEC]

"""
Test if compile forward ge graph is ok.
The pipeline will try to run through compiling forward ge graph process without comparing any result with any expect.
The pipeline is suitable for config in a case-by-case style.

Example:
    Examples:
    verification_set = [
        ('Add', {
            'block': (P.Add(), {'reduce_output': False}),
            'desc_inputs': [[1, 3, 3, 4], [1, 3, 3, 4]],
            'desc_bprop': [[1, 3, 3, 4]],
        })
    ]
"""
pipeline_for_compile_forward_ge_graph_for_case_by_case_config = [MeFacadeFC, GenerateFromShapeDC, RunBlockBC,
                                                                 IdCartesianProductFIPC, IdentityEC]

pipeline_for_compile_forward_ge_graph_for_case_by_case_config_exception = [MeFacadeFC, GenerateFromShapeDC, RunBlockBC,
                                                                           IdCartesianProductFIPC]
