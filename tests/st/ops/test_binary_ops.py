# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Tests for binary operation.
"""
import pytest
from tests.mark_utils import arg_mark
from tests.st.ops.share._internal.binary_ops import BinaryOpsFactory
from tests.st.ops.share._op_info.op_database import get_op_info, binary_op_db, binary_op_kbk_db, binary_op_error_db

ops_params_dict = {
    'test_binary_op_reference_forward' : (['pynative'], binary_op_db),
    'test_binary_op_reference_backward' : (['pynative'], binary_op_db),
    'test_binary_op_type_promotion': (['pynative'], ['mint.add', 'mint.sub']),
    'test_binary_op_reference_forward_kbk' : (['kbk'], binary_op_kbk_db),
    'test_binary_op_reference_backward_kbk' : (['kbk'], binary_op_kbk_db),
    'test_binary_op_dynamic_forward': (['kbk'], binary_op_kbk_db),
    'test_binary_op_dynamic_backward': (['kbk'], binary_op_kbk_db),
    'test_binary_op_error': (['pynative', 'kbk'], binary_op_error_db),
}

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ops_params_dict['test_binary_op_reference_forward'][0])
@pytest.mark.parametrize("op_info", ops_params_dict['test_binary_op_reference_forward'][1])
def test_binary_op_reference_forward(mode, op_info):
    '''
    Feature: Binary operations
    Description: Compare forward.
    Expectation: MindSpore matches Benchmark for outputs.
    '''
    fact = BinaryOpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_reference()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ops_params_dict['test_binary_op_reference_backward'][0])
@pytest.mark.parametrize("op_info", ops_params_dict['test_binary_op_reference_backward'][1])
def test_binary_op_reference_backward(mode, op_info):
    '''
    Feature: Binary operations
    Description: Compare gradients.
    Expectation: MindSpore matches Benchmark for gradients.
    '''
    fact = BinaryOpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_reference(grad_cmp=True)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ops_params_dict['test_binary_op_reference_forward_kbk'][0])
@pytest.mark.parametrize("op_info", ops_params_dict['test_binary_op_reference_forward_kbk'][1])
def test_binary_op_reference_forward_kbk(mode, op_info):
    '''
    Feature: Binary operations
    Description: Compare forward.
    Expectation: MindSpore matches Benchmark for outputs.
    '''
    fact = BinaryOpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_reference()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ops_params_dict['test_binary_op_reference_backward_kbk'][0])
@pytest.mark.parametrize("op_info", ops_params_dict['test_binary_op_reference_backward_kbk'][1])
def test_binary_op_reference_backward_kbk(mode, op_info):
    '''
    Feature: Binary operations
    Description: Compare gradients.
    Expectation: MindSpore matches Benchmark for gradients.
    '''
    fact = BinaryOpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_reference(grad_cmp=True)



@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ops_params_dict['test_binary_op_type_promotion'][0])
@pytest.mark.parametrize("op_info", ops_params_dict['test_binary_op_type_promotion'][1])
def test_binary_op_type_promotion(mode, op_info):

    '''
    Feature: Binary operations
    Description: Compare tensor type promotion.
    Expectation: MindSpore matches Benchmark for outputs.
    '''
    fact = BinaryOpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_binary_op_tensor_type_promotion()
    fact.test_binary_op_scalar_type_promotion()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ops_params_dict['test_binary_op_dynamic_forward'][0])
@pytest.mark.parametrize("op_info", ops_params_dict['test_binary_op_dynamic_forward'][1])
def test_binary_op_dynamic_forward(mode, op_info):
    '''
    Feature: Binary operations
    Description: Dynamic shape with tensor inputs, keeping alpha as mutable.
    Expectation: Outputs match Benchmark.
    '''
    fact = BinaryOpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_dynamic(only_dynamic_shape=True)
    fact.test_op_dynamic(only_dynamic_rank=True)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ops_params_dict['test_binary_op_dynamic_backward'][0])
@pytest.mark.parametrize("op_info", ops_params_dict['test_binary_op_dynamic_backward'][1])
def test_binary_op_dynamic_backward(mode, op_info):
    '''
    Feature: Binary operations
    Description: Dynamic shape with tensor inputs, keeping alpha as mutable
    Expectation: Gradients match Benchmark.
    '''
    fact = BinaryOpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_dynamic(only_dynamic_shape=True, grad_cmp=True)
    fact.test_op_dynamic(only_dynamic_rank=True, grad_cmp=True)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ops_params_dict['test_binary_op_error'][0])
@pytest.mark.parametrize("op_info", ops_params_dict['test_binary_op_error'][1])
def test_binary_op_error(mode, op_info):
    '''
    Feature: Binary operations
    Description: Test binary op error cases.
    Expectation: Run success without error.
    '''
    fact = BinaryOpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_binary_op_error()
