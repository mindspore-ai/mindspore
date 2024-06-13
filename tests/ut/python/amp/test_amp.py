# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore._c_expression import amp as amp_c
from mindspore._c_expression.amp import pop_amp_strategy, push_amp_strategy, create_amp_strategy, \
    get_curr_amp_strategy, AmpStrategy, AmpLevel, PrimCastStrategy, PrimCastStrategyInfo


def test_create_amp_strategy():
    """
    Feature: Test create amp strategy.
    Description: Create an amp strategy, check vars in the amp strategy.
    Expectation: Success.
    """
    white_list = [ops.Abs, ops.Conv2D, ops.Conv3D]
    white_list = [(prim.__name__, []) for prim in white_list if issubclass(prim, ops.Primitive)]
    black_list = [ops.LayerNorm, ops.BatchNorm]
    black_list = [(prim.__name__, []) for prim in black_list if issubclass(prim, ops.Primitive)]
    amp_strategy = create_amp_strategy(AmpLevel.AmpAuto, mstype.float16, white_list, black_list)
    assert isinstance(amp_strategy, AmpStrategy)
    assert amp_strategy.get_amp_level() == AmpLevel.AmpAuto
    assert amp_strategy.get_amp_dtype() == mstype.float16
    assert amp_strategy.get_white_list() == white_list
    assert amp_strategy.get_black_list() == black_list


def test_push_pop_amp_strategy():
    """
    Feature: Test push and pop amp strategy.
    Description: Push and pop amp strategy, check top amp strategy in the stack.
    Expectation: Success.
    """
    # amp strategy stack should be empty now
    curr_amp_strategy = get_curr_amp_strategy()
    assert curr_amp_strategy is None
    # push first amp strategy into stack
    first_white_list = [ops.Abs, ops.Conv2D, ops.Conv3D]
    first_white_list = [(prim.__name__, []) for prim in first_white_list if issubclass(prim, ops.Primitive)]
    first_black_list = [ops.LayerNorm, ops.BatchNorm]
    first_black_list = [(prim.__name__, []) for prim in first_black_list if issubclass(prim, ops.Primitive)]
    push_amp_strategy(AmpLevel.AmpAuto, mstype.float16, first_white_list, first_black_list)
    # check top amp strategy in the stack
    curr_amp_strategy = get_curr_amp_strategy()
    assert isinstance(curr_amp_strategy, AmpStrategy)
    assert curr_amp_strategy.get_amp_level() == AmpLevel.AmpAuto
    assert curr_amp_strategy.get_amp_dtype() == mstype.float16
    assert curr_amp_strategy.get_white_list() == first_white_list
    assert curr_amp_strategy.get_black_list() == first_black_list
    # push second amp strategy into stack
    second_white_list = [ops.ReLU, ops.Sin, ops.Tanh]
    second_white_list = [(prim.__name__, []) for prim in second_white_list if issubclass(prim, ops.Primitive)]
    second_black_list = [ops.Cos, ops.Sigmoid]
    second_black_list = [(prim.__name__, []) for prim in second_black_list if issubclass(prim, ops.Primitive)]
    push_amp_strategy(AmpLevel.AmpO1, mstype.bfloat16, second_white_list, second_black_list)
    # check top amp strategy in the stack
    curr_amp_strategy = get_curr_amp_strategy()
    assert isinstance(curr_amp_strategy, AmpStrategy)
    assert curr_amp_strategy.get_amp_level() == AmpLevel.AmpO1
    assert curr_amp_strategy.get_amp_dtype() == mstype.bfloat16
    assert curr_amp_strategy.get_white_list() == second_white_list
    assert curr_amp_strategy.get_black_list() == second_black_list
    # pop one amp strategy from stack
    pop_amp_strategy()
    # check top amp strategy in the stack
    curr_amp_strategy = get_curr_amp_strategy()
    assert isinstance(curr_amp_strategy, AmpStrategy)
    assert curr_amp_strategy.get_amp_level() == AmpLevel.AmpAuto
    assert curr_amp_strategy.get_amp_dtype() == mstype.float16
    assert curr_amp_strategy.get_white_list() == first_white_list
    assert curr_amp_strategy.get_black_list() == first_black_list
    # pop one amp strategy from stack
    pop_amp_strategy()
    # check top amp strategy in the stack, should be empty now
    curr_amp_strategy = get_curr_amp_strategy()
    assert curr_amp_strategy is None
    # pop again, should raise warning
    pop_amp_strategy()
    # check again, should be empty now
    curr_amp_strategy = get_curr_amp_strategy()
    assert curr_amp_strategy is None


def test_check_prim_cast_strategy():
    """
    Feature: Test check prim cast strategy.
    Description: Create an amp strategy, and then check the prim cast strategy under this amp strategy.
    Expectation: Success.
    """
    white_list = [("Abs", [0]), ("Conv2D", [0, 1])]
    black_list = [("LayerNorm", [0]), ("BatchNorm", [0])]
    amp_strategy = create_amp_strategy(AmpLevel.AmpAuto, mstype.float16, white_list, black_list)
    assert isinstance(amp_strategy, AmpStrategy)
    # test white list
    prim_strategy_info = amp_strategy.get_prim_cast_strategy_info("Conv2D")
    assert isinstance(prim_strategy_info, PrimCastStrategyInfo)
    assert prim_strategy_info.strategy == PrimCastStrategy.AmpDoCast
    assert prim_strategy_info.dtype == mstype.float16
    assert prim_strategy_info.arg_pos == [0, 1]
    # test black list
    prim_strategy_info = amp_strategy.get_prim_cast_strategy_info("LayerNorm")
    assert isinstance(prim_strategy_info, PrimCastStrategyInfo)
    assert prim_strategy_info.strategy == PrimCastStrategy.AmpDoCast
    assert prim_strategy_info.dtype == mstype.float32
    assert prim_strategy_info.arg_pos == [0]
    # test out of list
    prim_strategy_info = amp_strategy.get_prim_cast_strategy_info("Pad")
    assert isinstance(prim_strategy_info, PrimCastStrategyInfo)
    assert prim_strategy_info.strategy == PrimCastStrategy.AmpIgnore


def test_modify_amp_list():
    """
    Feature: Test modify amp operator lists.
    Description: Check amp operator lists, modify lists and check again.
    Expectation: Success.
    """
    assert isinstance(amp_c.SetDtypeOptList, list)
    assert isinstance(amp_c.SetDtypeList, list)
    assert isinstance(amp_c.AutoPromoteList, list)
    assert amp_c.SetDtypeOptList == [("ProdExt", []), ("SumExt", [])]
    assert not amp_c.SetDtypeList
    assert amp_c.AutoPromoteList == [("Addcdiv", []), ("Addcmul", []), ("Cross", []), ("Dot", []),
                                     ("GridSampler2D", []), ("GridSampler3D", []), ("IndexPut", [])]
    amp_c.SetDtypeOptList.append(("LogSoftmax", [0]))
    amp_c.SetDtypeList.append(("NormExt", [0]))
    amp_c.AutoPromoteList.remove(("Addcmul", []))
    assert amp_c.SetDtypeOptList == [("ProdExt", []), ("SumExt", []), ("LogSoftmax", [0])]
    assert amp_c.SetDtypeList == [("NormExt", [0])]
    assert amp_c.AutoPromoteList == [("Addcdiv", []), ("Cross", []), ("Dot", []), ("GridSampler2D", []),
                                     ("GridSampler3D", []), ("IndexPut", [])]
