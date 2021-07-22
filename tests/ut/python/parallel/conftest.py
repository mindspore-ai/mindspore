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
import pytest

import mindspore.context as context
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.parallel._cost_model_context import reset_cost_model_context
from mindspore.parallel._utils import _reset_op_id
from mindspore.parallel.algo_parameter_config import reset_algo_parameters


@pytest.fixture(scope="function", autouse="True")
def reset_test_context():
    context.reset_auto_parallel_context()
    auto_parallel_context().set_enable_all_reduce_fusion(enable_all_reduce_fusion=True)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    reset_cost_model_context()
    reset_algo_parameters()
    _reset_op_id()
