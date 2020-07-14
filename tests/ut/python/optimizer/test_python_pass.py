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
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.common.python_pass_register import registe_pass, PyPassManager
from mindspore.common.api import _generate_pip_args
from mindspore._c_expression import generate_key, Executor_

context.set_context(mode=context.GRAPH_MODE)

def get_func_graph(obj, *args, phase="predict"):
    args_names, args_list = _generate_pip_args(obj, *args)
    dic = dict(zip(args_names, args_list))
    key = generate_key(phase, dic)
    phase_prefix = str(key[1])
    if phase == 'export':
        phase = phase + '.' + phase_prefix + '.' + str(obj.create_time)
    else:
        phase = phase_prefix + phase + '.' + str(obj.create_time)
    _executor = Executor_.get_instance()
    _executor.compile(obj, args_list, phase, False)
    return _executor.get_func_graph(phase)

def test_softmax_relu():
    """
    Use python pass to transform from Softmax to ReLU.
    """
    inputs = Tensor(np.ones([42]), mindspore.float16)
    softmax_model = nn.Softmax()

    @registe_pass(run_only_once=True)
    def softmax_relu_pass():
        softmax = P.Softmax()
        relu = P.ReLU()
        def pattern(x):
            x = softmax(x)
            return x
        def target(x):
            x = relu(x)
            return x
        return pattern, target

    transformed_repr = get_func_graph(softmax_model, inputs).get_return().expanded_str(2)
    ppm = PyPassManager()
    ppm.unregiste(softmax_relu_pass)
    assert "ReLU" in transformed_repr
    assert "Softmax" not in transformed_repr
