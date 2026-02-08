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
Comparison base module.
Provides base classes and utility functions for comparing different test results.
"""

import copy
import numpy as np
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net
from ._utils import _count_unequal_element


class CompareBase:
    def __init__(self):
        pass


    def compare_nparray(self, data_expected, data_me, rtol, atol, equal_nan=True):
        if np.any(np.isnan(data_expected)):
            assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
        elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
            _count_unequal_element(data_expected, data_me, rtol, atol)
        else:
            assert np.array(data_expected).shape == np.array(data_me).shape


    def compare_checkpoint(self, expected_net, ckpt_expected, ckpt_actual, inputdata, rtol, atol):
        data_expected = load_checkpoint(ckpt_expected)
        data_actual = load_checkpoint(ckpt_actual)
        actual_net = copy.deepcopy(expected_net)
        load_param_into_net(expected_net, data_expected)
        expected_out = expected_net(inputdata)
        load_param_into_net(actual_net, data_actual)
        actual_out = actual_net(inputdata)
        self.compare_nparray(expected_out.asnumpy(), actual_out.asnumpy(), rtol, atol)


    def compare_checkpoint_dict(self, expected_net, ckpt_expected, ckpt_actual, *inputs,
                                rtol=0.001, atol=0.001):
        actual_net = copy.deepcopy(expected_net)
        load_param_into_net(expected_net, ckpt_expected)
        expected_out = expected_net(*inputs)
        load_param_into_net(actual_net, ckpt_actual)
        actual_out = actual_net(*inputs)
        self.compare_nparray(expected_out.asnumpy(), actual_out.asnumpy(), rtol, atol)


comparebase = CompareBase()
