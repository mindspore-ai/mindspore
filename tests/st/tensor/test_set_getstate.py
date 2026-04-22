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
import pickle
from mindspore import Tensor
from mindspore._c_expression import TensorPy as Tensor_
import numpy as np

from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_set_getstate_bypickle():
    """
    Feature: TensorPy.__setstate__ TensorPy.__getstate__
    Description: Verify the result of __setstate__ && __getstate__
    Expectation: success
    """
    x = Tensor(np.ones((3, 3)))
    x = Tensor_(x)
    serialized_data = pickle.dumps(x)
    deserialized_obj = pickle.loads(serialized_data)
    assert np.allclose(deserialized_obj.asnumpy(), np.ones((3, 3)))
