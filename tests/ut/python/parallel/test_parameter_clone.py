# Copyright 2022 Huawei Technologies Co., Ltd
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

from mindspore import Tensor
from mindspore import Parameter
from mindspore.common.initializer import Normal
import mindspore as ms


def test_parameter_clone():
    """
    Feature: test parameter clone api
    Description: assert data and repr
    Expectation: success
    """
    tensor = Tensor(input_data=None, shape=(16, 32), dtype=ms.float32, init=Normal())
    param = Parameter(tensor, requires_grad=False)
    param2 = param.clone()

    data1 = param.asnumpy()
    data2 = param2.asnumpy()
    repr1 = repr(param2)
    assert (data1 == data2).all()
    assert "requires_grad=False" in repr1
    assert "shape=(16, 32)" in repr1
    param3 = param2.clone()
    data3 = param3.asnumpy()
    repr2 = repr(param3)
    assert (data1 == data3).all()
    assert repr1 == repr2
