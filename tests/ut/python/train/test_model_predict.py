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
""" test model predict """
import pytest

from mindspore import nn, Tensor
from mindspore.train import Model


class TinyNet(nn.Cell):
    """ Net definition """

    def __init__(self):
        super(TinyNet, self).__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        return x


def test_model_predict_backend_input_error():
    """
    Feature: Model predict
    Description: Input wrong backend value.
    Expectation: Raise value error.
    """
    model = Model(TinyNet())
    input_tensor = Tensor(1)
    with pytest.raises(ValueError):
        model.predict(input_tensor, backend='gpu')

    with pytest.raises(ValueError):
        model.predict(input_tensor, backend=0)
