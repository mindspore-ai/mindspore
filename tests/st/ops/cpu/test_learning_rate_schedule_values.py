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
"""CPU execution tests for learning rate schedules."""
import numpy as np
import pytest

from mindspore import Tensor, nn
import mindspore.common.dtype as mstype
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="onecard")
@pytest.mark.parametrize("name, args, expected", [
    ("ExponentialDecayLR", (1.0, 0.5, 2), [1.0, 0.5, 0.25]),
    ("InverseDecayLR", (1.0, 0.5, 2), [1.0, 2 / 3, 0.5]),
    ("NaturalExpDecayLR", (1.0, 0.5, 2), [1.0, np.exp(-1.0), np.exp(-2.0)]),
    ("PolynomialDecayLR", (1.0, 0.25, 2, 1.0), [1.0, 0.25, 0.25]),
    ("WarmUpLR", (1.0, 2), [0.0, 1.0, 1.0]),
])
def test_schedule_values(name, args, expected):
    """
    Feature: Learning rate schedules.
    Description: Execute the schedule at steps 0, 2 and 4 on CPU.
    Expectation: The returned learning rates match the expected curve.
    """
    schedule = getattr(nn.learning_rate_schedule, name)(*args)
    values = [schedule(Tensor(step, mstype.int32)).asnumpy().item() for step in (0, 2, 4)]
    np.testing.assert_allclose(values, expected, rtol=1e-6)
