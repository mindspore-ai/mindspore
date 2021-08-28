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
""" Test Dynamic Learning Rate """
import pytest

from mindspore import Tensor
from mindspore.nn import learning_rate_schedule as lr_schedules
from mindspore.common.api import _cell_graph_executor
import mindspore.common.dtype as mstype


learning_rate = 0.1
end_learning_rate = 0.01
decay_rate = 0.9
decay_steps = 4
warmup_steps = 2
min_lr = 0.01
max_lr = 0.1
power = 0.5
global_step = Tensor(2, mstype.int32)


class TestInit:
    def test_learning_rate_type(self):
        lr = True
        with pytest.raises(TypeError):
            lr_schedules.ExponentialDecayLR(lr, decay_rate, decay_steps)

        with pytest.raises(TypeError):
            lr_schedules.PolynomialDecayLR(lr, end_learning_rate, decay_steps, power)

    def test_learning_rate_value(self):
        lr = -1.0
        with pytest.raises(ValueError):
            lr_schedules.ExponentialDecayLR(lr, decay_rate, decay_steps)

        with pytest.raises(ValueError):
            lr_schedules.PolynomialDecayLR(lr, end_learning_rate, decay_steps, power)

    def test_end_learning_rate_type(self):
        lr = True
        with pytest.raises(TypeError):
            lr_schedules.PolynomialDecayLR(learning_rate, lr, decay_steps, power)

    def test_end_learning_rate_value(self):
        lr = -1.0
        with pytest.raises(ValueError):
            lr_schedules.PolynomialDecayLR(learning_rate, lr, decay_steps, power)

    def test_decay_rate_type(self):
        rate = 'a'
        with pytest.raises(TypeError):
            lr_schedules.ExponentialDecayLR(learning_rate, rate, decay_steps)

    def test_decay_rate_value(self):
        rate = -1.0
        with pytest.raises(ValueError):
            lr_schedules.ExponentialDecayLR(learning_rate, rate, decay_steps)

    def test_decay_steps_type(self):
        decay_steps_e = 'm'
        with pytest.raises(TypeError):
            lr_schedules.ExponentialDecayLR(learning_rate, decay_rate, decay_steps_e)

        with pytest.raises(TypeError):
            lr_schedules.CosineDecayLR(min_lr, max_lr, decay_steps_e)

        with pytest.raises(TypeError):
            lr_schedules.PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps_e, power)

    def test_decay_steps_value(self):
        decay_steps_e = -2
        with pytest.raises(ValueError):
            lr_schedules.ExponentialDecayLR(learning_rate, decay_rate, decay_steps_e)

        with pytest.raises(ValueError):
            lr_schedules.CosineDecayLR(min_lr, max_lr, decay_steps_e)

        with pytest.raises(ValueError):
            lr_schedules.PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps_e, power)

    def test_is_stair(self):
        is_stair = 1
        with pytest.raises(TypeError):
            lr_schedules.ExponentialDecayLR(learning_rate, decay_rate, decay_steps, is_stair)

    def test_min_lr_type(self):
        min_lr1 = True
        with pytest.raises(TypeError):
            lr_schedules.CosineDecayLR(min_lr1, max_lr, decay_steps)

    def test_min_lr_value(self):
        min_lr1 = -1.0
        with pytest.raises(ValueError):
            lr_schedules.CosineDecayLR(min_lr1, max_lr, decay_steps)

    def test_max_lr_type(self):
        max_lr1 = 'a'
        with pytest.raises(TypeError):
            lr_schedules.CosineDecayLR(min_lr, max_lr1, decay_steps)

    def test_max_lr_value(self):
        max_lr1 = -1.0
        with pytest.raises(ValueError):
            lr_schedules.CosineDecayLR(min_lr, max_lr1, decay_steps)

    def test_power(self):
        power1 = True
        with pytest.raises(TypeError):
            lr_schedules.PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power1)


def test_exponential_decay():
    lr_schedule = lr_schedules.ExponentialDecayLR(learning_rate, decay_rate, decay_steps, True)
    _cell_graph_executor.compile(lr_schedule, global_step)


def test_enatural_exp_decay():
    lr_schedule = lr_schedules.NaturalExpDecayLR(learning_rate, decay_rate, decay_steps, True)
    _cell_graph_executor.compile(lr_schedule, global_step)


def test_inverse_decay():
    lr_schedule = lr_schedules.InverseDecayLR(learning_rate, decay_rate, decay_steps, True)
    _cell_graph_executor.compile(lr_schedule, global_step)


def test_cosine_decay():
    lr_schedule = lr_schedules.CosineDecayLR(min_lr, max_lr, decay_steps)
    _cell_graph_executor.compile(lr_schedule, global_step)


def test_polynomial_decay():
    lr_schedule = lr_schedules.PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
    _cell_graph_executor.compile(lr_schedule, global_step)


def test_polynomial_decay2():
    lr_schedule = lr_schedules.PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power, True)
    _cell_graph_executor.compile(lr_schedule, global_step)


def test_warmup():
    lr_schedule = lr_schedules.WarmUpLR(learning_rate, warmup_steps)
    _cell_graph_executor.compile(lr_schedule, global_step)
