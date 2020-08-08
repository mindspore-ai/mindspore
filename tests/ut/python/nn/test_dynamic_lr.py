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

from mindspore.nn import dynamic_lr as dr

milestone = [10, 20, 30]
learning_rates = [0.1, 0.05, 0.01]
learning_rate = 0.1
end_learning_rate = 0.01
decay_rate = 0.9
total_step = 30
step_per_epoch = 3
decay_epoch = 2
min_lr = 0.01
max_lr = 0.1
power = 0.5
warmup_epoch = 2

class TestInputs:
    def test_milestone1(self):
        milestone1 = 1
        with pytest.raises(TypeError):
            dr.piecewise_constant_lr(milestone1, learning_rates)

    def test_milestone2(self):
        milestone1 = [20, 10, 1]
        with pytest.raises(ValueError):
            dr.piecewise_constant_lr(milestone1, learning_rates)

        milestone2 = [1.0, 2.0, True]
        with pytest.raises(TypeError):
            dr.piecewise_constant_lr(milestone2, learning_rates)

    def test_learning_rates1(self):
        lr = True
        with pytest.raises(TypeError):
            dr.piecewise_constant_lr(milestone, lr)

    def test_learning_rates2(self):
        lr = [1, 2, 1]
        with pytest.raises(TypeError):
            dr.piecewise_constant_lr(milestone, lr)

    def test_learning_rate_type(self):
        lr = True
        with pytest.raises(TypeError):
            dr.exponential_decay_lr(lr, decay_rate, total_step, step_per_epoch, decay_epoch)

        with pytest.raises(TypeError):
            dr.polynomial_decay_lr(lr, end_learning_rate, total_step, step_per_epoch, decay_epoch, power)

    def test_learning_rate_value(self):
        lr = -1.0
        with pytest.raises(ValueError):
            dr.exponential_decay_lr(lr, decay_rate, total_step, step_per_epoch, decay_epoch)

        with pytest.raises(ValueError):
            dr.polynomial_decay_lr(lr, end_learning_rate, total_step, step_per_epoch, decay_epoch, power)

    def test_end_learning_rate_type(self):
        lr = True
        with pytest.raises(TypeError):
            dr.polynomial_decay_lr(learning_rate, lr, total_step, step_per_epoch, decay_epoch, power)

    def test_end_learning_rate_value(self):
        lr = -1.0
        with pytest.raises(ValueError):
            dr.polynomial_decay_lr(learning_rate, lr, total_step, step_per_epoch, decay_epoch, power)

    def test_decay_rate_type(self):
        rate = 'a'
        with pytest.raises(TypeError):
            dr.exponential_decay_lr(learning_rate, rate, total_step, step_per_epoch, decay_epoch)

    def test_decay_rate_value(self):
        rate = -1.0
        with pytest.raises(ValueError):
            dr.exponential_decay_lr(learning_rate, rate, total_step, step_per_epoch, decay_epoch)

    def test_total_step1(self):
        total_step1 = 2.0
        with pytest.raises(TypeError):
            dr.exponential_decay_lr(learning_rate, decay_rate, total_step1, step_per_epoch, decay_epoch)

        with pytest.raises(TypeError):
            dr.cosine_decay_lr(min_lr, max_lr, total_step1, step_per_epoch, decay_epoch)

        with pytest.raises(TypeError):
            dr.polynomial_decay_lr(learning_rate, end_learning_rate, total_step1, step_per_epoch, decay_epoch, power)

    def test_total_step2(self):
        total_step1 = -1
        with pytest.raises(ValueError):
            dr.exponential_decay_lr(learning_rate, decay_rate, total_step1, step_per_epoch, decay_epoch)

        with pytest.raises(ValueError):
            dr.cosine_decay_lr(min_lr, max_lr, total_step1, step_per_epoch, decay_epoch)

        with pytest.raises(ValueError):
            dr.polynomial_decay_lr(learning_rate, end_learning_rate, total_step1, step_per_epoch, decay_epoch, power)

    def test_step_per_epoch1(self):
        step_per_epoch1 = True
        with pytest.raises(TypeError):
            dr.exponential_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch1, decay_epoch)

        with pytest.raises(TypeError):
            dr.cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch1, decay_epoch)

        with pytest.raises(TypeError):
            dr.polynomial_decay_lr(learning_rate, end_learning_rate, total_step, step_per_epoch1, decay_epoch, power)

    def test_step_per_epoch2(self):
        step_per_epoch1 = -1
        with pytest.raises(ValueError):
            dr.exponential_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch1, decay_epoch)

        with pytest.raises(ValueError):
            dr.cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch1, decay_epoch)

        with pytest.raises(ValueError):
            dr.polynomial_decay_lr(learning_rate, end_learning_rate, total_step, step_per_epoch1, decay_epoch, power)

    def test_decay_epoch1(self):
        decay_epoch1 = 'm'
        with pytest.raises(TypeError):
            dr.exponential_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch1)

        with pytest.raises(TypeError):
            dr.cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch, decay_epoch1)

        with pytest.raises(TypeError):
            dr.polynomial_decay_lr(learning_rate, end_learning_rate, total_step, step_per_epoch, decay_epoch1, power)

    def test_decay_epoch2(self):
        decay_epoch1 = -1
        with pytest.raises(ValueError):
            dr.exponential_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch1)

        with pytest.raises(ValueError):
            dr.cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch, decay_epoch1)

        with pytest.raises(ValueError):
            dr.polynomial_decay_lr(learning_rate, end_learning_rate, total_step, step_per_epoch, decay_epoch1, power)

    def test_is_stair(self):
        is_stair = 1
        with pytest.raises(TypeError):
            dr.exponential_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, is_stair)

    def test_min_lr_type(self):
        min_lr1 = True
        with pytest.raises(TypeError):
            dr.cosine_decay_lr(min_lr1, max_lr, total_step, step_per_epoch, decay_epoch)

    def test_min_lr_value(self):
        min_lr1 = -1.0
        with pytest.raises(ValueError):
            dr.cosine_decay_lr(min_lr1, max_lr, total_step, step_per_epoch, decay_epoch)

    def test_max_lr_type(self):
        max_lr1 = 'a'
        with pytest.raises(TypeError):
            dr.cosine_decay_lr(min_lr, max_lr1, total_step, step_per_epoch, decay_epoch)

    def test_max_lr_value(self):
        max_lr1 = -1.0
        with pytest.raises(ValueError):
            dr.cosine_decay_lr(min_lr, max_lr1, total_step, step_per_epoch, decay_epoch)

    def test_power(self):
        power1 = True
        with pytest.raises(TypeError):
            dr.polynomial_decay_lr(learning_rate, end_learning_rate, total_step, step_per_epoch, decay_epoch, power1)

    def test_update_decay_epoch(self):
        update_decay_epoch = 1
        with pytest.raises(TypeError):
            dr.polynomial_decay_lr(learning_rate, end_learning_rate, total_step, step_per_epoch, decay_epoch,
                                   power, update_decay_epoch)


def test_learning_rate():
    lr = dr.piecewise_constant_lr(milestone, learning_rates)
    assert len(lr) == milestone[-1]


def test_exponential_decay():
    lr1 = dr.exponential_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch)
    assert len(lr1) == total_step

    lr2 = dr.exponential_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, True)
    assert len(lr2) == total_step


def test_enatural_exp_decay():
    lr1 = dr.natural_exp_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch)
    assert len(lr1) == total_step

    lr2 = dr.natural_exp_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, True)
    assert len(lr2) == total_step


def test_inverse_decay():
    lr1 = dr.inverse_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch)
    assert len(lr1) == total_step

    lr2 = dr.inverse_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, True)
    assert len(lr2) == total_step


def test_cosine_decay():
    lr = dr.cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch, decay_epoch)
    assert len(lr) == total_step


def test_polynomial_decay():
    lr1 = dr.polynomial_decay_lr(learning_rate, end_learning_rate, total_step, step_per_epoch, decay_epoch, power)
    assert len(lr1) == total_step
    lr2 = dr.polynomial_decay_lr(learning_rate, end_learning_rate, total_step, step_per_epoch, decay_epoch, power,
                                 True)
    assert len(lr2) == total_step


def test_warmup():
    lr1 = dr.warmup_lr(learning_rate, total_step, step_per_epoch, warmup_epoch)
    assert len(lr1) == total_step
