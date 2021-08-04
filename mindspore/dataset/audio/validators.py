# Copyright 2021 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
Validators for TensorOps.
"""
from functools import wraps

from mindspore.dataset.core.validator_helpers import check_not_zero, check_int32, check_float32, check_value_normalize_std, parse_user_args, type_check


def check_biquad_sample_rate(sample_rate):
    """Wrapper method to check the parameters of sample_rate."""
    type_check(sample_rate, (int,), "sample_rate")
    check_int32(sample_rate, "sample_rate")
    check_not_zero(sample_rate, "sample_rate")


def check_biquad_central_freq(central_freq):
    """Wrapper method to check the parameters of central_freq."""
    type_check(central_freq, (float, int), "central_freq")
    check_float32(central_freq, "central_freq")


def check_biquad_Q(Q):
    """Wrapper method to check the parameters of Q."""
    type_check(Q, (float, int), "Q")
    check_value_normalize_std(Q, [0, 1], "Q")


def check_biquad_noise(noise):
    """Wrapper method to check the parameters of noise."""
    type_check(noise, (bool,), "noise")


def check_biquad_const_skirt_gain(const_skirt_gain):
    """Wrapper method to check the parameters of const_skirt_gain."""
    type_check(const_skirt_gain, (bool,), "const_skirt_gain")


def check_biquad_gain(gain):
    """Wrapper method to check the parameters of gain."""
    type_check(gain, (float, int), "gain")
    check_float32(gain, "gain")


def check_band_biquad(method):
    """Wrapper method to check the parameters of BandBiquad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, central_freq, Q, noise], _ = parse_user_args(
            method, *args, **kwargs)
        check_biquad_sample_rate(sample_rate)
        check_biquad_central_freq(central_freq)
        check_biquad_Q(Q)
        check_biquad_noise(noise)
        return method(self, *args, **kwargs)

    return new_method


def check_allpass_biquad(method):
    """Wrapper method to check the parameters of CutMixBatch."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, central_freq, Q], _ = parse_user_args(
            method, *args, **kwargs)
        check_biquad_sample_rate(sample_rate)
        check_biquad_central_freq(central_freq)
        check_biquad_Q(Q)
        return method(self, *args, **kwargs)

    return new_method


def check_bandpass_biquad(method):
    """Wrapper method to check the parameters of BandpassBiquad."""

    @ wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, central_freq, Q, const_skirt_gain], _ = parse_user_args(
            method, *args, **kwargs)
        check_biquad_sample_rate(sample_rate)
        check_biquad_central_freq(central_freq)
        check_biquad_Q(Q)
        check_biquad_const_skirt_gain(const_skirt_gain)
        return method(self, *args, **kwargs)

    return new_method


def check_bandreject_biquad(method):
    """Wrapper method to check the parameters of BandrejectBiquad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, central_freq, Q], _ = parse_user_args(
            method, *args, **kwargs)
        check_biquad_sample_rate(sample_rate)
        check_biquad_central_freq(central_freq)
        check_biquad_Q(Q)
        return method(self, *args, **kwargs)

    return new_method


def check_bass_biquad(method):
    """Wrapper method to check the parameters of CutMixBatch."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, gain, central_freq, Q], _ = parse_user_args(
            method, *args, **kwargs)
        check_biquad_sample_rate(sample_rate)
        check_biquad_gain(gain)
        check_biquad_central_freq(central_freq)
        check_biquad_Q(Q)
        return method(self, *args, **kwargs)

    return new_method
