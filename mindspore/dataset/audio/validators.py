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

from mindspore.dataset.core.validator_helpers import check_float32, check_int32_not_zero, \
    check_non_negative_float32, check_non_negative_float64, check_pos_float32, check_pos_int64, check_value, \
    parse_user_args, type_check
from .utils import ScaleType


def check_amplitude_to_db(method):
    """Wrapper method to check the parameters of AmplitudeToDB."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [stype, ref_value, amin, top_db], _ = parse_user_args(method, *args, **kwargs)

        # type check stype
        type_check(stype, (ScaleType,), "stype")

        # type check ref_value
        type_check(ref_value, (int, float), "ref_value")
        # value check ref_value
        if ref_value is not None:
            check_pos_float32(ref_value, "ref_value")

        # type check amin
        type_check(amin, (int, float), "amin")
        # value check amin
        if amin is not None:
            check_pos_float32(amin, "amin")

        # type check top_db
        type_check(top_db, (int, float), "top_db")
        # value check top_db
        if top_db is not None:
            check_pos_float32(top_db, "top_db")

        return method(self, *args, **kwargs)

    return new_method


def check_biquad_sample_rate(sample_rate):
    """Wrapper method to check the parameters of sample_rate."""
    type_check(sample_rate, (int,), "sample_rate")
    check_int32_not_zero(sample_rate, "sample_rate")


def check_biquad_central_freq(central_freq):
    """Wrapper method to check the parameters of central_freq."""
    type_check(central_freq, (float, int), "central_freq")
    check_float32(central_freq, "central_freq")


def check_biquad_Q(Q):
    """Wrapper method to check the parameters of Q."""
    type_check(Q, (float, int), "Q")
    check_value(Q, [0, 1], "Q", True)


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
    """Wrapper method to check the parameters of AllpassBiquad."""

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

    @wraps(method)
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
    """Wrapper method to check the parameters of BassBiquad."""

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


def check_time_stretch(method):
    """Wrapper method to check the parameters of TimeStretch."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [hop_length, n_freq, fixed_rate], _ = parse_user_args(method, *args, **kwargs)

        if hop_length is not None:
            type_check(hop_length, (int,), "hop_length")
            check_pos_int64(hop_length, "hop_length")

        type_check(n_freq, (int,), "n_freq")
        check_pos_int64(n_freq, "n_freq")

        if fixed_rate is not None:
            type_check(fixed_rate, (int, float), "fixed_rate")
            check_pos_float32(fixed_rate, "fixed_rate")
        return method(self, *args, **kwargs)

    return new_method


def check_masking(method):
    """Wrapper method to check the parameters of time_masking and FrequencyMasking"""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [iid_masks, mask_param, mask_start, mask_value], _ = parse_user_args(
            method, *args, **kwargs)
        type_check(iid_masks, (bool,), "iid_masks")
        type_check(mask_param, (int,), "mask_param")
        check_non_negative_float32(mask_param, "mask_param")
        type_check(mask_start, (int,), "mask_start")
        check_non_negative_float32(mask_start, "mask_start")
        type_check(mask_value, (int, float), "mask_value")
        check_non_negative_float64(mask_value, "mask_value")
        return method(self, *args, **kwargs)

    return new_method


def check_complex_norm(method):
    """Wrapper method to check the parameters of ComplexNorm."""
    @wraps(method)
    def new_method(self, *args, **kwargs):
        [power], _ = parse_user_args(method, *args, **kwargs)
        check_non_negative_float32(power, "power")
        return method(self, *args, **kwargs)

    return new_method
