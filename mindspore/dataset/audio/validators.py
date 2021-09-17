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

from mindspore.dataset.core.validator_helpers import check_float32, check_float32_not_zero, check_int32_not_zero, \
    check_list_same_size, check_non_negative_float32, check_non_negative_int32, check_pos_float32, check_pos_int32, \
    check_value, parse_user_args, type_check
from .utils import FadeShape, GainType, ScaleType


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


def check_biquad_cutoff_freq(cutoff_freq):
    """Wrapper method to check the parameters of cutoff_freq."""
    type_check(cutoff_freq, (float, int), "cutoff_freq")
    check_float32(cutoff_freq, "cutoff_freq")


def check_highpass_biquad(method):
    """Wrapper method to check the parameters of HighpassBiquad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, cutoff_freq, Q], _ = parse_user_args(method, *args, **kwargs)
        check_biquad_sample_rate(sample_rate)
        check_biquad_cutoff_freq(cutoff_freq)
        check_biquad_Q(Q)
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


def check_contrast(method):
    """Wrapper method to check the parameters of Contrast."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [enhancement_amount], _ = parse_user_args(method, *args, **kwargs)
        type_check(enhancement_amount, (float, int), "enhancement_amount")
        check_value(enhancement_amount, [0, 100], "enhancement_amount")
        return method(self, *args, **kwargs)

    return new_method


def check_dc_shift(method):
    """Wrapper method to check the parameters of DCShift."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [shift, limiter_gain], _ = parse_user_args(method, *args, **kwargs)
        type_check(shift, (float, int), "shift")
        check_value(shift, [-2.0, 2.0], "shift")
        if limiter_gain is not None:
            type_check(limiter_gain, (float, int), "limiter_gain")
        return method(self, *args, **kwargs)
    return new_method


def check_deemph_biquad(method):
    """Wrapper method to check the parameters of CutMixBatch."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate], _ = parse_user_args(method, *args, **kwargs)
        type_check(sample_rate, (int,), "sample_rate")
        if sample_rate not in (44100, 48000):
            raise ValueError("Input sample_rate should be 44100 or 48000, but got {0}.".format(sample_rate))
        return method(self, *args, **kwargs)

    return new_method


def check_equalizer_biquad(method):
    """Wrapper method to check the parameters of EqualizerBiquad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, center_freq, gain, Q], _ = parse_user_args(method, *args, **kwargs)
        check_biquad_sample_rate(sample_rate)
        check_biquad_central_freq(center_freq)
        check_biquad_gain(gain)
        check_biquad_Q(Q)
        return method(self, *args, **kwargs)

    return new_method


def check_lfilter(method):
    """Wrapper method to check the parameters of lfilter."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [a_coeffs, b_coeffs, clamp], _ = parse_user_args(method, *args, **kwargs)
        type_check(a_coeffs, (list, tuple), "a_coeffs")
        type_check(b_coeffs, (list, tuple), "b_coeffs")
        for i, value in enumerate(a_coeffs):
            type_check(value, (float, int), "a_coeffs[{0}]".format(i))
            check_float32(value, "a_coeffs[{0}]".format(i))
        for i, value in enumerate(b_coeffs):
            type_check(value, (float, int), "b_coeffs[{0}]".format(i))
            check_float32(value, "b_coeffs[{0}]".format(i))
        check_list_same_size(a_coeffs, b_coeffs, "a_coeffs", "b_coeffs")
        type_check(clamp, (bool,), "clamp")
        return method(self, *args, **kwargs)

    return new_method


def check_lowpass_biquad(method):
    """Wrapper method to check the parameters of LowpassBiquad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, cutoff_freq, Q], _ = parse_user_args(method, *args, **kwargs)
        check_biquad_sample_rate(sample_rate)
        check_biquad_cutoff_freq(cutoff_freq)
        check_biquad_Q(Q)
        return method(self, *args, **kwargs)

    return new_method


def check_mu_law_decoding(method):
    """Wrapper method to check the parameters of MuLawDecoding"""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [quantization_channels], _ = parse_user_args(method, *args, **kwargs)
        check_pos_int32(quantization_channels, "quantization_channels")
        return method(self, *args, **kwargs)

    return new_method


def check_time_stretch(method):
    """Wrapper method to check the parameters of TimeStretch."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [hop_length, n_freq, fixed_rate], _ = parse_user_args(method, *args, **kwargs)

        if hop_length is not None:
            type_check(hop_length, (int,), "hop_length")
            check_pos_int32(hop_length, "hop_length")

        type_check(n_freq, (int,), "n_freq")
        check_pos_int32(n_freq, "n_freq")

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
        check_non_negative_float32(mask_value, "mask_value")
        return method(self, *args, **kwargs)

    return new_method


def check_power(power):
    """Wrapper method to check the parameters of power."""
    type_check(power, (int, float), "power")
    check_non_negative_float32(power, "power")


def check_complex_norm(method):
    """Wrapper method to check the parameters of ComplexNorm."""
    @wraps(method)
    def new_method(self, *args, **kwargs):
        [power], _ = parse_user_args(method, *args, **kwargs)
        check_power(power)
        return method(self, *args, **kwargs)

    return new_method


def check_magphase(method):
    """Wrapper method to check the parameters of Magphase."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [power], _ = parse_user_args(method, *args, **kwargs)
        check_power(power)
        return method(self, *args, **kwargs)

    return new_method


def check_biquad_coeff(coeff, arg_name):
    """Wrapper method to check the parameters of coeff."""
    type_check(coeff, (float, int), arg_name)
    check_float32(coeff, arg_name)


def check_biquad(method):
    """Wrapper method to check the parameters of Biquad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [b0, b1, b2, a0, a1, a2], _ = parse_user_args(
            method, *args, **kwargs)
        check_biquad_coeff(b0, "b0")
        check_biquad_coeff(b1, "b1")
        check_biquad_coeff(b2, "b2")
        type_check(a0, (float, int), "a0")
        check_float32_not_zero(a0, "a0")
        check_biquad_coeff(a1, "a1")
        check_biquad_coeff(a2, "a2")
        return method(self, *args, **kwargs)

    return new_method


def check_fade(method):
    """Wrapper method to check the parameters of Fade."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [fade_in_len, fade_out_len, fade_shape], _ = parse_user_args(method, *args, **kwargs)
        type_check(fade_in_len, (int,), "fade_in_len")
        check_non_negative_int32(fade_in_len, "fade_in_len")
        type_check(fade_out_len, (int,), "fade_out_len")
        check_non_negative_int32(fade_out_len, "fade_out_len")
        type_check(fade_shape, (FadeShape,), "fade_shape")
        return method(self, *args, **kwargs)

    return new_method


def check_vol(method):
    """Wrapper method to check the parameters of Vol."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [gain, gain_type], _ = parse_user_args(method, *args, **kwargs)
        # type check gain
        type_check(gain, (int, float), "gain")
        # type check gain_type and value check gain
        type_check(gain_type, (GainType,), "gain_type")
        if gain_type == GainType.AMPLITUDE:
            check_non_negative_float32(gain, "gain")
        elif gain_type == GainType.POWER:
            check_pos_float32(gain, "gain")
        else:
            check_float32(gain, "gain")
        return method(self, *args, **kwargs)

    return new_method
