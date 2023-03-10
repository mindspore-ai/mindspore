# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""Enum for audio ops."""
from __future__ import absolute_import

from enum import Enum

import mindspore._c_dataengine as cde
from mindspore.dataset.core.validator_helpers import check_non_negative_float32, check_non_negative_int32, \
    check_pos_float32, check_pos_int32, type_check


class BorderType(str, Enum):
    """
    Padding mode.

    Possible enumeration values are: BorderType.CONSTANT, BorderType.EDGE, BorderType.REFLECT, BorderType.SYMMETRIC.

    - BorderType.CONSTANT: Pad with a constant value.
    - BorderType.EDGE: Pad with the last value on the edge.
    - BorderType.REFLECT: Reflect the value on the edge while omitting the last one.
      For example, pad [1, 2, 3, 4] with 2 elements on both sides will result in [3, 2, 1, 2, 3, 4, 3, 2].
    - BorderType.SYMMETRIC: Reflect the value on the edge while repeating the last one.
      For example, pad [1, 2, 3, 4] with 2 elements on both sides will result in [2, 1, 1, 2, 3, 4, 4, 3].

    Note:
        This class derived from class str to support json serializable.
    """
    CONSTANT: str = "constant"
    EDGE: str = "edge"
    REFLECT: str = "reflect"
    SYMMETRIC: str = "symmetric"


class DensityFunction(str, Enum):
    """
    Density function type.

    Possible enumeration values are: DensityFunction.TPDF, DensityFunction.RPDF,
    DensityFunction.GPDF.

    - DensityFunction.TPDF: Triangular Probability Density Function.
    - DensityFunction.RPDF: Rectangular Probability Density Function.
    - DensityFunction.GPDF: Gaussian Probability Density Function.
    """
    TPDF: str = "TPDF"
    RPDF: str = "RPDF"
    GPDF: str = "GPDF"


class FadeShape(str, Enum):
    """
    Fade Shapes.

    Possible enumeration values are: FadeShape.QUARTER_SINE, FadeShape.HALF_SINE, FadeShape.LINEAR,
    FadeShape.LOGARITHMIC, FadeShape.EXPONENTIAL.

    - FadeShape.QUARTER_SINE: means the fade shape is quarter_sine mode.
    - FadeShape.HALF_SINE: means the fade shape is half_sine mode.
    - FadeShape.LINEAR: means the fade shape is linear mode.
    - FadeShape.LOGARITHMIC: means the fade shape is logarithmic mode.
    - FadeShape.EXPONENTIAL: means the fade shape is exponential mode.
    """
    QUARTER_SINE: str = "quarter_sine"
    HALF_SINE: str = "half_sine"
    LINEAR: str = "linear"
    LOGARITHMIC: str = "logarithmic"
    EXPONENTIAL: str = "exponential"


class GainType(str, Enum):
    """
    Gain Types.

    Possible enumeration values are: GainType.AMPLITUDE, GainType.POWER, GainType.DB.

    - GainType.AMPLITUDE: means input gain type is amplitude.
    - GainType.POWER: means input gain type is power.
    - GainType.DB: means input gain type is decibel.
    """
    AMPLITUDE: str = "amplitude"
    POWER: str = "power"
    DB: str = "db"


class Interpolation(str, Enum):
    """
    Interpolation Type.

    Possible enumeration values are: Interpolation.LINEAR, Interpolation.QUADRATIC.

    - Interpolation.LINEAR: means input interpolation type is linear.
    - Interpolation.QUADRATIC: means input interpolation type is quadratic.
    """
    LINEAR: str = "linear"
    QUADRATIC: str = "quadratic"


class MelType(str, Enum):
    """
    Mel scale implementation type.

    Possible enumeration values are: MelType.HTK, MelType.SLANEY.

    - MelType.HTK: The Hidden Markov Toolkit (HTK) implementation, refer to `HTK <https://htk.eng.cam.ac.uk/>`_ .
    - MelType.SLANEY: The MATLAB Auditory Toolbox of Slaney implementation,
      refer to `Auditory Toolbox <https://engineering.purdue.edu/~malcolm/interval/1998-010/>`_ .
    """
    HTK: str = "htk"
    SLANEY: str = "slaney"


class Modulation(str, Enum):
    """
    Modulation Type.

    Possible enumeration values are: Modulation.SINUSOIDAL, Modulation.TRIANGULAR.

    - Modulation.SINUSOIDAL: means input modulation type is sinusoidal.
    - Modulation.TRIANGULAR: means input modulation type is triangular.
    """
    SINUSOIDAL: str = "sinusoidal"
    TRIANGULAR: str = "triangular"


class NormMode(str, Enum):
    """
    Normalization mode.

    Possible enumeration values are: NormMode.ORTHO, NormMode.NONE.

    - NormMode.ORTHO: Use an ortho-normal DCT basis.
    - NormMode.NONE: No normalization.
    """
    ORTHO: str = "ortho"
    NONE: str = "none"


class NormType(str, Enum):
    """
    Normalization type.

    Possible enumeration values are: NormType.SLANEY, NormType.NONE.

    - NormType.SLANEY: Use an area normalization.
    - NormType.NONE: No narmalization.
    """
    SLANEY: str = "slaney"
    NONE: str = "none"


class ResampleMethod(str, Enum):
    """
    Resample method.

    Possible enumeration values are: ResampleMethod.SINC_INTERPOLATION, ResampleMethod.KAISER_WINDOW.

    - ResampleMethod.SINC_INTERPOLATION: The Whittaker-Shannon interpolation or sinc interpolation formula.
    - ResampleMethod.KAISER_WINDOW: The Kaiser window interpolation.
    """
    SINC_INTERPOLATION: str = "sinc_interpolation"
    KAISER_WINDOW: str = "kaiser_window"


class ScaleType(str, Enum):
    """
    Scale Types.

    Possible enumeration values are: ScaleType.POWER, ScaleType.MAGNITUDE.

    - ScaleType.POWER: means the scale of input audio is power.
    - ScaleType.MAGNITUDE: means the scale of input audio is magnitude.
    """
    POWER: str = "power"
    MAGNITUDE: str = "magnitude"


class WindowType(str, Enum):
    """
    Window function type.

    Possible enumeration values are: WindowType.BARTLETT, WindowType.BLACKMAN, WindowType.HAMMING, WindowType.HANN,
    WindowType.KAISER.

    - WindowType.BARTLETT: Bartlett window function.
    - WindowType.BLACKMAN: Blackman window function.
    - WindowType.HAMMING: Hamming window function.
    - WindowType.HANN: Hann window function.
    - WindowType.KAISER: Kaiser window function. Currently, it is not supported on macOS.
    """
    BARTLETT: str = "bartlett"
    BLACKMAN: str = "blackman"
    HAMMING: str = "hamming"
    HANN: str = "hann"
    KAISER: str = "kaiser"


DE_C_NORM_MODE = {NormMode.ORTHO: cde.NormMode.DE_NORM_MODE_ORTHO,
                  NormMode.NONE: cde.NormMode.DE_NORM_MODE_NONE}


def create_dct(n_mfcc, n_mels, norm=NormMode.NONE):
    """
    Create a DCT transformation matrix with shape (n_mels, n_mfcc), normalized depending on norm.

    Args:
        n_mfcc (int): Number of mfc coefficients to retain, the value must be greater than 0.
        n_mels (int): Number of mel filterbanks, the value must be greater than 0.
        norm (NormMode, optional): Normalization mode, can be NormMode.NONE or NormMode.ORTHO. Default: NormMode.NONE.

    Returns:
        numpy.ndarray, the transformation matrix, to be right-multiplied to row-wise data of size (n_mels, n_mfcc).

    Raises:
        TypeError: If `n_mfcc` is not of type int.
        ValueError: If `n_mfcc` is not positive.
        TypeError: If `n_mels` is not of type int.
        ValueError: If `n_mels` is not positive.
        TypeError: If `n_mels` is not of type :class:`mindspore.dataset.audio.NormMode` .

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.audio import create_dct, NormMode
        >>>
        >>> dct = create_dct(100, 200, NormMode.NONE)
    """

    if not isinstance(n_mfcc, int):
        raise TypeError("n_mfcc with value {0} is not of type {1}, but got {2}.".format(
            n_mfcc, int, type(n_mfcc)))
    if not isinstance(n_mels, int):
        raise TypeError("n_mels with value {0} is not of type {1}, but got {2}.".format(
            n_mels, int, type(n_mels)))
    if not isinstance(norm, NormMode):
        raise TypeError("norm with value {0} is not of type {1}, but got {2}.".format(
            norm, NormMode, type(norm)))
    if n_mfcc <= 0:
        raise ValueError("n_mfcc must be greater than 0, but got {0}.".format(n_mfcc))
    if n_mels <= 0:
        raise ValueError("n_mels must be greater than 0, but got {0}.".format(n_mels))
    return cde.create_dct(n_mfcc, n_mels, DE_C_NORM_MODE[norm]).as_array()


DE_C_MEL_TYPE = {MelType.HTK: cde.MelType.DE_MEL_TYPE_HTK,
                 MelType.SLANEY: cde.MelType.DE_MEL_TYPE_SLANEY}

DE_C_NORM_TYPE = {NormType.SLANEY: cde.NormType.DE_NORM_TYPE_SLANEY,
                  NormType.NONE: cde.NormType.DE_NORM_TYPE_NONE}


def linear_fbanks(n_freqs, f_min, f_max, n_filter, sample_rate):
    """
    Creates a linear triangular filterbank.

    Args:
        n_freqs (int): Number of frequencies to highlight/apply.
        f_min (float): Minimum frequency in Hz.
        f_max (float): Maximum frequency in Hz.
        n_filter (int): Number of (linear) triangular filter.
        sample_rate (int): Sample rate of the waveform.

    Returns:
        numpy.ndarray, the linear triangular filterbank.

    Raises:
        TypeError: If `n_freqs` is not of type int.
        ValueError: If `n_freqs` is negative.
        TypeError: If `f_min` is not of type float.
        ValueError: If `f_min` is negative.
        TypeError: If `f_max` is not of type float.
        ValueError: If `f_max` is negative.
        ValueError: If `f_min` is larger than `f_max`.
        TypeError: If `n_filter` is not of type int.
        ValueError: If `n_filter` is not positive.
        TypeError: If `sample_rate` is not of type int.
        ValueError: If `sample_rate` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.audio import linear_fbanks
        >>>
        >>> fbanks = linear_fbanks(n_freqs=4096, f_min=0, f_max=8000, n_filter=40, sample_rate=16000)
    """

    type_check(n_freqs, (int,), "n_freqs")
    check_non_negative_int32(n_freqs, "n_freqs")

    type_check(f_min, (int, float,), "f_min")
    check_non_negative_float32(f_min, "f_min")

    type_check(f_max, (int, float,), "f_max")
    check_pos_float32(f_max, "f_max")
    if f_min > f_max:
        raise ValueError(
            "Input f_min should be no more than f_max, but got f_min: {0} and f_max: {1}.".format(f_min, f_max))

    type_check(n_filter, (int,), "n_filter")
    check_pos_int32(n_filter, "n_filter")

    type_check(sample_rate, (int,), "sample_rate")
    check_pos_int32(sample_rate, "sample_rate")

    return cde.linear_fbanks(n_freqs, f_min, f_max, n_filter, sample_rate).as_array()


def melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate, norm=NormType.NONE, mel_type=MelType.HTK):
    """
    Create a frequency transformation matrix.

    Args:
        n_freqs (int): Number of frequencies to highlight/apply.
        f_min (float): Minimum of frequency in Hz.
        f_max (float): Maximum of frequency in Hz.
        n_mels (int): Number of mel filterbanks.
        sample_rate (int): Sample rate of the audio waveform.
        norm (NormType, optional): Normalization method, can be NormType.NONE or NormType.SLANEY.
            Default: NormType.NONE.
        mel_type (MelType, optional): Scale to use, can be MelType.HTK or MelType.SLANEY. Default: MelType.HTK.

    Returns:
        numpy.ndarray, the frequency transformation matrix with shape ( `n_freqs` , `n_mels` ).

    Raises:
        TypeError: If `n_freqs` is not of type int.
        ValueError: If `n_freqs` is a negative number.
        TypeError: If `f_min` is not of type float.
        ValueError: If `f_min` is greater than `f_max` .
        TypeError: If `f_max` is not of type float.
        ValueError: If `f_max` is a negative number.
        TypeError: If `n_mels` is not of type int.
        ValueError: If `n_mels` is not positive.
        TypeError: If `sample_rate` is not of type int.
        ValueError: If `sample_rate` is not positive.
        TypeError: If `norm` is not of type :class:`mindspore.dataset.audio.NormType` .
        TypeError: If `mel_type` is not of type :class:`mindspore.dataset.audio.MelType` .

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.audio import melscale_fbanks
        >>>
        >>> fbanks = melscale_fbanks(n_freqs=4096, f_min=0, f_max=8000, n_mels=40, sample_rate=16000)
    """

    type_check(n_freqs, (int,), "n_freqs")
    check_non_negative_int32(n_freqs, "n_freqs")

    type_check(f_min, (int, float,), "f_min")
    check_non_negative_float32(f_min, "f_min")

    type_check(f_max, (int, float,), "f_max")
    check_pos_float32(f_max, "f_max")
    if f_min > f_max:
        raise ValueError(
            "Input f_min should be no more than f_max, but got f_min: {0} and f_max: {1}.".format(f_min, f_max))

    type_check(n_mels, (int,), "n_mels")
    check_pos_int32(n_mels, "n_mels")

    type_check(sample_rate, (int,), "sample_rate")
    check_pos_int32(sample_rate, "sample_rate")

    type_check(norm, (NormType,), "norm")
    type_check(mel_type, (MelType,), "mel_type")
    return cde.melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate, DE_C_NORM_TYPE[norm],
                               DE_C_MEL_TYPE[mel_type]).as_array()
