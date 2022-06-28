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
"""
Enum for audio ops.
"""
from enum import Enum

import mindspore._c_dataengine as cde
from mindspore.dataset.core.validator_helpers import check_non_negative_float32, check_non_negative_int32, \
    check_pos_float32, check_pos_int32, type_check


class BorderType(str, Enum):
    """
    Padding Mode, BorderType Type.

    Possible enumeration values are: BorderType.CONSTANT, BorderType.EDGE, BorderType.REFLECT, BorderType.SYMMETRIC.

    - BorderType.CONSTANT: means it fills the border with constant values.
    - BorderType.EDGE: means it pads with the last value on the edge.
    - BorderType.REFLECT: means it reflects the values on the edge omitting the last value of edge.
    - BorderType.SYMMETRIC: means it reflects the values on the edge repeating the last value of edge.

    Note: This class derived from class str to support json serializable.
    """
    CONSTANT: str = "constant"
    EDGE: str = "edge"
    REFLECT: str = "reflect"
    SYMMETRIC: str = "symmetric"


class DensityFunction(str, Enum):
    """
    Density Functions.

    Possible enumeration values are: DensityFunction.TPDF, DensityFunction.RPDF,
    DensityFunction.GPDF.

    - DensityFunction.TPDF: means triangular probability density function.
    - DensityFunction.RPDF: means rectangular probability density function.
    - DensityFunction.GPDF: means gaussian probability density function.
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
    Mel Types.

    Possible enumeration values are: MelType.HTK, MelType.SLANEY.

    - MelType.NONE: scale the input data with htk.
    - MelType.ORTHO: scale the input data with slaney.
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
    Norm Types.

    Possible enumeration values are: NormMode.ORTHO, NormMode.NONE.

    - NormMode.ORTHO: means the mode of input audio is ortho.
    - NormMode.NONE: means the mode of input audio is none.
    """
    ORTHO: str = "ortho"
    NONE: str = "none"


class NormType(str, Enum):
    """
    Norm Types.

    Possible enumeration values are: NormType.SLANEY, NormType.NONE.

    - NormType.SLANEY: norm the input data with slaney.
    - NormType.NONE: norm the input data with none.
    """
    SLANEY: str = "slaney"
    NONE: str = "none"


class ResampleMethod(str, Enum):
    """
    Resample method

    Possible enumeration values are: ResampleMethod.SINC_INTERPOLATION, ResampleMethod.KAISER_WINDOW.

    - ResampleMethod.SINC_INTERPOLATION: means it resample the tensor with sinc interpolation.
    - ResampleMethod.KAISER_WINDOW: means it resample the tensor with kaiser window.
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
    Window Function types,

    Possible enumeration values are: WindowType.BARTLETT, WindowType.BLACKMAN, WindowType.HAMMING, WindowType.HANN,
    WindowType.KAISER.

    - WindowType.BARTLETT: means the type of window function is Bartlett.
    - WindowType.BLACKMAN: means the type of window function is Blackman.
    - WindowType.HAMMING: means the type of window function is Hamming.
    - WindowType.HANN: means the type of window function is Hann.
    - WindowType.KAISER: means the type of window function is Kaiser, currently not supported on macOS.
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
        norm (NormMode): Normalization mode, can be NormMode.NONE or NormMode.ORTHO (default=NormMode.NONE).

    Returns:
        numpy.ndarray, the transformation matrix, to be right-multiplied to row-wise data of size (n_mels, n_mfcc).

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


def melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate, norm=NormType.NONE, mel_type=MelType.HTK):
    """
    Create a frequency transformation matrix with shape (n_freqs, n_mels).

    Args:
        n_freqs (int): Number of frequency.
        f_min (float): Minimum of frequency in Hz.
        f_max (float): Maximum of frequency in Hz.
        n_mels (int): Number of mel filterbanks.
        sample_rate (int): Sample rate.
        norm (NormType, optional): Norm to use, can be NormType.NONE or NormType.SLANEY (Default: NormType.NONE).
        mel_type (MelType, optional): Scale to use, can be MelType.HTK or MelType.SLANEY (Default: NormType.SLANEY).

    Returns:
        numpy.ndarray, the frequency transformation matrix.

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
