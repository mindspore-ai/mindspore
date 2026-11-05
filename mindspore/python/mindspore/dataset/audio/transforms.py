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
The module audio.transforms is inherited from _c_dataengine and is
implemented based on C++. It's a high performance module to process
audio. Users can apply suitable augmentations on audio data to improve
their training models.
"""

import numpy as np

import mindspore._c_dataengine as cde
from .utils import BorderType, DensityFunction, FadeShape, GainType, Interpolation, MelType, Modulation, NormMode, \
    NormType, ResampleMethod, ScaleType, WindowType
from .validators import (
    check_allpass_biquad,
    check_band_biquad,
    check_bandpass_biquad,
    check_bandreject_biquad,
    check_bass_biquad,
    check_biquad,
    check_deemph_biquad,
    check_equalizer_biquad,
    check_highpass_biquad,
    check_lfilter,
    check_lowpass_biquad,
    check_riaa_biquad,
    check_treble_biquad,
)
from ..transforms.py_transforms_util import Implementation
from ..transforms.transforms import TensorOperation


class AudioTensorOperation(TensorOperation):
    """
    Base class of Audio Tensor Ops.
    """

    def __init__(self):
        super().__init__()
        self.implementation = Implementation.C

    def __call__(self, *input_tensor_list):
        for tensor in input_tensor_list:
            if not isinstance(tensor, (np.ndarray,)):
                raise TypeError("Input should be NumPy audio, got {}.".format(type(tensor)))
        return super().__call__(*input_tensor_list)

    def parse(self):
        raise NotImplementedError("AudioTensorOperation has to implement parse() method.")


class AllpassBiquad(AudioTensorOperation):
    r"""
    Design two-pole all-pass filter with central frequency and bandwidth for audio waveform.

    An all-pass filter changes the audio's frequency to phase relationship without changing
    its frequency to amplitude relationship. The system function is:

    .. math::
        H(s) = \frac{s^2 - \frac{s}{Q} + 1}{s^2 + \frac{s}{Q} + 1}

    Similar to the `SoX <https://sourceforge.net/projects/sox/>`_ implementation.

    Note:
        The shape of the audio waveform to be processed needs to be <..., time>.

    Args:
        sample_rate (int): Sampling rate (in Hz), which can't be zero.
        central_freq (float): Central frequency (in Hz).
        Q (float, optional): `Quality factor <https://en.wikipedia.org/wiki/Q_factor>`_ ,
            in range of (0, 1]. Default: ``0.707``.

    Raises:
        TypeError: If `sample_rate` is not of type int.
        ValueError: If `sample_rate` is 0.
        TypeError: If `central_freq` is not of type float.
        TypeError: If `Q` is not of type float.
        ValueError: If `Q` is not in range of (0, 1].
        RuntimeError: If input tensor is not in shape of <..., time>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode.
        >>> waveform = np.random.random([5, 16])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.AllpassBiquad(44100, 200.0)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (16,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([16])  # 1 sample
        >>> output = audio.AllpassBiquad(44100, 200.0)(waveform)
        >>> print(output.shape, output.dtype)
        (16,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_allpass_biquad
    def __init__(self, sample_rate, central_freq, Q=0.707):
        super().__init__()
        self.sample_rate = sample_rate
        self.central_freq = central_freq
        self.quality_factor = Q

    def parse(self):
        return cde.AllpassBiquadOperation(self.sample_rate, self.central_freq, self.quality_factor)


DE_C_SCALE_TYPE = {ScaleType.POWER: cde.ScaleType.DE_SCALE_TYPE_POWER,
                   ScaleType.MAGNITUDE: cde.ScaleType.DE_SCALE_TYPE_MAGNITUDE}


class BandBiquad(AudioTensorOperation):
    """
    Design two-pole band-pass filter for audio waveform.

    The frequency response drops logarithmically around the center frequency. The
    bandwidth gives the slope of the drop. The frequencies at band edge will be
    half of their original amplitudes.

    Similar to the `SoX <https://sourceforge.net/projects/sox/>`_ implementation.

    Note:
        The shape of the audio waveform to be processed needs to be <..., time>.

    Args:
        sample_rate (int): Sampling rate (in Hz), which can't be zero.
        central_freq (float): Central frequency (in Hz).
        Q (float, optional): `Quality factor <https://en.wikipedia.org/wiki/Q_factor>`_ ,
            in range of (0, 1]. Default: ``0.707``.
        noise (bool, optional): If ``True``, uses the alternate mode for un-pitched audio (e.g. percussion).
            If ``False``, uses mode oriented to pitched audio, i.e. voice, singing, or instrumental music.
            Default: ``False``.

    Raises:
        TypeError: If `sample_rate` is not of type int.
        ValueError: If `sample_rate` is 0.
        TypeError: If `central_freq` is not of type float.
        TypeError: If `Q` is not of type float.
        ValueError: If `Q` is not in range of (0, 1].
        TypeError: If `noise` is not of type bool.
        RuntimeError: If input tensor is not in shape of <..., time>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 16])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.BandBiquad(44100, 200.0)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (16,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([16])  # 1 sample
        >>> output = audio.BandBiquad(44100, 200.0)(waveform)
        >>> print(output.shape, output.dtype)
        (16,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_band_biquad
    def __init__(self, sample_rate, central_freq, Q=0.707, noise=False):
        super().__init__()
        self.sample_rate = sample_rate
        self.central_freq = central_freq
        self.quality_factor = Q
        self.noise = noise

    def parse(self):
        return cde.BandBiquadOperation(self.sample_rate, self.central_freq, self.quality_factor, self.noise)


class BandpassBiquad(AudioTensorOperation):
    r"""
    Design two-pole Butterworth band-pass filter for audio waveform.

    The frequency response of the Butterworth filter is maximally flat (i.e. has no ripples)
    in the passband and rolls off towards zero in the stopband.

    The system function of Butterworth band-pass filter is:

    .. math::
        H(s) = \begin{cases}
            \frac{s}{s^2 + \frac{s}{Q} + 1}, &\text{if const_skirt_gain=True}; \cr
            \frac{\frac{s}{Q}}{s^2 + \frac{s}{Q} + 1}, &\text{if const_skirt_gain=False}.
        \end{cases}

    Similar to the `SoX <https://sourceforge.net/projects/sox/>`_ implementation.

    Note:
        The shape of the audio waveform to be processed needs to be <..., time>.

    Args:
        sample_rate (int): Sampling rate (in Hz), which can't be zero.
        central_freq (float): Central frequency (in Hz).
        Q (float, optional): `Quality factor <https://en.wikipedia.org/wiki/Q_factor>`_ ,
            in range of (0, 1]. Default: ``0.707``.
        const_skirt_gain (bool, optional): If ``True``, uses a constant skirt gain (peak gain = Q);
            If ``False``, uses a constant 0dB peak gain. Default: ``False``.

    Raises:
        TypeError: If `sample_rate` is not of type int.
        ValueError: If `sample_rate` is 0.
        TypeError: If `central_freq` is not of type float.
        TypeError: If `Q` is not of type float.
        ValueError: If `Q` is not in range of (0, 1].
        TypeError: If `const_skirt_gain` is not of type bool.
        RuntimeError: If input tensor is not in shape of <..., time>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 16])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.BandpassBiquad(44100, 200.0)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (16,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([16])  # 1 sample
        >>> output = audio.BandpassBiquad(44100, 200.0)(waveform)
        >>> print(output.shape, output.dtype)
        (16,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_bandpass_biquad
    def __init__(self, sample_rate, central_freq, Q=0.707, const_skirt_gain=False):
        super().__init__()
        self.sample_rate = sample_rate
        self.central_freq = central_freq
        self.quality_factor = Q
        self.const_skirt_gain = const_skirt_gain

    def parse(self):
        return cde.BandpassBiquadOperation(self.sample_rate, self.central_freq, self.quality_factor,
                                           self.const_skirt_gain)


class BandrejectBiquad(AudioTensorOperation):
    r"""
    Design two-pole Butterworth band-reject filter for audio waveform.

    The frequency response of the Butterworth filter is maximally flat (i.e. has no ripples)
    in the passband and rolls off towards zero in the stopband.

    The system function of Butterworth band-reject filter is:

    .. math::
        H(s) = \frac{s^2 + 1}{s^2 + \frac{s}{Q} + 1}

    Similar to the `SoX <https://sourceforge.net/projects/sox/>`_ implementation.

    Note:
        The shape of the audio waveform to be processed needs to be <..., time>.

    Args:
        sample_rate (int): Sampling rate (in Hz), which can't be zero.
        central_freq (float): Central frequency (in Hz).
        Q (float, optional): `Quality factor <https://en.wikipedia.org/wiki/Q_factor>`_ ,
            in range of (0, 1]. Default: ``0.707``.

    Raises:
        TypeError: If `sample_rate` is not of type int.
        ValueError: If `sample_rate` is 0.
        TypeError: If `central_freq` is not of type float.
        TypeError: If `Q` is not of type float.
        ValueError: If `Q` is not in range of (0, 1].
        RuntimeError: If input tensor is not in shape of <..., time>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 16])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.BandrejectBiquad(44100, 200.0)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (16,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([16])  # 1 sample
        >>> output = audio.BandrejectBiquad(44100, 200.0)(waveform)
        >>> print(output.shape, output.dtype)
        (16,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_bandreject_biquad
    def __init__(self, sample_rate, central_freq, Q=0.707):
        super().__init__()
        self.sample_rate = sample_rate
        self.central_freq = central_freq
        self.quality_factor = Q

    def parse(self):
        return cde.BandrejectBiquadOperation(self.sample_rate, self.central_freq, self.quality_factor)


class BassBiquad(AudioTensorOperation):
    r"""
    Design a bass tone-control effect, also known as two-pole low-shelf filter for audio waveform.

    A low-shelf filter passes all frequencies, but increases or reduces frequencies below the shelf
    frequency by a specified amount. The system function is:

    .. math::
        H(s) = A\frac{s^2 + \frac{\sqrt{A}}{Q}s + A}{As^2 + \frac{\sqrt{A}}{Q}s + 1}

    Similar to the `SoX <https://sourceforge.net/projects/sox/>`_ implementation.

    Note:
        The shape of the audio waveform to be processed needs to be <..., time>.

    Args:
        sample_rate (int): Sampling rate (in Hz), which can't be zero.
        gain (float): Desired gain at the boost (or attenuation) in dB.
        central_freq (float, optional): Central frequency (in Hz). Default: ``100.0``.
        Q (float, optional): `Quality factor <https://en.wikipedia.org/wiki/Q_factor>`_ ,
            in range of (0, 1]. Default: ``0.707``.

    Raises:
        TypeError: If `sample_rate` is not of type int.
        ValueError: If `sample_rate` is 0.
        TypeError: If `gain` is not of type float.
        TypeError: If `central_freq` is not of type float.
        TypeError: If `Q` is not of type float.
        ValueError: If `Q` is not in range of (0, 1].
        RuntimeError: If input tensor is not in shape of <..., time>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 16])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.BassBiquad(44100, 100.0)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (16,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([16])  # 1 sample
        >>> output = audio.BassBiquad(44100, 200.0)(waveform)
        >>> print(output.shape, output.dtype)
        (16,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_bass_biquad
    def __init__(self, sample_rate, gain, central_freq=100.0, Q=0.707):
        super().__init__()
        self.sample_rate = sample_rate
        self.gain = gain
        self.central_freq = central_freq
        self.quality_factor = Q

    def parse(self):
        return cde.BassBiquadOperation(self.sample_rate, self.gain, self.central_freq, self.quality_factor)


class Biquad(TensorOperation):
    """
    Perform a biquad filter of input audio.
    Mathematical formulas refer to: `Digital_biquad_filter <https://en.wikipedia.org/wiki/Digital_biquad_filter>`_ .

    Args:
        b0 (float): Numerator coefficient of current input, x[n].
        b1 (float): Numerator coefficient of input one time step ago, x[n-1].
        b2 (float): Numerator coefficient of input two time steps ago, x[n-2].
        a0 (float): Denominator coefficient of current output y[n], the value can't be 0, typically 1.
        a1 (float): Denominator coefficient of current output y[n-1].
        a2 (float): Denominator coefficient of current output y[n-2].

    Raises:
        TypeError: If `b0` is not of type float.
        TypeError: If `b1` is not of type float.
        TypeError: If `b2` is not of type float.
        TypeError: If `a0` is not of type float.
        TypeError: If `a1` is not of type float.
        TypeError: If `a2` is not of type float.
        ValueError: If `a0` is 0.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 16])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.Biquad(0.01, 0.02, 0.13, 1, 0.12, 0.3)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (16,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([16])  # 1 sample
        >>> output = audio.Biquad(0.01, 0.02, 0.13, 1, 0.12, 0.3)(waveform)
        >>> print(output.shape, output.dtype)
        (16,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_biquad
    def __init__(self, b0, b1, b2, a0, a1, a2):
        super().__init__()
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2

    def parse(self):
        return cde.BiquadOperation(self.b0, self.b1, self.b2, self.a0, self.a1, self.a2)


DE_C_BORDER_TYPE = {
    BorderType.CONSTANT: cde.BorderType.DE_BORDER_CONSTANT,
    BorderType.EDGE: cde.BorderType.DE_BORDER_EDGE,
    BorderType.REFLECT: cde.BorderType.DE_BORDER_REFLECT,
    BorderType.SYMMETRIC: cde.BorderType.DE_BORDER_SYMMETRIC,
}




class DeemphBiquad(AudioTensorOperation):
    """
    Apply Compact Disc (IEC 60908) de-emphasis (a treble attenuation shelving filter) to the audio waveform.

    Similar to the `SoX <https://sourceforge.net/projects/sox/>`_ implementation.

    Args:
        sample_rate (int): Sampling rate of the waveform, must be ``44100`` or ``48000`` (Hz).

    Raises:
        TypeError: If `sample_rate` is not of type int.
        ValueError: If `sample_rate` is not ``44100`` or ``48000`` .
        RuntimeError: If input tensor is not in shape of <..., time>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 8])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.DeemphBiquad(44100)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (8,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([8])  # 1 sample
        >>> output = audio.DeemphBiquad(44100)(waveform)
        >>> print(output.shape, output.dtype)
        (8,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_deemph_biquad
    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate

    def parse(self):
        return cde.DeemphBiquadOperation(self.sample_rate)


DE_C_DENSITY_FUNCTION = {DensityFunction.TPDF: cde.DensityFunction.DE_DENSITY_FUNCTION_TPDF,
                         DensityFunction.RPDF: cde.DensityFunction.DE_DENSITY_FUNCTION_RPDF,
                         DensityFunction.GPDF: cde.DensityFunction.DE_DENSITY_FUNCTION_GPDF}


class EqualizerBiquad(AudioTensorOperation):
    """
    Design biquad equalizer filter and perform filtering.

    Similar to the `SoX <https://sourceforge.net/projects/sox/>`_ implementation.

    Args:
        sample_rate (int): Sampling rate of the waveform, e.g. ``44100`` (Hz), the value can't be 0.
        center_freq (float): Central frequency (in Hz).
        gain (float): Desired gain at the boost (or attenuation) in dB.
        Q (float, optional): https://en.wikipedia.org/wiki/Q_factor, range: (0, 1]. Default: ``0.707``.

    Raises:
        TypeError: If `sample_rate` is not of type int.
        ValueError: If `sample_rate` is 0.
        TypeError: If `center_freq` is not of type float.
        TypeError: If `gain` is not of type float.
        TypeError: If `Q` is not of type float.
        ValueError: If `Q` is not in range of (0, 1].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 16])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.EqualizerBiquad(44100, 1500, 5.5, 0.7)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (16,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([16])  # 1 sample
        >>> output = audio.EqualizerBiquad(44100, 1500, 5.5, 0.7)(waveform)
        >>> print(output.shape, output.dtype)
        (16,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_equalizer_biquad
    def __init__(self, sample_rate, center_freq, gain, Q=0.707):
        super().__init__()
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.gain = gain
        self.quality_factor = Q

    def parse(self):
        return cde.EqualizerBiquadOperation(self.sample_rate, self.center_freq, self.gain, self.quality_factor)


DE_C_FADE_SHAPE = {FadeShape.QUARTER_SINE: cde.FadeShape.DE_FADE_SHAPE_QUARTER_SINE,
                   FadeShape.HALF_SINE: cde.FadeShape.DE_FADE_SHAPE_HALF_SINE,
                   FadeShape.LINEAR: cde.FadeShape.DE_FADE_SHAPE_LINEAR,
                   FadeShape.LOGARITHMIC: cde.FadeShape.DE_FADE_SHAPE_LOGARITHMIC,
                   FadeShape.EXPONENTIAL: cde.FadeShape.DE_FADE_SHAPE_EXPONENTIAL}


class Filtfilt(AudioTensorOperation):
    """
    Apply an IIR filter forward and backward to a waveform.

    Args:
        a_coeffs (Sequence[float]): Denominator coefficients of difference equation of dimension.
            Lower delay coefficients are first, e.g. [a0, a1, a2, ...].
            Must be same size as b_coeffs (pad with 0's as necessary).
        b_coeffs (Sequence[float]): Numerator coefficients of difference equation of dimension.
            Lower delay coefficients are first, e.g. [b0, b1, b2, ...].
            Must be same size as a_coeffs (pad with 0's as necessary).
        clamp (bool, optional): If ``True``, clamp the output signal to be in the range [-1, 1].
            Default: ``True``.

    Raises:
        TypeError: If `a_coeffs` is not of type Sequence[float].
        TypeError: If `b_coeffs` is not of type Sequence[float].
        ValueError: If `a_coeffs` and `b_coeffs` are of different sizes.
        TypeError: If `clamp` is not of type bool.
        RuntimeError: If shape of the input audio is not <..., time>.

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 16])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.Filtfilt(a_coeffs=[0.1, 0.2, 0.3], b_coeffs=[0.1, 0.2, 0.3])]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (16,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([16])  # 1 sample
        >>> output = audio.Filtfilt(a_coeffs=[0.1, 0.2, 0.3], b_coeffs=[0.1, 0.2, 0.3])(waveform)
        >>> print(output.shape, output.dtype)
        (16,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_lfilter
    def __init__(self, a_coeffs, b_coeffs, clamp=True):
        super().__init__()
        self.a_coeffs = a_coeffs
        self.b_coeffs = b_coeffs
        self.clamp = clamp

    def parse(self):
        return cde.FiltfiltOperation(self.a_coeffs, self.b_coeffs, self.clamp)


DE_C_MODULATION = {Modulation.SINUSOIDAL: cde.Modulation.DE_MODULATION_SINUSOIDAL,
                   Modulation.TRIANGULAR: cde.Modulation.DE_MODULATION_TRIANGULAR}

DE_C_INTERPOLATION = {Interpolation.LINEAR: cde.Interpolation.DE_INTERPOLATION_LINEAR,
                      Interpolation.QUADRATIC: cde.Interpolation.DE_INTERPOLATION_QUADRATIC}




class HighpassBiquad(AudioTensorOperation):
    """
    Design biquad highpass filter and perform filtering.

    Similar to the `SoX <https://sourceforge.net/projects/sox/>`_ implementation.

    Args:
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be 0.
        cutoff_freq (float): Filter cutoff frequency (in Hz).
        Q (float, optional): Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1]. Default: ``0.707``.

    Raises:
        TypeError: If `sample_rate` is not of type int.
        ValueError: If `sample_rate` is 0.
        TypeError: If `cutoff_freq` is not of type float.
        TypeError: If `Q` is not of type float.
        ValueError: If `Q` is not in range of (0, 1].
        RuntimeError: If the shape of input audio waveform does not match <..., time>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 16])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.HighpassBiquad(44100, 1500, 0.7)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (16,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([16])  # 1 sample
        >>> output = audio.HighpassBiquad(44100, 1500, 0.7)(waveform)
        >>> print(output.shape, output.dtype)
        (16,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_highpass_biquad
    def __init__(self, sample_rate, cutoff_freq, Q=0.707):
        super().__init__()
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        self.quality_factor = Q

    def parse(self):
        return cde.HighpassBiquadOperation(self.sample_rate, self.cutoff_freq, self.quality_factor)


DE_C_NORM_MODE = {NormMode.ORTHO: cde.NormMode.DE_NORM_MODE_ORTHO,
                  NormMode.NONE: cde.NormMode.DE_NORM_MODE_NONE}


class LFilter(AudioTensorOperation):
    """
    Perform an IIR filter by evaluating a different equation.

    Args:
        a_coeffs (Sequence[float]): Denominator coefficients of difference equation of dimension.
            Lower delays coefficients are first, e.g. [a0, a1, a2, ...].
            Must be same size as b_coeffs (pad with 0's as necessary).
        b_coeffs (Sequence[float]): Numerator coefficients of difference equation of dimension.
            Lower delays coefficients are first, e.g. [b0, b1, b2, ...].
            Must be same size as a_coeffs (pad with 0's as necessary).
        clamp (bool, optional): If True, clamp the output signal to be in the range [-1, 1]. Default: ``True``.

    Raises:
        TypeError: If `a_coeffs` is not of type Sequence[float].
        TypeError: If `b_coeffs` is not of type Sequence[float].
        ValueError: If `a_coeffs` and `b_coeffs` are of different sizes.
        TypeError: If `clamp` is not of type bool.
        RuntimeError: If input tensor is not in shape of <..., time>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 16])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.LFilter(a_coeffs=[0.1, 0.2, 0.3], b_coeffs=[0.3, 0.2, 0.1])]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (16,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([16])  # 1 sample
        >>> output = audio.LFilter(a_coeffs=[0.1, 0.2, 0.3], b_coeffs=[0.3, 0.2, 0.1])(waveform)
        >>> print(output.shape, output.dtype)
        (16,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_lfilter
    def __init__(self, a_coeffs, b_coeffs, clamp=True):
        super().__init__()
        self.a_coeffs = a_coeffs
        self.b_coeffs = b_coeffs
        self.clamp = clamp

    def parse(self):
        return cde.LFilterOperation(self.a_coeffs, self.b_coeffs, self.clamp)


class LowpassBiquad(AudioTensorOperation):
    r"""
    Design two-pole low-pass filter for audio waveform.

    A low-pass filter passes frequencies lower than a selected cutoff frequency
    but attenuates frequencies higher than it. The system function is:

    .. math::
        H(s) = \frac{1}{s^2 + \frac{s}{Q} + 1}

    Similar to the `SoX <https://sourceforge.net/projects/sox/>`_ implementation.

    Note:
        The shape of the audio waveform to be processed needs to be <..., time>.

    Args:
        sample_rate (int): Sampling rate (in Hz), which can't be zero.
        cutoff_freq (float): Filter cutoff frequency (in Hz).
        Q (float, optional): `Quality factor <https://en.wikipedia.org/wiki/Q_factor>`_ ,
            in range of (0, 1]. Default: ``0.707``.

    Raises:
        TypeError: If `sample_rate` is not of type int.
        ValueError: If `sample_rate` is 0.
        TypeError: If `cutoff_freq` is not of type float.
        TypeError: If `Q` is not of type float.
        ValueError: If `Q` is not in range of (0, 1].
        RuntimeError: If input tensor is not in shape of <..., time>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 10])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.LowpassBiquad(4000, 1500, 0.7)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (10,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([10])  # 1 sample
        >>> output = audio.LowpassBiquad(4000, 1500, 0.7)(waveform)
        >>> print(output.shape, output.dtype)
        (10,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_lowpass_biquad
    def __init__(self, sample_rate, cutoff_freq, Q=0.707):
        super().__init__()
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        self.quality_factor = Q

    def parse(self):
        return cde.LowpassBiquadOperation(self.sample_rate, self.cutoff_freq, self.quality_factor)


DE_C_MEL_TYPE = {MelType.SLANEY: cde.MelType.DE_MEL_TYPE_SLANEY,
                 MelType.HTK: cde.MelType.DE_MEL_TYPE_HTK}

DE_C_NORM_TYPE = {NormType.NONE: cde.NormType.DE_NORM_TYPE_NONE,
                  NormType.SLANEY: cde.NormType.DE_NORM_TYPE_SLANEY}


DE_C_RESAMPLE_METHOD = {ResampleMethod.SINC_INTERPOLATION: cde.ResampleMethod.DE_RESAMPLE_SINC_INTERPOLATION,
                        ResampleMethod.KAISER_WINDOW: cde.ResampleMethod.DE_RESAMPLE_KAISER_WINDOW}


class RiaaBiquad(AudioTensorOperation):
    """
    Apply RIAA vinyl playback equalization.

    Similar to the `SoX <https://sourceforge.net/projects/sox/>`_ implementation.

    Args:
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz),
            can only be one of 44100, 48000, 88200, 96000.

    Raises:
        TypeError: If `sample_rate` is not of type int.
        ValueError: If `sample_rate` is not any of [44100, 48000, 88200, 96000].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 24])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.RiaaBiquad(44100)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (24,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([24])  # 1 sample
        >>> output = audio.RiaaBiquad(44100)(waveform)
        >>> print(output.shape, output.dtype)
        (24,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_riaa_biquad
    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate

    def parse(self):
        return cde.RiaaBiquadOperation(self.sample_rate)


DE_C_WINDOW_TYPE = {WindowType.BARTLETT: cde.WindowType.DE_WINDOW_TYPE_BARTLETT,
                    WindowType.BLACKMAN: cde.WindowType.DE_WINDOW_TYPE_BLACKMAN,
                    WindowType.HAMMING: cde.WindowType.DE_WINDOW_TYPE_HAMMING,
                    WindowType.HANN: cde.WindowType.DE_WINDOW_TYPE_HANN,
                    WindowType.KAISER: cde.WindowType.DE_WINDOW_TYPE_KAISER}


class TrebleBiquad(AudioTensorOperation):
    """
    Design a treble tone-control effect.

    Similar to the `SoX <https://sourceforge.net/projects/sox/>`_ implementation.

    Args:
        sample_rate (int): Sampling rate (in Hz), which can't be zero.
        gain (float): Desired gain at the boost (or attenuation) in dB.
        central_freq (float, optional): Central frequency (in Hz). Default: ``3000``.
        Q (float, optional): `Quality factor <https://en.wikipedia.org/wiki/Q_factor>`_ ,
            in range of (0, 1]. Default: ``0.707``.

    Raises:
        TypeError: If `sample_rate` is not of type int.
        ValueError: If `sample_rate` is 0.
        TypeError: If `gain` is not of type float.
        TypeError: If `central_freq` is not of type float.
        TypeError: If `Q` is not of type float.
        ValueError: If `Q` is not in range of (0, 1].
        RuntimeError: If input tensor is not in shape of <..., time>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 20])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.TrebleBiquad(44100, 200.0)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (20,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([20])  # 1 sample
        >>> output = audio.TrebleBiquad(44100, 200.0)(waveform)
        >>> print(output.shape, output.dtype)
        (20,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_treble_biquad
    def __init__(self, sample_rate, gain, central_freq=3000, Q=0.707):
        super().__init__()
        self.sample_rate = sample_rate
        self.gain = gain
        self.central_freq = central_freq
        self.quality_factor = Q

    def parse(self):
        return cde.TrebleBiquadOperation(self.sample_rate, self.gain, self.central_freq, self.quality_factor)


DE_C_GAIN_TYPE = {GainType.AMPLITUDE: cde.GainType.DE_GAIN_TYPE_AMPLITUDE,
                  GainType.POWER: cde.GainType.DE_GAIN_TYPE_POWER,
                  GainType.DB: cde.GainType.DE_GAIN_TYPE_DB}
