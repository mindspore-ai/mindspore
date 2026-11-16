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
from .validators import check_allpass_biquad, check_band_biquad, check_bandpass_biquad, \
    check_bandreject_biquad, check_bass_biquad, check_biquad, check_contrast, check_dc_shift, \
    check_deemph_biquad, check_dither, check_equalizer_biquad, check_fade, check_flanger, check_gain, \
    check_highpass_biquad, check_lfilter, check_lowpass_biquad, check_mu_law_coding, check_overdrive, \
    check_phaser, check_riaa_biquad, check_treble_biquad, check_vad, check_vol
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


class Contrast(AudioTensorOperation):
    """
    Apply contrast effect for audio waveform.

    Comparable with compression, this effect modifies an audio signal to make it sound louder.

    Similar to the `SoX <https://sourceforge.net/projects/sox/>`_ implementation.

    Note:
        The shape of the audio waveform to be processed needs to be <..., time>.

    Args:
        enhancement_amount (float, optional): Controls the amount of the enhancement,
            in the range of [0, 100]. Default: ``75.0``. Note that `enhancement_amount` equal
            to 0 still gives a significant contrast enhancement.

    Raises:
        TypeError: If `enhancement_amount` is not of type float.
        ValueError: If `enhancement_amount` is not in range [0, 100].
        RuntimeError: If input tensor is not in the shape of <..., time>.

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
        >>> transforms = [audio.Contrast()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (16,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([16])  # 1 sample
        >>> output = audio.Contrast()(waveform)
        >>> print(output.shape, output.dtype)
        (16,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_contrast
    def __init__(self, enhancement_amount=75.0):
        super().__init__()
        self.enhancement_amount = enhancement_amount

    def parse(self):
        return cde.ContrastOperation(self.enhancement_amount)


class DCShift(AudioTensorOperation):
    """
    Apply a DC shift to the audio. This can be useful to remove DC offset from audio.

    Args:
        shift (float): The amount to shift the audio, the value must be in the range [-2.0, 2.0].
        limiter_gain (float, optional): Used only on peaks to prevent clipping,
            the value should be much less than 1, such as ``0.05`` or ``0.02``. Default: ``None``,
            will be set to `shift` .

    Raises:
        TypeError: If `shift` is not of type float.
        ValueError: If `shift` is not in range [-2.0, 2.0].
        TypeError: If `limiter_gain` is not of type float.

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
        >>> transforms = [audio.DCShift(0.5, 0.02)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (16,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([16])  # 1 sample
        >>> output = audio.DCShift(0.5, 0.02)(waveform)
        >>> print(output.shape, output.dtype)
        (16,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_dc_shift
    def __init__(self, shift, limiter_gain=None):
        super().__init__()
        self.shift = shift
        self.limiter_gain = limiter_gain if limiter_gain else shift

    def parse(self):
        return cde.DCShiftOperation(self.shift, self.limiter_gain)


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


class Dither(AudioTensorOperation):
    """
    Dither increases the perceived dynamic range of audio stored at a
    particular bit-depth by eliminating nonlinear truncation distortion.

    Args:
        density_function (DensityFunction, optional): The density function of a continuous
            random variable, can be ``DensityFunction.TPDF`` (Triangular Probability Density Function),
            ``DensityFunction.RPDF`` (Rectangular Probability Density Function) or
            ``DensityFunction.GPDF`` (Gaussian Probability Density Function).
            Default: ``DensityFunction.TPDF``.
        noise_shaping (bool, optional): A filtering process that shapes the spectral
            energy of quantisation error. Default: ``False``.

    Raises:
        TypeError: If `density_function` is not of type :class:`mindspore.dataset.audio.DensityFunction` .
        TypeError: If `noise_shaping` is not of type bool.
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
        >>> transforms = [audio.Dither()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (16,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([16])  # 1 sample
        >>> output = audio.Dither()(waveform)
        >>> print(output.shape, output.dtype)
        (16,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_dither
    def __init__(self, density_function=DensityFunction.TPDF, noise_shaping=False):
        super().__init__()
        self.density_function = density_function
        self.noise_shaping = noise_shaping

    def parse(self):
        return cde.DitherOperation(DE_C_DENSITY_FUNCTION.get(self.density_function), self.noise_shaping)


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


class Fade(AudioTensorOperation):
    """
    Add a fade in and/or fade out to a waveform.

    Args:
        fade_in_len (int, optional): Length of fade-in (time frames), which must be non-negative. Default: ``0``.
        fade_out_len (int, optional): Length of fade-out (time frames), which must be non-negative. Default: ``0``.
        fade_shape (FadeShape, optional): Shape of fade, five different types can be chosen as defined in FadeShape.
            Default: ``FadeShape.LINEAR``.

            - ``FadeShape.QUARTER_SINE``, means it tends to 0 in a quarter sin function.

            - ``FadeShape.HALF_SINE``, means it tends to 0 in a half sin function.

            - ``FadeShape.LINEAR``, means it goes linear to 0.

            - ``FadeShape.LOGARITHMIC``, means it tends to 0 in a logarithmic function.

            - ``FadeShape.EXPONENTIAL``, means it tends to 0 in an exponential function.

    Raises:
        RuntimeError: If fade_in_len exceeds waveform length.
        RuntimeError: If fade_out_len exceeds waveform length.

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
        >>> transforms = [audio.Fade(fade_in_len=3, fade_out_len=2, fade_shape=audio.FadeShape.LINEAR)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (16,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([16])  # 1 sample
        >>> output = audio.Fade(fade_in_len=3, fade_out_len=2, fade_shape=audio.FadeShape.LINEAR)(waveform)
        >>> print(output.shape, output.dtype)
        (16,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_fade
    def __init__(self, fade_in_len=0, fade_out_len=0, fade_shape=FadeShape.LINEAR):
        super().__init__()
        self.fade_in_len = fade_in_len
        self.fade_out_len = fade_out_len
        self.fade_shape = fade_shape

    def parse(self):
        return cde.FadeOperation(self.fade_in_len, self.fade_out_len, DE_C_FADE_SHAPE.get(self.fade_shape))


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


class Flanger(AudioTensorOperation):
    """
    Apply a flanger effect to the audio.

    Similar to the `SoX <https://sourceforge.net/projects/sox/>`_ implementation.

    Args:
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz).
        delay (float, optional): Desired delay in milliseconds, in range of [0, 30]. Default: ``0.0``.
        depth (float, optional): Desired delay depth in milliseconds, in range of [0, 10]. Default: ``2.0``.
        regen (float, optional): Desired regen (feedback gain) in dB, in range of [-95, 95]. Default: ``0.0``.
        width (float, optional): Desired width (delay gain) in dB, in range of [0, 100]. Default: ``71.0``.
        speed (float, optional): Modulation speed in Hz, in range of [0.1, 10]. Default: ``0.5``.
        phase (float, optional): Percentage phase-shift for multi-channel, in range of [0, 100]. Default: ``25.0``.
        modulation (Modulation, optional): Modulation method, can be ``Modulation.SINUSOIDAL`` or
            ``Modulation.TRIANGULAR``. Default: ``Modulation.SINUSOIDAL``.
        interpolation (Interpolation, optional): Interpolation method, can be ``Interpolation.LINEAR`` or
            ``Interpolation.QUADRATIC``. Default: ``Interpolation.LINEAR``.

    Raises:
        TypeError: If `sample_rate` is not of type int.
        ValueError: If `sample_rate` is zero.
        TypeError: If `delay` is not of type float.
        ValueError: If `delay` is not in range of [0, 30].
        TypeError: If `depth` is not of type float.
        ValueError: If `depth` is not in range of [0, 10].
        TypeError: If `regen` is not of type float.
        ValueError: If `regen` is not in range of [-95, 95].
        TypeError: If `width` is not of type float.
        ValueError: If `width` is not in range of [0, 100].
        TypeError: If `speed` is not of type float.
        ValueError: If `speed` is not in range of [0.1, 10].
        TypeError: If `phase` is not of type float.
        ValueError: If `phase` is not in range of [0, 100].
        TypeError: If `modulation` is not of type :class:`mindspore.dataset.audio.Modulation` .
        TypeError: If `interpolation` is not of type :class:`mindspore.dataset.audio.Interpolation` .
        RuntimeError: If input tensor is not in shape of <..., channel, time>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 4, 16])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.Flanger(44100)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (4, 16) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([4, 16])  # 1 sample
        >>> output = audio.Flanger(44100)(waveform)
        >>> print(output.shape, output.dtype)
        (4, 16) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_flanger
    def __init__(self, sample_rate, delay=0.0, depth=2.0, regen=0.0, width=71.0, speed=0.5, phase=25.0,
                 modulation=Modulation.SINUSOIDAL, interpolation=Interpolation.LINEAR):
        super().__init__()
        self.sample_rate = sample_rate
        self.delay = delay
        self.depth = depth
        self.regen = regen
        self.width = width
        self.speed = speed
        self.phase = phase
        self.modulation = modulation
        self.interpolation = interpolation

    def parse(self):
        return cde.FlangerOperation(self.sample_rate, self.delay, self.depth, self.regen, self.width, self.speed,
                                    self.phase, DE_C_MODULATION.get(self.modulation),
                                    DE_C_INTERPOLATION.get(self.interpolation))


class Gain(AudioTensorOperation):
    """
    Apply amplification or attenuation to the whole waveform.

    Args:
        gain_db (float, optional): Gain adjustment in decibels (dB). Default: ``1.0``.

    Raises:
        TypeError: If `gain_db` is not of type float.

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
        >>> transforms = [audio.Gain(1.2)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (8,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([8])  # 1 sample
        >>> output = audio.Gain(1.2)(waveform)
        >>> print(output.shape, output.dtype)
        (8,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_gain
    def __init__(self, gain_db=1.0):
        super().__init__()
        self.gain_db = gain_db

    def parse(self):
        return cde.GainOperation(self.gain_db)


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


class MuLawDecoding(AudioTensorOperation):
    """
    Decode mu-law encoded signal, refer to `mu-law algorithm <https://en.wikipedia.org/wiki/M-law_algorithm>`_ .

    Args:
        quantization_channels (int, optional): Number of channels, which must be positive. Default: ``256``.

    Raises:
        TypeError: If `quantization_channels` is not of type int.
        ValueError: If `quantization_channels` is not a positive number.
        RuntimeError: If input tensor is not in shape of <..., time>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 3, 4])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.MuLawDecoding()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (3, 4) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([3, 4])  # 1 sample
        >>> output = audio.MuLawDecoding()(waveform)
        >>> print(output.shape, output.dtype)
        (3, 4) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_mu_law_coding
    def __init__(self, quantization_channels=256):
        super().__init__()
        self.quantization_channels = quantization_channels

    def parse(self):
        return cde.MuLawDecodingOperation(self.quantization_channels)


class MuLawEncoding(AudioTensorOperation):
    """
    Encode signal based on mu-law companding.

    Args:
        quantization_channels (int, optional): Number of channels, which must be positive. Default: ``256``.

    Raises:
        TypeError: If `quantization_channels` is not of type int.
        ValueError: If `quantization_channels` is not a positive number.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 3, 4])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.MuLawEncoding()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (3, 4) int32
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([3, 4])  # 1 sample
        >>> output = audio.MuLawEncoding()(waveform)
        >>> print(output.shape, output.dtype)
        (3, 4) int32

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_mu_law_coding
    def __init__(self, quantization_channels=256):
        super().__init__()
        self.quantization_channels = quantization_channels

    def parse(self):
        return cde.MuLawEncodingOperation(self.quantization_channels)


class Overdrive(AudioTensorOperation):
    """
    Apply an overdrive effect to the audio waveform.

    Similar to the `SoX <https://sourceforge.net/projects/sox/>`_ implementation.

    Args:
        gain (float, optional): Desired gain at the boost (or attenuation) in dB, in range of [0, 100].
            Default: ``20.0``.
        color (float, optional): Controls the amount of even harmonic content in the over-driven output,
            in range of [0, 100]. Default: ``20.0``.

    Raises:
        TypeError: If `gain` is not of type float.
        ValueError: If `gain` is not in range of [0, 100].
        TypeError: If `color` is not of type float.
        ValueError: If `color` is not in range of [0, 100].
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
        >>> transforms = [audio.Overdrive()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (10,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([10])  # 1 sample
        >>> output = audio.Overdrive()(waveform)
        >>> print(output.shape, output.dtype)
        (10,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_overdrive
    def __init__(self, gain=20.0, color=20.0):
        super().__init__()
        self.gain = gain
        self.color = color

    def parse(self):
        return cde.OverdriveOperation(self.gain, self.color)


class Phaser(AudioTensorOperation):
    """
    Apply a phasing effect to the audio.

    Similar to the `SoX <https://sourceforge.net/projects/sox/>`_ implementation.

    Args:
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz).
        gain_in (float, optional): Desired input gain at the boost (or attenuation) in dB,
            in range of [0.0, 1.0]. Default: ``0.4``.
        gain_out (float, optional): Desired output gain at the boost (or attenuation) in dB,
            in range of [0.0, 1e9]. Default: ``0.74``.
        delay_ms (float, optional): Desired delay in milliseconds, in range of [0.0, 5.0]. Default: ``3.0``.
        decay (float, optional): Desired decay relative to gain-in, in range of [0.0, 0.99]. Default: ``0.4``.
        mod_speed (float, optional): Modulation speed in Hz, in range of [0.1, 2.0]. Default: ``0.5``.
        sinusoidal (bool, optional): If ``True``, use sinusoidal modulation (preferable for multiple instruments).
            If ``False``, use triangular modulation (gives single instruments a sharper phasing effect).
            Default: ``True``.

    Raises:
        TypeError: If `sample_rate` is not of type int.
        TypeError: If `gain_in` is not of type float.
        ValueError: If `gain_in` is not in range of [0.0, 1.0].
        TypeError: If `gain_out` is not of type float.
        ValueError: If `gain_out` is not in range of [0.0, 1e9].
        TypeError: If `delay_ms` is not of type float.
        ValueError: If `delay_ms` is not in range of [0.0, 5.0].
        TypeError: If `decay` is not of type float.
        ValueError: If `decay` is not in range of [0.0, 0.99].
        TypeError: If `mod_speed` is not of type float.
        ValueError: If `mod_speed` is not in range of [0.1, 2.0].
        TypeError: If `sinusoidal` is not of type bool.
        RuntimeError: If input tensor is not in shape of <..., time>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 12])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.Phaser(44100)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (12,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([12])  # 1 sample
        >>> output = audio.Phaser(44100)(waveform)
        >>> print(output.shape, output.dtype)
        (12,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_phaser
    def __init__(self, sample_rate, gain_in=0.4, gain_out=0.74, delay_ms=3.0, decay=0.4, mod_speed=0.5,
                 sinusoidal=True):
        super().__init__()
        self.decay = decay
        self.delay_ms = delay_ms
        self.gain_in = gain_in
        self.gain_out = gain_out
        self.mod_speed = mod_speed
        self.sample_rate = sample_rate
        self.sinusoidal = sinusoidal

    def parse(self):
        return cde.PhaserOperation(self.sample_rate, self.gain_in, self.gain_out,
                                   self.delay_ms, self.decay, self.mod_speed, self.sinusoidal)


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


class Vad(AudioTensorOperation):
    """
    Voice activity detector.

    Attempt to trim silence and quiet background sounds from the ends of recordings of speech.

    Similar to the `SoX <https://sourceforge.net/projects/sox/>`_ implementation.

    Args:
        sample_rate (int): Sampling rate of audio signal.
        trigger_level (float, optional): The measurement level used to trigger activity detection. Default: ``7.0``.
        trigger_time (float, optional): The time constant (in seconds) used to help ignore short bursts of
            sounds. Default: ``0.25``.
        search_time (float, optional): The amount of audio (in seconds) to search for quieter/shorter bursts of audio
            to include prior to the detected trigger point. Default: ``1.0``.
        allowed_gap (float, optional): The allowed gap (in seconds) between quieter/shorter bursts of audio to include
            prior to the detected trigger point. Default: ``0.25``.
        pre_trigger_time (float, optional): The amount of audio (in seconds) to preserve before the trigger point and
            any found quieter/shorter bursts. Default: ``0.0``.
        boot_time (float, optional): The time for the initial noise estimate. Default: ``0.35``.
        noise_up_time (float, optional): Time constant used by the adaptive noise estimator for when the noise level is
            increasing. Default: ``0.1``.
        noise_down_time (float, optional): Time constant used by the adaptive noise estimator for when the noise level
            is decreasing. Default: ``0.01``.
        noise_reduction_amount (float, optional): Amount of noise reduction to use in the detection algorithm.
            Default: ``1.35``.
        measure_freq (float, optional): Frequency of the algorithm's processing/measurements. Default: ``20.0``.
        measure_duration (float, optional): The duration of measurement. Default: ``None``,
            will use twice the measurement period.
        measure_smooth_time (float, optional): Time constant used to smooth spectral measurements. Default: ``0.4``.
        hp_filter_freq (float, optional): The 'Brick-wall' frequency of high-pass filter applied at the input to the
            detector algorithm. Default: ``50.0``.
        lp_filter_freq (float, optional): The 'Brick-wall' frequency of low-pass filter applied at the input to the
            detector algorithm. Default: ``6000.0``.
        hp_lifter_freq (float, optional): The 'Brick-wall' frequency of high-pass lifter used in the
            detector algorithm. Default: ``150.0``.
        lp_lifter_freq (float, optional): The 'Brick-wall' frequency of low-pass lifter used in the
            detector algorithm. Default: ``2000.0``.

    Raises:
        TypeError: If `sample_rate` is not of type int.
        ValueError: If `sample_rate` is not a positive number.
        TypeError: If `trigger_level` is not of type float.
        TypeError: If `trigger_time` is not of type float.
        ValueError: If `trigger_time` is a negative number.
        TypeError: If `search_time` is not of type float.
        ValueError: If `search_time` is a negative number.
        TypeError: If `allowed_gap` is not of type float.
        ValueError: If `allowed_gap` is a negative number.
        TypeError: If `pre_trigger_time` is not of type float.
        ValueError: If `pre_trigger_time` is a negative number.
        TypeError: If `boot_time` is not of type float.
        ValueError: If `boot_time` is a negative number.
        TypeError: If `noise_up_time` is not of type float.
        ValueError: If `noise_up_time` is a negative number.
        TypeError: If `noise_down_time` is not of type float.
        ValueError: If `noise_down_time` is a negative number.
        ValueError: If `noise_up_time` is less than `noise_down_time` .
        TypeError: If `noise_reduction_amount` is not of type float.
        ValueError: If `noise_reduction_amount` is a negative number.
        TypeError: If `measure_freq` is not of type float.
        ValueError: If `measure_freq` is not a positive number.
        TypeError: If `measure_duration` is not of type float.
        ValueError: If `measure_duration` is a negative number.
        TypeError: If `measure_smooth_time` is not of type float.
        ValueError: If `measure_smooth_time` is a negative number.
        TypeError: If `hp_filter_freq` is not of type float.
        ValueError: If `hp_filter_freq` is not a positive number.
        TypeError: If `lp_filter_freq` is not of type float.
        ValueError: If `lp_filter_freq` is not a positive number.
        TypeError: If `hp_lifter_freq` is not of type float.
        ValueError: If `hp_lifter_freq` is not a positive number.
        TypeError: If `lp_lifter_freq` is not of type float.
        ValueError: If `lp_lifter_freq` is not a positive number.
        RuntimeError: If input tensor is not in shape of <..., time>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 1000])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.Vad(sample_rate=600)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (660,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([1000])  # 1 sample
        >>> output = audio.Vad(sample_rate=600)(waveform)
        >>> print(output.shape, output.dtype)
        (660,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_vad
    def __init__(self, sample_rate, trigger_level=7.0, trigger_time=0.25, search_time=1.0, allowed_gap=0.25,
                 pre_trigger_time=0.0, boot_time=0.35, noise_up_time=0.1, noise_down_time=0.01,
                 noise_reduction_amount=1.35, measure_freq=20.0, measure_duration=None, measure_smooth_time=0.4,
                 hp_filter_freq=50.0, lp_filter_freq=6000.0, hp_lifter_freq=150.0, lp_lifter_freq=2000.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.trigger_level = trigger_level
        self.trigger_time = trigger_time
        self.search_time = search_time
        self.allowed_gap = allowed_gap
        self.pre_trigger_time = pre_trigger_time
        self.boot_time = boot_time
        self.noise_up_time = noise_up_time
        self.noise_down_time = noise_down_time
        self.noise_reduction_amount = noise_reduction_amount
        self.measure_freq = measure_freq
        self.measure_duration = measure_duration if measure_duration else 2.0 / measure_freq
        self.measure_smooth_time = measure_smooth_time
        self.hp_filter_freq = hp_filter_freq
        self.lp_filter_freq = lp_filter_freq
        self.hp_lifter_freq = hp_lifter_freq
        self.lp_lifter_freq = lp_lifter_freq

    def parse(self):
        return cde.VadOperation(self.sample_rate, self.trigger_level, self.trigger_time, self.search_time,
                                self.allowed_gap, self.pre_trigger_time, self.boot_time, self.noise_up_time,
                                self.noise_down_time, self.noise_reduction_amount, self.measure_freq,
                                self.measure_duration, self.measure_smooth_time, self.hp_filter_freq,
                                self.lp_filter_freq, self.hp_lifter_freq, self.lp_lifter_freq)


DE_C_GAIN_TYPE = {GainType.AMPLITUDE: cde.GainType.DE_GAIN_TYPE_AMPLITUDE,
                  GainType.POWER: cde.GainType.DE_GAIN_TYPE_POWER,
                  GainType.DB: cde.GainType.DE_GAIN_TYPE_DB}


class Vol(AudioTensorOperation):
    """
    Adjust volume of waveform.

    Args:
        gain (float): Gain at the boost (or attenuation).
            If `gain_type` is ``GainType.AMPLITUDE``, it is a non-negative amplitude ratio.
            If `gain_type` is ``GainType.POWER``, it is a power (voltage squared).
            If `gain_type` is ``GainType.DB``, it is in decibels.
        gain_type (GainType, optional): Type of gain, can be ``GainType.AMPLITUDE``, ``GainType.POWER``
            or ``GainType.DB``. Default: ``GainType.AMPLITUDE``.

    Raises:
        TypeError: If `gain` is not of type float.
        TypeError: If `gain_type` is not of type :class:`mindspore.dataset.audio.GainType` .
        ValueError: If `gain` is a negative number when `gain_type` is ``GainType.AMPLITUDE``.
        ValueError: If `gain` is not a positive number when `gain_type` is ``GainType.POWER``.
        RuntimeError: If input tensor is not in shape of <..., time>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.audio as audio
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> waveform = np.random.random([5, 30])  # 5 samples
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.Vol(gain=10, gain_type=audio.GainType.DB)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["audio"].shape, item["audio"].dtype)
        ...     break
        (30,) float64
        >>>
        >>> # Use the transform in eager mode
        >>> waveform = np.random.random([30])  # 1 sample
        >>> output = audio.Vol(gain=10, gain_type=audio.GainType.DB)(waveform)
        >>> print(output.shape, output.dtype)
        (30,) float64

    Tutorial Examples:
        - `Illustration of audio transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_
    """

    @check_vol
    def __init__(self, gain, gain_type=GainType.AMPLITUDE):
        super().__init__()
        self.gain = gain
        self.gain_type = gain_type

    def parse(self):
        return cde.VolOperation(self.gain, DE_C_GAIN_TYPE.get(self.gain_type))
