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
The module audio.transforms is inherited from _c_dataengine and is
implemented based on C++. It's a high performance module to process
audio. Users can apply suitable augmentations on audio data to improve
their training models.
"""

import numpy as np

import mindspore._c_dataengine as cde
from ..transforms.c_transforms import TensorOperation
from .utils import FadeShape, GainType, ScaleType
from .validators import check_allpass_biquad, check_amplitude_to_db, check_band_biquad, check_bandpass_biquad, \
    check_bandreject_biquad, check_bass_biquad, check_biquad, check_complex_norm, check_contrast, check_dc_shift, \
    check_deemph_biquad, check_equalizer_biquad, check_fade, check_highpass_biquad, check_lfilter, \
    check_lowpass_biquad, check_magphase, check_masking, check_mu_law_decoding, check_time_stretch, check_vol


class AudioTensorOperation(TensorOperation):
    """
    Base class of Audio Tensor Ops
    """

    def __call__(self, *input_tensor_list):
        for tensor in input_tensor_list:
            if not isinstance(tensor, (np.ndarray,)):
                raise TypeError("Input should be NumPy audio, got {}.".format(type(tensor)))
        return super().__call__(*input_tensor_list)

    def parse(self):
        raise NotImplementedError("AudioTensorOperation has to implement parse() method.")


class AllpassBiquad(AudioTensorOperation):
    """
    Design two-pole all-pass filter for audio waveform of dimension of (..., time).

    Args:
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
        central_freq (float): central frequency (in Hz).
        Q(float, optional): Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (default=0.707).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.AllpassBiquad(44100, 200.0)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_allpass_biquad
    def __init__(self, sample_rate, central_freq, Q=0.707):
        self.sample_rate = sample_rate
        self.central_freq = central_freq
        self.Q = Q

    def parse(self):
        return cde.AllpassBiquadOperation(self.sample_rate, self.central_freq, self.Q)


DE_C_SCALETYPE_TYPE = {ScaleType.MAGNITUDE: cde.ScaleType.DE_SCALETYPE_MAGNITUDE,
                       ScaleType.POWER: cde.ScaleType.DE_SCALETYPE_POWER}


class AmplitudeToDB(AudioTensorOperation):
    """
    Converts the input tensor from amplitude/power scale to decibel scale.

    Args:
        stype (ScaleType, optional): Scale of the input tensor (default=ScaleType.POWER).
            It can be one of ScaleType.MAGNITUDE or ScaleType.POWER.
        ref_value (float, optional): Param for generate db_multiplier.
        amin (float, optional): Lower bound to clamp the input waveform. It must be greater than zero.
        top_db (float, optional): Minimum cut-off decibels. The range of values is non-negative.
            Commonly set at 80 (default=80.0).
    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.random.random([1, 400//2+1, 30])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.AmplitudeToDB(stype=ScaleType.POWER)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_amplitude_to_db
    def __init__(self, stype=ScaleType.POWER, ref_value=1.0, amin=1e-10, top_db=80.0):
        self.stype = stype
        self.ref_value = ref_value
        self.amin = amin
        self.top_db = top_db

    def parse(self):
        return cde.AmplitudeToDBOperation(DE_C_SCALETYPE_TYPE[self.stype], self.ref_value, self.amin, self.top_db)


class Angle(AudioTensorOperation):
    """
    Calculate the angle of the complex number sequence of shape (..., 2).
    The first dimension represents the real part while the second represents the imaginary.

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[1.43, 5.434], [23.54, 89.38]])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.Angle()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    def parse(self):
        return cde.AngleOperation()


class BandBiquad(AudioTensorOperation):
    """
    Design two-pole band filter for audio waveform of dimension of (..., time).

    Args:
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
        central_freq (float): Central frequency (in Hz).
        Q(float, optional): Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (default=0.707).
        noise (bool, optional) : If True, uses the alternate mode for un-pitched audio (e.g. percussion).
            If False, uses mode oriented to pitched audio, i.e. voice, singing, or instrumental music (default=False).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.BandBiquad(44100, 200.0)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_band_biquad
    def __init__(self, sample_rate, central_freq, Q=0.707, noise=False):
        self.sample_rate = sample_rate
        self.central_freq = central_freq
        self.Q = Q
        self.noise = noise

    def parse(self):
        return cde.BandBiquadOperation(self.sample_rate, self.central_freq, self.Q, self.noise)


class BandpassBiquad(AudioTensorOperation):
    """
    Design two-pole band-pass filter. Similar to SoX implementation.

    Args:
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
        central_freq (float): Central frequency (in Hz).
        Q (float, optional): Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0,1] (default=0.707).
        const_skirt_gain (bool, optional) : If True, uses a constant skirt gain (peak gain = Q).
            If False, uses a constant 0dB peak gain (default=False).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.BandpassBiquad(44100, 200.0)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_bandpass_biquad
    def __init__(self, sample_rate, central_freq, Q=0.707, const_skirt_gain=False):
        self.sample_rate = sample_rate
        self.central_freq = central_freq
        self.Q = Q
        self.const_skirt_gain = const_skirt_gain

    def parse(self):
        return cde.BandpassBiquadOperation(self.sample_rate, self.central_freq, self.Q, self.const_skirt_gain)


class BandrejectBiquad(AudioTensorOperation):
    """
    Design two-pole band filter for audio waveform of dimension of (..., time).

    Args:
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
        central_freq (float): central frequency (in Hz).
        Q(float, optional): Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (default=0.707).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03],[9.246826171875e-03, 1.0894775390625e-02]])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.BandrejectBiquad(44100, 200.0)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_bandreject_biquad
    def __init__(self, sample_rate, central_freq, Q=0.707):
        self.sample_rate = sample_rate
        self.central_freq = central_freq
        self.Q = Q

    def parse(self):
        return cde.BandrejectBiquadOperation(self.sample_rate, self.central_freq, self.Q)


class BassBiquad(AudioTensorOperation):
    """
    Design a bass tone-control effect for audio waveform of dimension of (..., time).

    Args:
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
        gain (float): Desired gain at the boost (or attenuation) in dB.
        central_freq (float): Central frequency (in Hz) (default=100.0).
        Q(float, optional): Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (default=0.707).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.BassBiquad(44100, 100.0)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_bass_biquad
    def __init__(self, sample_rate, gain, central_freq=100.0, Q=0.707):
        self.sample_rate = sample_rate
        self.gain = gain
        self.central_freq = central_freq
        self.Q = Q

    def parse(self):
        return cde.BassBiquadOperation(self.sample_rate, self.gain, self.central_freq, self.Q)


class Biquad(TensorOperation):
    """
    Perform a biquad filter of input tensor.

    Args:
        b0 (float): Numerator coefficient of current input, x[n].
        b1 (float): Numerator coefficient of input one time step ago x[n-1].
        b2 (float): Numerator coefficient of input two time steps ago x[n-2].
        a0 (float): Denominator coefficient of current output y[n], the value can't be zero, typically 1.
        a1 (float): Denominator coefficient of current output y[n-1].
        a2 (float): Denominator coefficient of current output y[n-2].

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
        >>> biquad_op = audio.Biquad(0.01, 0.02, 0.13, 1, 0.12, 0.3)
        >>> waveform_filtered = biquad_op(waveform)
    """
    @check_biquad
    def __init__(self, b0, b1, b2, a0, a1, a2):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2

    def parse(self):
        return cde.BiquadOperation(self.b0, self.b1, self.b2, self.a0, self.a1, self.a2)


class ComplexNorm(AudioTensorOperation):
    """
    Compute the norm of complex tensor input.

    Args:
        power (float, optional): Power of the norm, which must be non-negative (default=1.0).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.random.random([2, 4, 2])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.ComplexNorm()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """
    @check_complex_norm
    def __init__(self, power=1.0):
        self.power = power

    def parse(self):
        return cde.ComplexNormOperation(self.power)


class Contrast(AudioTensorOperation):
    """
    Apply contrast effect. Similar to SoX implementation.
    Comparable with compression, this effect modifies an audio signal to make it sound louder.

    Args:
        enhancement_amount (float): Controls the amount of the enhancement. Allowed range is [0, 100] (default=75.0).
            Note that enhancement_amount equal to 0 still gives a significant contrast enhancement.

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.Contrast()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_contrast
    def __init__(self, enhancement_amount=75.0):
        self.enhancement_amount = enhancement_amount

    def parse(self):
        return cde.ContrastOperation(self.enhancement_amount)


class DCShift(AudioTensorOperation):
    """
    Apply a DC shift to the audio.

    Args:
        shift (float): The amount to shift the audio, the value must be in the range [-2.0, 2.0].
        limiter_gain (float, optional): Used only on peaks to prevent clipping,
            the value should be much less than 1, such as 0.05 or 0.02.

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([0.60, 0.97, -1.04, -1.26, 0.97, 0.91, 0.48, 0.93])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.DCShift(0.5, 0.02)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operation=transforms, input_columns=["audio"])
    """

    @check_dc_shift
    def __init__(self, shift, limiter_gain=None):
        self.shift = shift
        self.limiter_gain = limiter_gain if limiter_gain else shift

    def parse(self):
        return cde.DCShiftOperation(self.shift, self.limiter_gain)


class DeemphBiquad(AudioTensorOperation):
    """
    Design two-pole deemph filter for audio waveform of dimension of (..., time).

    Args:
        Sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz),
            the value must be 44100 or 48000.

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.DeemphBiquad(44100)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """
    @check_deemph_biquad
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def parse(self):
        return cde.DeemphBiquadOperation(self.sample_rate)


class EqualizerBiquad(AudioTensorOperation):
    """
    Design biquad equalizer filter and perform filtering. Similar to SoX implementation.

    Args:
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
        center_freq (float): Central frequency (in Hz).
        gain (float): Desired gain at the boost (or attenuation) in dB.
        Q (float, optional): https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (default=0.707).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.EqualizerBiquad(44100, 1500, 5.5, 0.7)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_equalizer_biquad
    def __init__(self, sample_rate, center_freq, gain, Q=0.707):
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.gain = gain
        self.Q = Q

    def parse(self):
        return cde.EqualizerBiquadOperation(self.sample_rate, self.center_freq, self.gain, self.Q)


DE_C_FADESHAPE_TYPE = {FadeShape.LINEAR: cde.FadeShape.DE_FADESHAPE_LINEAR,
                       FadeShape.EXPONENTIAL: cde.FadeShape.DE_FADESHAPE_EXPONENTIAL,
                       FadeShape.LOGARITHMIC: cde.FadeShape.DE_FADESHAPE_LOGARITHMIC,
                       FadeShape.QUARTERSINE: cde.FadeShape.DE_FADESHAPE_QUARTERSINE,
                       FadeShape.HALFSINE: cde.FadeShape.DE_FADESHAPE_HALFSINE}


class Fade(AudioTensorOperation):
    """
    Add a fade in and/or fade out to an waveform.

    Args:
        fade_in_len (int, optional): Length of fade-in (time frames), which must be non-negative (default=0).
        fade_out_len (int, optional): Length of fade-out (time frames), which must be non-negative (default=0).
        fade_shape (FadeShape, optional): Shape of fade (default=FadeShape.LINEAR). Can be one of
            [FadeShape.LINEAR, FadeShape.EXPONENTIAL, FadeShape.LOGARITHMIC, FadeShape.QUARTERSINC, FadeShape.HALFSINC].

            -FadeShape.LINEAR, means it linear to 0.

            -FadeShape.EXPONENTIAL, means it tend to 0 in an exponential function.

            -FadeShape.LOGARITHMIC, means it tend to 0 in an logrithmic function.

            -FadeShape.QUARTERSINE, means it tend to 0 in an quarter sin function.

            -FadeShape.HALFSINE, means it tend to 0 in an half sin function.

    Raises:
        RuntimeError: If fade_in_len exceeds waveform length.
        RuntimeError: If fade_out_len exceeds waveform length.

    Examples:
          >>> import numpy as np
          >>>
          >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03, 9.246826171875e-03, 1.0894775390625e-02]])
          >>> dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
          >>> transforms = [audio.Fade(fade_in_len=3, fade_out_len=2, fade_shape=FadeShape.LINEAR)]
          >>> dataset = dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_fade
    def __init__(self, fade_in_len=0, fade_out_len=0, fade_shape=FadeShape.LINEAR):
        self.fade_in_len = fade_in_len
        self.fade_out_len = fade_out_len
        self.fade_shape = fade_shape

    def parse(self):
        return cde.FadeOperation(self.fade_in_len, self.fade_out_len, DE_C_FADESHAPE_TYPE[self.fade_shape])


class FrequencyMasking(AudioTensorOperation):
    """
    Apply masking to a spectrogram in the frequency domain.

    Args:
        iid_masks (bool, optional): Whether to apply different masks to each example (default=false).
        frequency_mask_param (int): Maximum possible length of the mask, range: [0, freq_length] (default=0).
            Indices uniformly sampled from [0, frequency_mask_param].
        mask_start (int): Mask start takes effect when iid_masks=true,
            range: [0, freq_length-frequency_mask_param] (default=0).
        mask_value (double): Mask value (default=0.0).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.random.random([1, 3, 2])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.FrequencyMasking(frequency_mask_param=1)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """
    @check_masking
    def __init__(self, iid_masks=False, frequency_mask_param=0, mask_start=0, mask_value=0.0):
        self.iid_masks = iid_masks
        self.frequency_mask_param = frequency_mask_param
        self.mask_start = mask_start
        self.mask_value = mask_value

    def parse(self):
        return cde.FrequencyMaskingOperation(self.iid_masks, self.frequency_mask_param, self.mask_start,
                                             self.mask_value)


class HighpassBiquad(AudioTensorOperation):
    """
    Design biquad highpass filter and perform filtering. Similar to SoX implementation.

    Args:
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
        cutoff_freq (float): Filter cutoff frequency (in Hz).
        Q (float, optional): Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (default=0.707).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.HighpassBiquad(44100, 1500, 0.7)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_highpass_biquad
    def __init__(self, sample_rate, cutoff_freq, Q=0.707):
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        self.Q = Q

    def parse(self):
        return cde.HighpassBiquadOperation(self.sample_rate, self.cutoff_freq, self.Q)


class LFilter(AudioTensorOperation):
    """
    Design two-pole filter for audio waveform of dimension of (..., time).

    Args:
        a_coeffs (sequence): denominator coefficients of difference equation of dimension of (n_order + 1).
            Lower delays coefficients are first, e.g. [a0, a1, a2, ...].
            Must be same size as b_coeffs (pad with 0's as necessary).
        b_coeffs (sequence): numerator coefficients of difference equation of dimension of (n_order + 1).
            Lower delays coefficients are first, e.g. [b0, b1, b2, ...].
            Must be same size as a_coeffs (pad with 0's as necessary).
        clamp (bool, optional): If True, clamp the output signal to be in the range [-1, 1] (default=True).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
        >>> a_coeffs = [0.1, 0.2, 0.3]
        >>> b_coeffs = [0.1, 0.2, 0.3]
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.LFilter(a_coeffs, b_coeffs)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """
    @check_lfilter
    def __init__(self, a_coeffs, b_coeffs, clamp=True):
        self.a_coeffs = a_coeffs
        self.b_coeffs = b_coeffs
        self.clamp = clamp

    def parse(self):
        return cde.LFilterOperation(self.a_coeffs, self.b_coeffs, self.clamp)


class LowpassBiquad(AudioTensorOperation):
    """
    Design biquad lowpass filter and perform filtering. Similar to SoX implementation.

    Args:
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
        cutoff_freq (float): Filter cutoff frequency.
        Q(float, optional): Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (default=0.707).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[0.8236, 0.2049, 0.3335], [0.5933, 0.9911, 0.2482],
        ...                      [0.3007, 0.9054, 0.7598], [0.5394, 0.2842, 0.5634], [0.6363, 0.2226, 0.2288]])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.LowpassBiquad(4000, 1500, 0.7)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """
    @check_lowpass_biquad
    def __init__(self, sample_rate, cutoff_freq, Q=0.707):
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        self.Q = Q

    def parse(self):
        return cde.LowpassBiquadOperation(self.sample_rate, self.cutoff_freq, self.Q)


class Magphase(AudioTensorOperation):
    """
    Separate a complex-valued spectrogram with shape (..., 2) into its magnitude and phase.

    Args:
        power (float): Power of the norm, which must be non-negative (default=1.0).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.random.random([2, 4, 2])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.Magphase()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_magphase
    def __init__(self, power=1.0):
        self.power = power

    def parse(self):
        return cde.MagphaseOperation(self.power)


class MuLawDecoding(AudioTensorOperation):
    """
    Decode mu-law encoded signal.

    Args:
        quantization_channels (int): Number of channels, which must be positive (Default: 256).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.random.random([1, 3, 4])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.MuLawDecoding()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """
    @check_mu_law_decoding
    def __init__(self, quantization_channels=256):
        self.quantization_channels = quantization_channels

    def parse(self):
        return cde.MuLawDecodingOperation(self.quantization_channels)


class TimeMasking(AudioTensorOperation):
    """
    Apply masking to a spectrogram in the time domain.

    Args:
        iid_masks (bool, optional): Whether to apply different masks to each example (default=false).
        time_mask_param (int): Maximum possible length of the mask, range: [0, time_length] (default=0).
            Indices uniformly sampled from [0, time_mask_param].
        mask_start (int): Mask start takes effect when iid_masks=true,
            range: [0, time_length-time_mask_param] (default=0).
        mask_value (double): Mask value (default=0.0).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.random.random([1, 3, 2])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.TimeMasking(time_mask_param=1)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_masking
    def __init__(self, iid_masks=False, time_mask_param=0, mask_start=0, mask_value=0.0):
        self.iid_masks = iid_masks
        self.time_mask_param = time_mask_param
        self.mask_start = mask_start
        self.mask_value = mask_value

    def parse(self):
        return cde.TimeMaskingOperation(self.iid_masks, self.time_mask_param, self.mask_start, self.mask_value)


class TimeStretch(AudioTensorOperation):
    """
    Stretch STFT in time at a given rate, without changing the pitch.

    Args:
        hop_length (int, optional): Length of hop between STFT windows (default=None).
        n_freq (int, optional): Number of filter banks form STFT (default=201).
        fixed_rate (float, optional): Rate to speed up or slow down the input in time (default=None).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.random.random([1, 30])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.TimeStretch()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """
    @check_time_stretch
    def __init__(self, hop_length=None, n_freq=201, fixed_rate=None):
        self.n_freq = n_freq
        self.fixed_rate = fixed_rate

        n_fft = (n_freq - 1) * 2
        self.hop_length = hop_length if hop_length is not None else n_fft // 2
        self.fixed_rate = fixed_rate if fixed_rate is not None else np.nan

    def parse(self):
        return cde.TimeStretchOperation(self.hop_length, self.n_freq, self.fixed_rate)


DE_C_GAINTYPE_TYPE = {GainType.AMPLITUDE: cde.GainType.DE_GAINTYPE_AMPLITUDE,
                      GainType.POWER: cde.GainType.DE_GAINTYPE_POWER,
                      GainType.DB: cde.GainType.DE_GAINTYPE_DB}


class Vol(AudioTensorOperation):
    """
    Apply amplification or attenuation to the whole waveform.

    Args:
        gain (float): Value of gain adjustment.
            If gain_type = amplitude, gain stands for nonnegative amplitude ratio.
            If gain_type = power, gain stands for power.
            If gain_type = db, gain stands for decibels.
        gain_type (ScaleType, optional): Type of gain, contains the following three enumeration values
            GainType.AMPLITUDE, GainType.POWER and GainType.DB (default=GainType.AMPLITUDE).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.random.random([20, 30])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.Vol(gain=10, gain_type=GainType.DB)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_vol
    def __init__(self, gain, gain_type=GainType.AMPLITUDE):
        self.gain = gain
        self.gain_type = gain_type

    def parse(self):
        return cde.VolOperation(self.gain, DE_C_GAINTYPE_TYPE[self.gain_type])
