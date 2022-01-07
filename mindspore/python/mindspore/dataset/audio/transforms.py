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
from ..transforms.c_transforms import TensorOperation
from .utils import BorderType, DensityFunction, FadeShape, GainType, Interpolation, Modulation, ScaleType, WindowType
from .validators import check_allpass_biquad, check_amplitude_to_db, check_band_biquad, check_bandpass_biquad, \
    check_bandreject_biquad, check_bass_biquad, check_biquad, check_complex_norm, check_compute_deltas, \
    check_contrast, check_db_to_amplitude, check_dc_shift, check_deemph_biquad, check_detect_pitch_frequency, \
    check_dither, check_equalizer_biquad, check_fade, check_flanger, check_gain, check_highpass_biquad, \
    check_lfilter, check_lowpass_biquad, check_magphase, check_masking, check_mu_law_coding, check_overdrive, \
    check_phaser, check_riaa_biquad, check_sliding_window_cmn, check_spectral_centroid, check_spectrogram, \
    check_time_stretch, check_treble_biquad, check_vol


class AudioTensorOperation(TensorOperation):
    """
    Base class of Audio Tensor Ops.
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
        ref_value (float, optional): Param for generate db_multiplier (default=1.0).
        amin (float, optional): Lower bound to clamp the input waveform. It must be greater than zero (default=1e-10).
        top_db (float, optional): Minimum cut-off decibels. The range of values is non-negative.
            Commonly set at 80 (default=80.0).
    Examples:
        >>> import numpy as np
        >>> from mindspore.dataset.audio import ScaleType
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
    Design two-pole band-reject filter for audio waveform of dimension of (..., time).

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


DE_C_BORDER_TYPE = {
    BorderType.CONSTANT: cde.BorderType.DE_BORDER_CONSTANT,
    BorderType.EDGE: cde.BorderType.DE_BORDER_EDGE,
    BorderType.REFLECT: cde.BorderType.DE_BORDER_REFLECT,
    BorderType.SYMMETRIC: cde.BorderType.DE_BORDER_SYMMETRIC,
}


class ComputeDeltas(AudioTensorOperation):
    """
    Compute delta coefficients of a spectrogram.

    Args:
        win_length (int): The window length used for computing delta, must be no less than 3 (default=5).
        pad_mode (BorderType): Mode parameter passed to padding (default=BorderType.EDGE).It can be any of
            [BorderType.CONSTANT, BorderType.EDGE, BorderType.REFLECT, BordBorderTypeer.SYMMETRIC].

            - BorderType.CONSTANT, means it fills the border with constant values.

            - BorderType.EDGE, means it pads with the last value on the edge.

            - BorderType.REFLECT, means it reflects the values on the edge omitting the last
              value of edge.

            - BorderType.SYMMETRIC, means it reflects the values on the edge repeating the last
              value of edge.

    Examples:
        >>> import numpy as np
        >>> from mindspore.dataset.audio import BorderType
        >>>
        >>> waveform = np.random.random([1, 400//2+1, 30])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.ComputeDeltas(win_length=7, pad_mode = BorderType.EDGE)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_compute_deltas
    def __init__(self, win_length=5, pad_mode=BorderType.EDGE):
        self.win_len = win_length
        self.pad_mode = pad_mode

    def parse(self):
        return cde.ComputeDeltasOperation(self.win_len, DE_C_BORDER_TYPE[self.pad_mode])


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


class DBToAmplitude(AudioTensorOperation):
    """
    Turn a waveform from the decibel scale to the power/amplitude scale.

    Args:
        ref (float): Reference which the output will be scaled by.
        power (float): If power equals 1, will compute DB to power. If 0.5, will compute DB to amplitude.

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.DBToAmplitude(0.5, 0.5)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_db_to_amplitude
    def __init__(self, ref, power):
        self.ref = ref
        self.power = power

    def parse(self):
        return cde.DBToAmplitudeOperation(self.ref, self.power)


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
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz),
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


class DetectPitchFrequency(AudioTensorOperation):
    """
    Detect pitch frequency.

    It is implemented using normalized cross-correlation function and median smoothing.

    Args:
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
        frame_time (float, optional): Duration of a frame, the value must be greater than zero (default=0.01).
        win_length (int, optional): The window length for median smoothing (in number of frames), the value must be
            greater than zero (default=30).
        freq_low (int, optional): Lowest frequency that can be detected (Hz), the value must be greater than zero
            (default=85).
        freq_high (int, optional): Highest frequency that can be detected (Hz), the value must be greater than zero
            (default=3400).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[0.716064e-03, 5.347656e-03, 6.246826e-03, 2.089477e-02, 7.138305e-02],
        ...                      [4.156616e-02, 1.394653e-02, 3.550292e-02, 0.614379e-02, 3.840209e-02]])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.DetectPitchFrequency(30, 0.1, 3, 5, 25)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_detect_pitch_frequency
    def __init__(self, sample_rate, frame_time=0.01, win_length=30, freq_low=85, freq_high=3400):
        self.sample_rate = sample_rate
        self.frame_time = frame_time
        self.win_length = win_length
        self.freq_low = freq_low
        self.freq_high = freq_high

    def parse(self):
        return cde.DetectPitchFrequencyOperation(self.sample_rate, self.frame_time,
                                                 self.win_length, self.freq_low, self.freq_high)


DE_C_DENSITYFUNCTION_TYPE = {DensityFunction.TPDF: cde.DensityFunction.DE_DENSITYFUNCTION_TPDF,
                             DensityFunction.RPDF: cde.DensityFunction.DE_DENSITYFUNCTION_RPDF,
                             DensityFunction.GPDF: cde.DensityFunction.DE_DENSITYFUNCTION_GPDF}


class Dither(AudioTensorOperation):
    """
    Dither increases the perceived dynamic range of audio stored at a
    particular bit-depth by eliminating nonlinear truncation distortion.

    Args:
        density_function (DensityFunction, optional): The density function of a continuous
            random variable. Can be one of DensityFunction.TPDF (Triangular Probability Density Function),
            DensityFunction.RPDF (Rectangular Probability Density Function) or
            DensityFunction.GPDF (Gaussian Probability Density Function)
            (default=DensityFunction.TPDF).
        noise_shaping (bool, optional): A filtering process that shapes the spectral
            energy of quantisation error (default=False).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[1, 2, 3], [4, 5, 6]])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.Dither()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_dither
    def __init__(self, density_function=DensityFunction.TPDF, noise_shaping=False):
        self.density_function = density_function
        self.noise_shaping = noise_shaping

    def parse(self):
        return cde.DitherOperation(DE_C_DENSITYFUNCTION_TYPE[self.density_function], self.noise_shaping)


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
        >>> from mindspore.dataset.audio import FadeShape
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


DE_C_MODULATION_TYPE = {Modulation.SINUSOIDAL: cde.Modulation.DE_MODULATION_SINUSOIDAL,
                        Modulation.TRIANGULAR: cde.Modulation.DE_MODULATION_TRIANGULAR}

DE_C_INTERPOLATION_TYPE = {Interpolation.LINEAR: cde.Interpolation.DE_INTERPOLATION_LINEAR,
                           Interpolation.QUADRATIC: cde.Interpolation.DE_INTERPOLATION_QUADRATIC}


class Flanger(AudioTensorOperation):
    """
    Apply a flanger effect to the audio.

    Args:
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz).
        delay (float, optional): Desired delay in milliseconds (ms), range: [0, 30] (default=0.0).
        depth (float, optional): Desired delay depth in milliseconds (ms), range: [0, 10] (default=2.0).
        regen (float, optional): Desired regen (feedback gain) in dB, range: [-95, 95] (default=0.0).
        width (float, optional): Desired width (delay gain) in dB, range: [0, 100] (default=71.0).
        speed (float, optional): Modulation speed in Hz, range: [0.1, 10] (default=0.5).
        phase (float, optional): Percentage phase-shift for multi-channel, range: [0, 100] (default=25.0).
        modulation (Modulation, optional): Modulation of the input tensor (default=Modulation.SINUSOIDAL).
            It can be one of Modulation.SINUSOIDAL or Modulation.TRIANGULAR.
        interpolation (Interpolation, optional): Interpolation of the input tensor (default=Interpolation.LINEAR).
            It can be one of Interpolation.LINEAR or Interpolation.QUADRATIC.

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.Flanger(44100)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_flanger
    def __init__(self, sample_rate, delay=0.0, depth=2.0, regen=0.0, width=71.0, speed=0.5,
                 phase=25.0, modulation=Modulation.SINUSOIDAL, interpolation=Interpolation.LINEAR):
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
                                    self.phase, DE_C_MODULATION_TYPE[self.modulation],
                                    DE_C_INTERPOLATION_TYPE[self.interpolation])


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


class Gain(AudioTensorOperation):
    """
    Apply amplification or attenuation to the whole waveform.

    Args:
        gain_db (float): Gain adjustment in decibels (dB) (default=1.0).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.Gain(1.2)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_gain
    def __init__(self, gain_db=1.0):
        self.gain_db = gain_db

    def parse(self):
        return cde.GainOperation(self.gain_db)


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

    @check_mu_law_coding
    def __init__(self, quantization_channels=256):
        self.quantization_channels = quantization_channels

    def parse(self):
        return cde.MuLawDecodingOperation(self.quantization_channels)


class MuLawEncoding(AudioTensorOperation):
    """
    Encode signal based on mu-law companding.

    Args:
        quantization_channels (int): Number of channels, which must be positive (Default: 256).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.random.random([1, 3, 4])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.MuLawEncoding()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_mu_law_coding
    def __init__(self, quantization_channels=256):
        self.quantization_channels = quantization_channels

    def parse(self):
        return cde.MuLawEncodingOperation(self.quantization_channels)


class Overdrive(AudioTensorOperation):
    """
    Apply overdrive on input audio.

    Args:
        gain (float): Desired gain at the boost (or attenuation) in dB, in range of [0, 100] (default=20.0).
        color (float): Controls the amount of even harmonic content in the over-driven output,
            in range of [0, 100] (default=20.0).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.Overdrive()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_overdrive
    def __init__(self, gain=20.0, color=20.0):
        self.gain = gain
        self.color = color

    def parse(self):
        return cde.OverdriveOperation(self.gain, self.color)


class Phaser(AudioTensorOperation):
    """
    Apply a phasing effect to the audio.

    Args:
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz).
        gain_in (float): Desired input gain at the boost (or attenuation) in dB.
            Allowed range of values is [0, 1] (default=0.4).
        gain_out (float): Desired output gain at the boost (or attenuation) in dB.
            Allowed range of values is [0, 1e9] (default=0.74).
        delay_ms (float): Desired delay in milli seconds. Allowed range of values is [0, 5] (default=3.0).
        decay (float): Desired decay relative to gain-in. Allowed range of values is [0, 0.99] (default=0.4).
        mod_speed (float): Modulation speed in Hz. Allowed range of values is [0.1, 2] (default=0.5).
        sinusoidal (bool): If True, use sinusoidal modulation (preferable for multiple instruments).
            If False, use triangular modulation (gives single instruments a sharper
            phasing effect) (default=True).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.Phaser(44100)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_phaser
    def __init__(self, sample_rate, gain_in=0.4, gain_out=0.74,
                 delay_ms=3.0, decay=0.4, mod_speed=0.5, sinusoidal=True):
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


class RiaaBiquad(AudioTensorOperation):
    """
    Apply RIAA vinyl playback equalization. Similar to SoX implementation.

    Args:
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz),
            can only be one of 44100, 48000, 88200, 96000.

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.RiaaBiquad(44100)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_riaa_biquad
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def parse(self):
        return cde.RiaaBiquadOperation(self.sample_rate)


class SlidingWindowCmn(AudioTensorOperation):
    """
    Apply sliding-window cepstral mean (and optionally variance) normalization per utterance.

    Args:
        cmn_window (int, optional): Window in frames for running average CMN computation (default=600).
        min_cmn_window (int, optional): Minimum CMN window used at start of decoding (adds latency only at start).
            Only applicable if center is False, ignored if center is True (default=100).
        center (bool, optional): If True, use a window centered on the current frame. If False, window is
            to the left. (default=False).
        norm_vars (bool, optional): If True, normalize variance to one. (default=False).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float64)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.SlidingWindowCmn()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_sliding_window_cmn
    def __init__(self, cmn_window=600, min_cmn_window=100, center=False, norm_vars=False):
        self.cmn_window = cmn_window
        self.min_cmn_window = min_cmn_window
        self.center = center
        self.norm_vars = norm_vars

    def parse(self):
        return cde.SlidingWindowCmnOperation(self.cmn_window, self.min_cmn_window, self.center, self.norm_vars)


DE_C_WINDOW_TYPE = {WindowType.BARTLETT: cde.WindowType.DE_BARTLETT,
                    WindowType.BLACKMAN: cde.WindowType.DE_BLACKMAN,
                    WindowType.HAMMING: cde.WindowType.DE_HAMMING,
                    WindowType.HANN: cde.WindowType.DE_HANN,
                    WindowType.KAISER: cde.WindowType.DE_KAISER}


class SpectralCentroid(TensorOperation):
    """
    Create a spectral centroid from an audio signal.

    Args:
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz).
        n_fft (int, optional): Size of FFT, creates n_fft // 2 + 1 bins (default=400).
        win_length (int, optional): Window size (default=None, will use n_fft).
        hop_length (int, optional): Length of hop between STFT windows (default=None, will use win_length // 2).
        pad (int, optional): Two sided padding of signal (default=0).
        window (WindowType, optional): Window function that is applied/multiplied to each frame/window,
            which can be WindowType.BARTLETT, WindowType.BLACKMAN, WindowType.HAMMING, WindowType.HANN
            or WindowType.KAISER (default=WindowType.HANN).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.random.random([5, 10, 20])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.SpectralCentroid(44100)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_spectral_centroid
    def __init__(self, sample_rate, n_fft=400, win_length=None, hop_length=None, pad=0, window=WindowType.HANN):
        self.sample_rate = sample_rate
        self.pad = pad
        self.window = window
        self.n_fft = n_fft
        self.win_length = win_length if win_length else n_fft
        self.hop_length = hop_length if hop_length else self.win_length // 2

    def parse(self):
        return cde.SpectralCentroidOperation(self.sample_rate, self.n_fft, self.win_length, self.hop_length,
                                             self.pad, DE_C_WINDOW_TYPE[self.window])


class Spectrogram(TensorOperation):
    """
    Create a spectrogram from an audio signal.

    Args:
        n_fft (int, optional): Size of FFT, creates n_fft // 2 + 1 bins (default=400).
        win_length (int, optional): Window size (default=None, will use n_fft).
        hop_length (int, optional): Length of hop between STFT windows (default=None, will use win_length // 2).
        pad (int): Two sided padding of signal (default=0).
        window (WindowType, optional): Window function that is applied/multiplied to each frame/window,
            which can be WindowType.BARTLETT, WindowType.BLACKMAN, WindowType.HAMMING, WindowType.HANN
            or WindowType.KAISER (default=WindowType.HANN). Currently kaiser window is not supported on macOS.
        power (float, optional): Exponent for the magnitude spectrogram, which must be greater
            than or equal to 0, e.g., 1 for energy, 2 for power, etc. (default=2.0).
        normalized (bool, optional): Whether to normalize by magnitude after stft (default=False).
        center (bool, optional): Whether to pad waveform on both sides (default=True).
        pad_mode (BorderType, optional): Controls the padding method used when center is True,
            which can be BorderType.REFLECT, BorderType.CONSTANT, BorderType.EDGE, BorderType.SYMMETRIC
            (default=BorderType.REFLECT).
        onesided (bool, optional): Controls whether to return half of results to avoid redundancy (default=True).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.random.random([5, 10, 20])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.Spectrogram()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_spectrogram
    def __init__(self, n_fft=400, win_length=None, hop_length=None, pad=0, window=WindowType.HANN, power=2.0,
                 normalized=False, center=True, pad_mode=BorderType.REFLECT, onesided=True):
        self.n_fft = n_fft
        self.win_length = win_length if win_length else n_fft
        self.hop_length = hop_length if hop_length else self.win_length // 2
        self.pad = pad
        self.window = window
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided

    def parse(self):
        return cde.SpectrogramOperation(self.n_fft, self.win_length, self.hop_length, self.pad,
                                        DE_C_WINDOW_TYPE[self.window], self.power, self.normalized,
                                        self.center, DE_C_BORDER_TYPE[self.pad_mode], self.onesided)


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
        hop_length (int, optional): Length of hop between STFT windows (default=None, will use ((n_freq - 1) * 2) // 2).
        n_freq (int, optional): Number of filter banks form STFT (default=201).
        fixed_rate (float, optional): Rate to speed up or slow down the input in time
            (default=None, will keep the original rate).

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
        self.fixed_rate = fixed_rate if fixed_rate is not None else 1

    def parse(self):
        return cde.TimeStretchOperation(self.hop_length, self.n_freq, self.fixed_rate)


class TrebleBiquad(AudioTensorOperation):
    """
    Design a treble tone-control effect. Similar to SoX implementation.

    Args:
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
        gain (float): Desired gain at the boost (or attenuation) in dB.
        central_freq (float, optional): Central frequency (in Hz) (default=3000).
        Q(float, optional): Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (default=0.707).

    Examples:
        >>> import numpy as np
        >>>
        >>> waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
        >>> transforms = [audio.TrebleBiquad(44100, 200.0)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @check_treble_biquad
    def __init__(self, sample_rate, gain, central_freq=3000, Q=0.707):
        self.sample_rate = sample_rate
        self.gain = gain
        self.central_freq = central_freq
        self.Q = Q

    def parse(self):
        return cde.TrebleBiquadOperation(self.sample_rate, self.gain, self.central_freq, self.Q)


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
        gain_type (GainType, optional): Type of gain, contains the following three enumeration values
            GainType.AMPLITUDE, GainType.POWER and GainType.DB (default=GainType.AMPLITUDE).

    Examples:
        >>> import numpy as np
        >>> from mindspore.dataset.audio import GainType
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
