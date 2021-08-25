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
from .utils import ScaleType
from .validators import check_allpass_biquad, check_amplitude_to_db, check_band_biquad, check_bandpass_biquad, \
    check_bandreject_biquad, check_bass_biquad, check_complex_norm, check_masking, check_time_stretch


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
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz), the value must be greater than 0.
        central_freq (float): central frequency (in Hz), the value must be greater than 0.
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
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz).
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
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz), the value must be greater than 0.
        central_freq (float): central frequency (in Hz), the value must be greater than 0.
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
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz).
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
        >>> freq = 44100
        >>> num_frame = 30
        >>> def gen():
        ...     np.random.seed(0)
        ...     data =  np.random.random([freq, num_frame])
        ...     yield (np.array(data, dtype=np.float32), )
        >>> data1 = ds.GeneratorDataset(source=gen, column_names=["multi_dimensional_data"])
        >>> transforms = [py_audio.TimeStretch()]
        >>> data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
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
