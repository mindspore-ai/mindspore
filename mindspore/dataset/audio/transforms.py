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
"""
The module audio.transforms is inherited from _c_dataengine.
and is implemented based on  C++. It's a high performance module to
process audio. Users can apply suitable augmentations on audio data
to improve their training models.
"""
import mindspore._c_dataengine as cde
import numpy as np
from ..transforms.c_transforms import TensorOperation
from .utils import ScaleType
from .validators import check_allpass_biquad, check_amplitude_to_db, check_band_biquad, check_bandpass_biquad, \
    check_bandreject_biquad, check_bass_biquad, check_time_stretch


class AudioTensorOperation(TensorOperation):
    """
    Base class of Audio Tensor Ops
    """

    def __call__(self, *input_tensor_list):
        for tensor in input_tensor_list:
            if not isinstance(tensor, (np.ndarray,)):
                raise TypeError(
                    "Input should be NumPy audio, got {}.".format(type(tensor)))
        return super().__call__(*input_tensor_list)

    def parse(self):
        raise NotImplementedError(
            "AudioTensorOperation has to implement parse() method.")


class AllpassBiquad(AudioTensorOperation):
    """
    Design two-pole all-pass filter for audio waveform of dimension of `(..., time)`

        Args:
            sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz),
                the value must be greater than 0 .
            central_freq (float): central frequency (in Hz),
                the value must be greater than 0 .
            Q(float, optional): Quality factor,https://en.wikipedia.org/wiki/Q_factor,
                Range: (0, 1] (Default=0.707).

        Examples:
            >>> import mindspore.dataset.audio.transforms as audio
            >>> import numpy as np

            >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03],[9.246826171875e-03, 1.0894775390625e-02]])
            >>> allpasspass_biquad_op = audio.AllpassBiquad(44100, 200.0)
            >>> waveform_filtered = allpass_biquad_op(waveform)

        References:
            https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF
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
        stype (ScaleType, optional): Scale of the input tensor. (Default="ScaleType.POWER").
        It can be any of [ScaleType.MAGNITUDE, ScaleType.POWER].
        ref_value (float, optional): Param for generate db_multiplier.
        amin (float, optional): Lower bound to clamp the input waveform.
        top_db (float, optional): Minimum cut-off decibels. The range of values is non-negative. Commonly set at 80.
            (Default=80.0)
    Examples:
        >>> channel = 1
        >>> n_fft = 400
        >>> n_frame = 30
        >>> specrogram = np.random.random([channel, n_fft//2+1, n_frame])
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=specrogram, column_names=["audio"])
        >>> transforms = [audio.AmplitudeToDB(stype=ScaleType.POWER)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
    """

    @ check_amplitude_to_db
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
    Args:

    Examples:
        >>> import mindspore.dataset.audio.transforms as audio
        >>> import numpy as np

        >>> input_complex = np.array([[1.43, 5.434], [23.54, 89.38]])
        >>> angle_op = audio.Angle()
        >>> angles = angle_op(input_complex)
    """

    def parse(self):
        return cde.AngleOperation()


class BandBiquad(AudioTensorOperation):
    """
    Design two-pole band filter for audio waveform of dimension of `(..., time)`

    Args:
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
        central_freq (float): central frequency (in Hz),
        Q(float, optional): Quality factor, https://en.wikipedia.org/wiki/Q_factor, Range: (0, 1] (Default=0.707).
        noise (bool, optional) : If ``True``, uses the alternate mode for un-pitched audio (e.g. percussion).
            If ``False``, uses mode oriented to pitched audio, i.e. voice, singing,
            or instrumental music (Default: ``False``).

    Examples:
        >>> import mindspore.dataset.audio.transforms as audio
        >>> import numpy as np

        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03],[9.246826171875e-03, 1.0894775390625e-02]])
        >>> band_biquad_op = audio.BandBiquad(44100, 200.0)
        >>> waveform_filtered = band_biquad_op(waveform)
    """
    @check_band_biquad
    def __init__(self, sample_rate, central_freq, Q=0.707, noise=False):
        self.sample_rate = sample_rate
        self.central_freq = central_freq
        self.Q = Q
        self.noise = noise

    def parse(self):
        return cde.BandBiquadOperation(self.sample_rate, self.central_freq, self.Q, self.noise)


class BandpassBiquad(TensorOperation):
    """
    Design two-pole band-pass filter.  Similar to SoX implementation.

    Args:
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        central_freq (float): central frequency (in Hz)
        Q (float, optional): https://en.wikipedia.org/wiki/Q_factor Range: (0,1] (Default=0.707).
        const_skirt_gain (bool, optional) : If ``True``, uses a constant skirt gain (peak gain = Q).
            If ``False``, uses a constant 0dB peak gain. (Default: ``False``)

    Examples:
        >>> import mindspore.dataset.audio.transforms as audio
        >>> import numpy as np

        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03],[9.246826171875e-03, 1.0894775390625e-02]])
        >>> bandpass_biquad_op = audio.BandpassBiquad(44100, 200.0)
        >>> waveform_filtered = bandpass_biquad_op(waveform)
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
    Design two-pole band filter for audio waveform of dimension of `(..., time)`

    Args:
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz),
            the value must be greater than 0 .
        central_freq (float): central frequency (in Hz),
            the value must be greater than 0 .
        Q(float, optional): Quality factor,https://en.wikipedia.org/wiki/Q_factor,
            Range: (0, 1] (Default=0.707).

    Examples:
        >>> import mindspore.dataset.audio.transforms as audio
        >>> import numpy as np

        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03],[9.246826171875e-03, 1.0894775390625e-02]])
        >>> band_biquad_op = audio.BandBiquad(44100, 200.0)
        >>> waveform_filtered = band_biquad_op(waveform)
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
    Design a bass tone-control effect for audio waveform of dimension of `(..., time)`

    Args:
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        gain (float): desired gain at the boost (or attenuation) in dB.
        central_freq (float): central frequency (in Hz)(Default=100.0).
        Q(float, optional): Quality factor, https://en.wikipedia.org/wiki/Q_factor, Range: (0, 1] (Default=0.707).

    Examples:
        >>> import mindspore.dataset.audio.transforms as audio
        >>> import numpy as np

        >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03],[9.246826171875e-03, 1.0894775390625e-02]])
        >>> bass_biquad_op = audio.BassBiquad(44100, 100.0)
        >>> waveform_filtered = bass_biquad_op(waveform)
    """
    @check_bass_biquad
    def __init__(self, sample_rate, gain, central_freq=100.0, Q=0.707):
        self.sample_rate = sample_rate
        self.gain = gain
        self.central_freq = central_freq
        self.Q = Q

    def parse(self):
        return cde.BassBiquadOperation(self.sample_rate, self.gain, self.central_freq, self.Q)


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
