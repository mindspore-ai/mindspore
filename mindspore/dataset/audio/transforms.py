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
from .validators import check_allpass_biquad, check_band_biquad, check_bandpass_biquad, check_bandreject_biquad, \
    check_bass_biquad


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
