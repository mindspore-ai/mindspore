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
from .validators import check_band_biquad


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
