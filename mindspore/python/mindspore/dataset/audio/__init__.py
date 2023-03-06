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
This module is to support audio augmentations.
It includes two parts: audio transforms and utils.
audio transforms is a high performance processing module with common audio operations.
utils provides some general methods for audio processing.

Common imported modules in corresponding API examples are as follows:

.. code-block::

    import mindspore.dataset as ds
    import mindspore.dataset.audio as audio
    from mindspore.dataset.audio import utils

Alternative and equivalent imported audio module is as follows:

.. code-block::

    import mindspore.dataset.audio.transforms as audio

Descriptions of common data processing terms are as follows:

- TensorOperation, the base class of all data processing operations implemented in C++.
- AudioTensorOperation, the base class of all audio processing operations. It is a derived class of TensorOperation.

The data transform operation can be executed in the data processing pipeline or in the eager mode:

- Pipeline mode is generally used to process datasets. For examples, please refer to
  `introduction to data processing pipeline <https://www.mindspore.cn/docs/en/master/api_python/
  mindspore.dataset.html#introduction-to-data-processing-pipeline>`_ .
- Eager mode is generally used for scattered samples. Examples of audio preprocessing are as follows:

  .. code-block::

      import numpy as np
      import mindspore.dataset.audio as audio
      from mindspore.dataset.audio import ResampleMethod

      # audio sample
      waveform = np.random.random([1, 30])

      # transform
      resample_op = audio.Resample(orig_freq=48000, new_freq=16000,
                                   resample_method=ResampleMethod.SINC_INTERPOLATION,
                                   lowpass_filter_width=6, rolloff=0.99, beta=None)
      waveform_resampled = resample_op(waveform)
      print("waveform reampled: {}".format(waveform_resampled), flush=True)
"""
from __future__ import absolute_import

from mindspore.dataset.audio import transforms
from mindspore.dataset.audio import utils
from mindspore.dataset.audio.transforms import AllpassBiquad, AmplitudeToDB, Angle, BandBiquad, \
    BandpassBiquad, BandrejectBiquad, BassBiquad, Biquad, ComplexNorm, ComputeDeltas, Contrast, DBToAmplitude, \
    DCShift, DeemphBiquad, DetectPitchFrequency, Dither, EqualizerBiquad, Fade, Filtfilt, Flanger, FrequencyMasking, \
    Gain, GriffinLim, HighpassBiquad, InverseMelScale, InverseSpectrogram, LFCC, LFilter, LowpassBiquad, Magphase, \
    MaskAlongAxis, MaskAlongAxisIID, MelScale, MelSpectrogram, MFCC, MuLawDecoding, MuLawEncoding, Overdrive, \
    Phaser, PhaseVocoder, PitchShift, Resample, RiaaBiquad, SlidingWindowCmn, SpectralCentroid, Spectrogram, \
    TimeMasking, TimeStretch, TrebleBiquad, Vad, Vol
from mindspore.dataset.audio.utils import BorderType, DensityFunction, FadeShape, GainType, Interpolation, \
    MelType, Modulation, NormMode, NormType, ResampleMethod, ScaleType, WindowType, create_dct, linear_fbanks, \
    melscale_fbanks
