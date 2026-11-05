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


DE_C_SCALE_TYPE = {ScaleType.POWER: cde.ScaleType.DE_SCALE_TYPE_POWER,
                   ScaleType.MAGNITUDE: cde.ScaleType.DE_SCALE_TYPE_MAGNITUDE}


DE_C_BORDER_TYPE = {
    BorderType.CONSTANT: cde.BorderType.DE_BORDER_CONSTANT,
    BorderType.EDGE: cde.BorderType.DE_BORDER_EDGE,
    BorderType.REFLECT: cde.BorderType.DE_BORDER_REFLECT,
    BorderType.SYMMETRIC: cde.BorderType.DE_BORDER_SYMMETRIC,
}


DE_C_DENSITY_FUNCTION = {DensityFunction.TPDF: cde.DensityFunction.DE_DENSITY_FUNCTION_TPDF,
                         DensityFunction.RPDF: cde.DensityFunction.DE_DENSITY_FUNCTION_RPDF,
                         DensityFunction.GPDF: cde.DensityFunction.DE_DENSITY_FUNCTION_GPDF}


DE_C_FADE_SHAPE = {FadeShape.QUARTER_SINE: cde.FadeShape.DE_FADE_SHAPE_QUARTER_SINE,
                   FadeShape.HALF_SINE: cde.FadeShape.DE_FADE_SHAPE_HALF_SINE,
                   FadeShape.LINEAR: cde.FadeShape.DE_FADE_SHAPE_LINEAR,
                   FadeShape.LOGARITHMIC: cde.FadeShape.DE_FADE_SHAPE_LOGARITHMIC,
                   FadeShape.EXPONENTIAL: cde.FadeShape.DE_FADE_SHAPE_EXPONENTIAL}


DE_C_MODULATION = {Modulation.SINUSOIDAL: cde.Modulation.DE_MODULATION_SINUSOIDAL,
                   Modulation.TRIANGULAR: cde.Modulation.DE_MODULATION_TRIANGULAR}

DE_C_INTERPOLATION = {Interpolation.LINEAR: cde.Interpolation.DE_INTERPOLATION_LINEAR,
                      Interpolation.QUADRATIC: cde.Interpolation.DE_INTERPOLATION_QUADRATIC}


DE_C_NORM_MODE = {NormMode.ORTHO: cde.NormMode.DE_NORM_MODE_ORTHO,
                  NormMode.NONE: cde.NormMode.DE_NORM_MODE_NONE}


DE_C_MEL_TYPE = {MelType.SLANEY: cde.MelType.DE_MEL_TYPE_SLANEY,
                 MelType.HTK: cde.MelType.DE_MEL_TYPE_HTK}

DE_C_NORM_TYPE = {NormType.NONE: cde.NormType.DE_NORM_TYPE_NONE,
                  NormType.SLANEY: cde.NormType.DE_NORM_TYPE_SLANEY}


DE_C_RESAMPLE_METHOD = {ResampleMethod.SINC_INTERPOLATION: cde.ResampleMethod.DE_RESAMPLE_SINC_INTERPOLATION,
                        ResampleMethod.KAISER_WINDOW: cde.ResampleMethod.DE_RESAMPLE_KAISER_WINDOW}


DE_C_WINDOW_TYPE = {WindowType.BARTLETT: cde.WindowType.DE_WINDOW_TYPE_BARTLETT,
                    WindowType.BLACKMAN: cde.WindowType.DE_WINDOW_TYPE_BLACKMAN,
                    WindowType.HAMMING: cde.WindowType.DE_WINDOW_TYPE_HAMMING,
                    WindowType.HANN: cde.WindowType.DE_WINDOW_TYPE_HANN,
                    WindowType.KAISER: cde.WindowType.DE_WINDOW_TYPE_KAISER}


DE_C_GAIN_TYPE = {GainType.AMPLITUDE: cde.GainType.DE_GAIN_TYPE_AMPLITUDE,
                  GainType.POWER: cde.GainType.DE_GAIN_TYPE_POWER,
                  GainType.DB: cde.GainType.DE_GAIN_TYPE_DB}
