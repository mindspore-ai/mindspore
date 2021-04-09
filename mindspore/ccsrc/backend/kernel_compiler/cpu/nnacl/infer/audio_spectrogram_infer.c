/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/infer/audio_spectrogram_infer.h"
#include "nnacl/infer/infer_register.h"

int Log2Ceil(uint32_t length) {
  if (length == 0) {
    return -1;
  }
  int floor = 0;
  for (int i = 4; i >= 0; --i) {
    const int shift = (1 << i);
    uint32_t tmp = length >> shift;
    if (tmp != 0) {
      length = tmp;
      floor += shift;
    }
  }
  return length == (length & ~(length - 1)) ? floor : floor + 1;
}

uint32_t GetFftLength(uint32_t length) {
  int shift = Log2Ceil(length);
  return 1 << shift;
}

int AudioSpectrogramInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                               OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  if (input->shape_size_ != 2) {
    return NNACL_ERR;
  }
  AudioSpectrogramParameter *param = (AudioSpectrogramParameter *)parameter;
  if (param->window_size_ < 2) {
    return NNACL_ERR;
  }
  if (param->stride_ < 1) {
    return NNACL_ERR;
  }
  int output_shape[3];
  output_shape[0] = input->shape_[1];
  int sample_sub_window = input->shape_[0] - param->window_size_;
  output_shape[1] = sample_sub_window < 0 ? 0 : 1 + sample_sub_window / param->stride_;
  // compute fft length
  int fft_length = GetFftLength(param->window_size_);
  output_shape[2] = fft_length / 2 + 1;
  SetShapeArray(output, output_shape, 3);
  return NNACL_OK;
}

REG_INFER(AudioSpectrogram, PrimType_AudioSpectrogram, AudioSpectrogramInferShape)
