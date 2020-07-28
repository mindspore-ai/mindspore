/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_QUANTIZATION_QUANTIZE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_QUANTIZATION_QUANTIZE_H_

#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>

struct QuantArg {
  double scale_;
  int32_t zp_;
};

struct ConvQuantArg {
  QuantArg **quant_args_;
  double *real_multiplier_;
  int32_t *left_shift_;
  int32_t *right_shift_;
  int32_t *quant_multiplier_;
  int32_t *out_act_min_;
  int32_t *out_act_max_;
};

struct ConcatQuantArg {
  int *input_sizes_;
  int output_size_;
  int **input_shapes_;
  int *output_shape_;
  size_t input_num_;
  size_t output_dim_;
  QuantArg *in_quant_args_;
  QuantArg out_quant_args_;
};

struct FcQuantArg {
  QuantArg input;
  QuantArg weight;
  QuantArg output;
  int32_t out_act_min;
  int32_t out_act_max;
  int32_t output_shift;
  int32_t quant_multiplier;
};

struct PadQuantArg {
  QuantArg *in_quant_args_ = nullptr;
  QuantArg *out_quanr_args_ = nullptr;
  int8_t *constant_value_ = nullptr;
};

struct MulQuantArg {
  QuantArg in_quant_args_[2];
  QuantArg out_quant_arg_;
  int output_multiplier_;
  int output_activation_min_;
  int output_activation_max_;
  int shift_left_;
  int shift_right_;
};

void QuantizeMultiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift);

inline void QuantizeMultiplierSmallerThanOne(double double_multiplier, int32_t *quantized_multiplier,
                                             int *right_shift) {
  if (quantized_multiplier == nullptr || right_shift == nullptr) {
    return;
  }
  int shift;
  QuantizeMultiplier(double_multiplier, quantized_multiplier, &shift);
  *right_shift = -shift;
}

inline void QuantizeRoundParameter(double double_multiplier, int32_t *quantized_multiplier, int *left_shift,
                                   int *right_shift) {
  int shift;
  QuantizeMultiplierSmallerThanOne(double_multiplier, quantized_multiplier, &shift);
  shift = -shift;
  if (shift < 0) {
    *left_shift = 0;
    *right_shift = shift;
  } else {
    *left_shift = shift;
    *right_shift = 0;
  }
}

inline uint8_t QuantizeToUint8(float real_value, float scale, int32_t zp) { return round(real_value / scale + zp); }

inline int32_t QuantizeToInt8(float real_value, float scale, int32_t zp) { return round(real_value / scale + zp); }

inline void CalculateActivationRangeQuantized(float fmax, float fmin, float scale, int zero_point, int *imax,
                                              int *imin) {
  int8_t qmin = (int8_t)CHAR_MIN;
  int8_t qmax = (int8_t)CHAR_MAX;
  int8_t qfmin = QuantizeToInt8(fmin, scale, zero_point);
  int8_t qfmax = QuantizeToInt8(fmax, scale, zero_point);
  *imin = qmin < qfmin ? qmin : qfmin;
  *imax = qmax > qfmax ? qmax : qfmax;
}
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_QUANTIZATION_QUANTIZE_H_
