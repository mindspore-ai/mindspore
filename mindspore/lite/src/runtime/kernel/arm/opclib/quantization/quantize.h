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
#include <limits>

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
  int32_t left_shift;
  int32_t right_shift;
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

struct CropQuantArg {
  QuantArg in_args_;
  QuantArg out_args_;
  int output_activation_min_;
  int output_activation_max_;
};

struct ArithSelfQuantArg {
  QuantArg in_args_;
  QuantArg out_args_;
  int output_activation_min_;
  int output_activation_max_;
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

inline void CalculateActivationRangeQuantized(bool is_relu, bool is_relu6, int32_t zp, int32_t scale, int *mini,
                                              int *maxi) {
  int32_t min = std::numeric_limits<int8_t>::min();
  int32_t max = std::numeric_limits<int8_t>::max();
  int32_t quantized_zero = QuantizeToInt8(0, scale, zp);
  int32_t quantized_six = QuantizeToInt8(6, scale, zp);
  if (is_relu) {
    min = min > quantized_zero ? min : quantized_zero;
  } else if (is_relu6) {
    min = min > quantized_zero ? min : quantized_zero;
    max = max < quantized_six ? max : quantized_six;
  } else {
    // do nothing
  }
  *mini = min;
  *maxi = max;
}
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_QUANTIZATION_QUANTIZE_H_
