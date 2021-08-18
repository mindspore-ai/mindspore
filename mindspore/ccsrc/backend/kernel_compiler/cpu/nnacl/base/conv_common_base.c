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
#include "nnacl/base/conv_common_base.h"
#include "nnacl/errorcode.h"

#define MIN_UNIT 2
#define MAX_UNIT 8

#if defined(ENABLE_ARM) || (defined(ENABLE_SSE) && !defined(ENABLE_AVX))
bool CheckConvDw1DWinograd(const ConvParameter *conv_param, int thread_num) {
  return conv_param->kernel_h_ == 3 && conv_param->kernel_w_ == 3 && conv_param->stride_w_ == 1 &&
         conv_param->stride_h_ == 1 && conv_param->dilation_h_ == 1 && conv_param->dilation_w_ == 1 &&
         conv_param->pad_u_ == 1 && conv_param->pad_d_ == 1 && conv_param->pad_l_ == 1 && conv_param->pad_r_ == 1 &&
         conv_param->input_channel_ == conv_param->output_channel_ && conv_param->output_w_ >= 4 &&
         conv_param->output_h_ >= thread_num * 4;  // better had more than 4 rows for each thread
}
#endif

bool CheckWinogradInputOutputUnit(int input_unit, int output_unit) {
  if (input_unit != 4 && input_unit != 6 && input_unit != 8) {
    return false;
  }
  if ((output_unit >= input_unit) || (output_unit < 2)) {
    return false;
  }
  return true;
}

// Reference to the paper "Fast Algorithms for Convolutional Neural Networks"
// Utilize cost model to compute performance gain.
// If the gain is greater than got from Im2col, winograd algorithm will be chosen.
int SelectOutputUnit(const ConvParameter *conv_param) {
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int in_c = conv_param->input_channel_;
  int out_w = conv_param->output_w_;
  int out_h = conv_param->output_h_;
  int out_c = conv_param->output_channel_;
  if (conv_param->op_parameter_.thread_num_ == 0) {
    return NNACL_PARAM_INVALID;
  }
  int unit2 = UP_DIV(out_w * out_h, C12NUM * conv_param->op_parameter_.thread_num_);
  int max_out_unit = (int)(sqrtf((float)unit2));
  max_out_unit = max_out_unit < MAX_UNIT ? max_out_unit : MAX_UNIT;
  max_out_unit = max_out_unit > MIN_UNIT ? max_out_unit : MIN_UNIT;

  int unit = 0;
  float max_rate = 0.0f;
  float common_cost = (float)out_h * out_w * in_c * out_c * kernel_h * kernel_w;

  for (int i = MIN_UNIT; i <= max_out_unit; ++i) {
    int input_unit = i + kernel_w - 1;
    if (!CheckWinogradInputOutputUnit(input_unit, i)) {
      continue;
    }
    float penalty = ((float)input_unit * input_unit) / ((float)kernel_h * kernel_w) * 0.12f;
    float wino_cost = ((2 + out_c) * (float)input_unit * input_unit * in_c + ((float)input_unit + i) * i * out_c) *
                      UP_DIV(out_w, i) * UP_DIV(out_h, i);
    float reduce_rate = common_cost / wino_cost - penalty;
    if (reduce_rate > max_rate) {
      max_rate = reduce_rate;
      unit = i;
    }
  }
  if (max_rate < 1.0f) {
    return 1;
  }
  // If output_unit is 1, then it is conventional convolution
  return unit;
}

bool CheckIfUseWinograd(int *output_unit, const ConvParameter *conv_param) {
  if (conv_param->kernel_w_ == conv_param->kernel_h_ && conv_param->dilation_h_ == 1 && conv_param->dilation_w_ == 1 &&
      conv_param->stride_h_ == 1 && conv_param->stride_w_ == 1 && conv_param->input_channel_ != 1) {
    *output_unit = SelectOutputUnit(conv_param);
    if (*output_unit > 1) {
      return true;
    }
  }
  return false;
}
