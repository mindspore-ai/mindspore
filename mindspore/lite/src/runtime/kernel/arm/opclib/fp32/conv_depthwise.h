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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_P32_CONV_DEPTHWISE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_P32_CONV_DEPTHWISE_H_

#include "src/runtime/kernel/arm/opclib/conv_parameter.h"

struct SlidingWindowParam {
  int left_;
  int right_;
  int top_;
  int bottom_;
  int c_block_;
  int block_channel_;
  int out_step_;
  int out_h_step_;
  int in_step_;
  int in_h_step_;
  int in_sh_step_;  // stride H
  int in_sw_step_;  // stride W
  int in_kh_step_;  // kernel H
  int in_kw_step_;  // kernel W
  int kernel_step_;
};

void InitSlidingParam(SlidingWindowParam *sliding, const ConvParameter *conv_param, int block);

void ConvDwC4Fp32(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
                  const ConvParameter *conv_param, const SlidingWindowParam *sliding, int task_id);

void DeconvDwC4Fp32(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
                    const ConvParameter *conv_param, const SlidingWindowParam *sliding, int task_id);

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_P32_CONV_DEPTHWISE_H_

