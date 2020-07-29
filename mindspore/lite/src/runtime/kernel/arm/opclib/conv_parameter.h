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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_CONV_PARAMETER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_CONV_PARAMETER_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "src/runtime/kernel/arm/opclib/op_base.h"
#include "src/runtime/kernel/arm/opclib/quantization/quantize.h"

struct ConvParameter {
  OpParameter op_parameter_;
  ConvQuantArg conv_quant_arg_;
  int kernel_h_;
  int kernel_w_;
  int stride_h_;
  int stride_w_;
  int dilation_h_;
  int dilation_w_;
  int pad_h_;
  int pad_w_;
  int pad_u_;
  int pad_d_;
  int pad_l_;
  int pad_r_;
  int group_;
  int tile_num_;
  int input_batch_;
  int input_h_;
  int input_w_;
  int input_channel_;
  int output_batch_;
  int output_h_;
  int output_w_;
  int output_channel_;
  int thread_num_;
  int input_unit_;
  int output_unit_;
  bool is_relu_;
  bool is_relu6_;
};

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_CONV_PARAMETER_H_

