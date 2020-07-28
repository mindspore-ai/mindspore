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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_POWER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_POWER_H_
#include <math.h>
#include "src/runtime/kernel/arm/opclib/op_base.h"

struct PowerParameter {
  OpParameter op_parameter_;
  float power_;
  float scale_;
  float shift_;
};

inline void Power(const float *input_data, float *output_data, int len, float power, float scale, float shift) {
  for (int i = 0; i < len; ++i) {
    output_data[i] = pow((scale * input_data[i] + shift), power);
  }
}

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_POWER_H_

