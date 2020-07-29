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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_SCALE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_SCALE_H_

#include "src/runtime/kernel/arm/opclib/op_base.h"

struct ScaleParameter {
  OpParameter op_parameter_;
  int out_count_;
  int channel_;
  int in_stride_;
  int axis_;
  int num_axis_;
};

int DoScale(float *in_data, float *out_data, float *scale, float *offset, int units_offset, int num_unit,
            ScaleParameter *scale_param);
int DoScale(float *in_data, float *out_data, float *scale, int units_offset, int num_unit, ScaleParameter *scale_param);
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_SCALE_H_

