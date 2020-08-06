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
#ifndef MINDSPORE_LITE_SRC_BACKEND_ARM_OPCLIB_STRIDED_SLICE_H_
#define MINDSPORE_LITE_SRC_BACKEND_ARM_OPCLIB_STRIDED_SLICE_H_

#include "src/runtime/kernel/arm/opclib/op_base.h"

struct StridedSliceParameter {
  OpParameter op_parameter_;
  int begins_[8] = {0};
  int ends_[8] = {0};
  int strides_[8] = {1};
  int isScale;
  int num_axes_;
  int in_shape_[8];
  LiteDataType data_type;
};

int DoStridedSlice(const void *inputs, void *output, StridedSliceParameter *param);
#endif  // MINDSPORE_LITE_SRC_BACKEND_ARM_OPCLIB_STRIDED_SLICE_H_
