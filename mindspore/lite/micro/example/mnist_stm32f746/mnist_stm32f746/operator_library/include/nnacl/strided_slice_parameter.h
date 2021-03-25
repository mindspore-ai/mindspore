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
#ifndef MINDSPORE_LITE_NNACL_STRIDED_SLICE_PARAMETER_H_
#define MINDSPORE_LITE_NNACL_STRIDED_SLICE_PARAMETER_H_

#include "nnacl/op_base.h"

typedef struct StridedSliceParameter {
  // primitive parameter
  OpParameter op_parameter_;
  int begins_[MAX_SHAPE_SIZE];
  int ends_[MAX_SHAPE_SIZE];
  int strides_[MAX_SHAPE_SIZE];
  int isScale;

  // shape correlative
  int in_shape_length_;
  int in_shape_[MAX_SHAPE_SIZE];

  // other parameter
  int num_axes_;
  LiteDataType data_type;
  int begins_mask_;
  int ends_mask_;
  int ellipsisMask_;
  int newAxisMask_;
  int shrinkAxisMask_;
} StridedSliceParameter;

#endif  // MINDSPORE_LITE_NNACL_STRIDED_SLICE_PARAMETER_H_
