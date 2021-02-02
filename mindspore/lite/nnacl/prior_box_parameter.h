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
#ifndef MINDSPORE_LITE_NNACL_PRIOR_BOX_PARAMETER_H_
#define MINDSPORE_LITE_NNACL_PRIOR_BOX_PARAMETER_H_

#include "nnacl/op_base.h"

typedef struct PriorBoxParameter {
  // Primitive parameter
  OpParameter op_parameter_;
  int32_t min_sizes_size;
  int32_t min_sizes[MAX_SHAPE_SIZE];
  int32_t max_sizes_size;
  int32_t max_sizes[MAX_SHAPE_SIZE];
  int32_t aspect_ratios_size;
  float aspect_ratios[MAX_SHAPE_SIZE];
  float variances[COMM_SHAPE_SIZE];
  int32_t image_size_w;
  int32_t image_size_h;
  float step_w;
  float step_h;
  bool clip;
  bool flip;
  float offset;
} PriorBoxParameter;

#endif  // MINDSPORE_LITE_NNACL_PRIOR_BOX_PARAMETER_H_
