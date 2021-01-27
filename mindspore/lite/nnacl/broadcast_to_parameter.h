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
#ifndef MINDSPORE_LITE_NNACL_FP32_BROADCAST_TO_PARAMETER_H_
#define MINDSPORE_LITE_NNACL_FP32_BROADCAST_TO_PARAMETER_H_

#include "nnacl/op_base.h"

typedef struct BroadcastToParameter {
  OpParameter op_parameter_;
  int shape_[COMM_SHAPE_SIZE];
  size_t shape_size_;
} BroadcastToParameter;

typedef struct BroadcastShapeInfo {
  int input_shape_[COMM_SHAPE_SIZE];
  int input_shape_size_;
  int output_shape_[COMM_SHAPE_SIZE];
  int output_shape_size_;
} BroadcastShapeInfo;

#endif  // MINDSPORE_LITE_NNACL_FP32_BROADCAST_TO_PARAMETER_H_
