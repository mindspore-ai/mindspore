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

#ifndef MINDSPORE_NNACL_LSH_PROJECTION_PARAMETER_H_
#define MINDSPORE_NNACL_LSH_PROJECTION_PARAMETER_H_

#include "nnacl/op_base.h"

typedef struct LshProjectionParameter {
  // Primitive parameter
  OpParameter op_parameter_;
  // shape correlative
  int hash_shape_[2];
  // other parameter
  int lsh_type_;
  int feature_num_;
  char **hash_buffs_;
  size_t hash_buff_size_;
  int64_t thread_stride_;
} LshProjectionParameter;

#endif  // MINDSPORE_NNACL_LSH_PROJECTION_PARAMETER_H_
