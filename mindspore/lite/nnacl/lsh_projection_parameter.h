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

#ifndef MINDSPORE_LITE_NNACL_LSH_PROJECTION_PARAMETER_H_
#define MINDSPORE_LITE_NNACL_LSH_PROJECTION_PARAMETER_H_

#include "nnacl/op_base.h"

typedef struct LshProjectionParameter {
  OpParameter op_parameter_;
  int lsh_type_;
  int hash_shape_[2];
  int in_item_num_;
  size_t in_item_size_;
  size_t seed_size_;
  size_t key_size_;
  int64_t real_dst_count;
  int task_id_;
  int64_t count_unit_;
} LshProjectionParameter;

#endif  // MINDSPORE_LITE_NNACL_LSH_PROJECTION_PARAMETER_H_
