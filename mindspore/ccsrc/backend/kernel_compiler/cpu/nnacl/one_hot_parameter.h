/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_ONE_HOT_PARAMETER_H_
#define MINDSPORE_NNACL_ONE_HOT_PARAMETER_H_
#include "nnacl/op_base.h"

typedef struct OneHotParameter {
  // Primitive parameter
  OpParameter op_parameter_;
  int axis_;
  // other parameter
  int depth_;
  int outer_size_;
  int inner_size_;
  bool support_neg_index_;  // if true, support neg index in indices tensor; if false, set off_value on neg index.
} OneHotParameter;

#endif  // MINDSPORE_NNACL_ONE_HOT_PARAMETER_H_
