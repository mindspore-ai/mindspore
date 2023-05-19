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
#ifndef NNACL_RNADOM_PARAMETER_H_
#define NNACL_RNADOM_PARAMETER_H_

#include "nnacl/op_base.h"

typedef struct RandomParam {
  OpParameter op_parameter_;
  int seed_;
  int seed2_;
} RandomParam;

typedef struct RandomNormalParam {
  OpParameter op_parameter_;
  float seed_;
  float mean_;
  float scale_;
} RandomNormalParam;

#endif  // NNACL_RNADOM_STANDARD_NORMAL_PARAMETER_H_
