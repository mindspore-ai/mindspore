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
#ifndef LITE_SRC_BACKEND_ARM_NNACL_SPACE_TO_DEPTH_PARAMETER_H_
#define LITE_SRC_BACKEND_ARM_NNACL_SPACE_TO_DEPTH_PARAMETER_H_
#include "nnacl/op_base.h"

typedef struct SpaceToDepthParameter {
  // primitive parameter
  OpParameter op_parameter_;
  int32_t block_size_;
  int32_t date_type_len;
} SpaceToDepthParameter;

#endif  // LITE_SRC_BACKEND_ARM_NNACL_SPACE_TO_DEPTH_PARAMETER_H_
