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
#ifndef NNACL_FP16_RAGGED_RANGE_FP16_H_
#define NNACL_FP16_RAGGED_RANGE_FP16_H_

#include <math.h>
#include "nnacl/op_base.h"
#include "nnacl/kernel/ragged_range.h"

void RaggedRangeFp16(const float16_t *starts, const float16_t *limits, const float16_t *deltas, int *splits,
                     float16_t *value, const RaggedRangeStruct *param);

#endif  //   NNACL_FP16_RAGGED_RANGE_FP16_H_
