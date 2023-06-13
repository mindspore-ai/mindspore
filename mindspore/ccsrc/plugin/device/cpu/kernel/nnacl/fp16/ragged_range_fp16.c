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
#include "nnacl/fp16/ragged_range_fp16.h"

void RaggedRangeFp16(const float16_t *starts, const float16_t *limits, const float16_t *deltas, int *splits,
                     float16_t *value, const RaggedRangeStruct *param) {
  splits[0] = 0;
  for (int i = 0; i < param->rows_; i++) {
    float16_t start = param->starts_is_scalar_ ? starts[0] : starts[i];
    float16_t limit = param->limits_is_scalar_ ? limits[0] : limits[i];
    float16_t delta = param->deltas_is_scalar_ ? deltas[0] : deltas[i];
    int len = NNACL_MAX((int)ceil((float16_t)(limit - start) / delta), 0);
    splits[i + 1] = splits[i] + len;
    for (int j = 0; j < len; j++) {
      *value++ = start;
      start += delta;
    }
  }
}
