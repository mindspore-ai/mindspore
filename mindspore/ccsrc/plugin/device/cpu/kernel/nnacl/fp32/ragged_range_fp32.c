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

#include "nnacl/fp32/ragged_range_fp32.h"
#include <math.h>
#include "nnacl/op_base.h"

void RaggedRangeFp32(const float *starts, const float *limits, const float *deltas, int32_t *splits, float *value,
                     RaggedRangeStruct *ragged_range) {
  splits[0] = 0;
  for (int i = 0; i < ragged_range->rows_; i++) {
    float start = ragged_range->starts_is_scalar_ ? starts[0] : starts[i];
    float limit = ragged_range->limits_is_scalar_ ? limits[0] : limits[i];
    float delta = ragged_range->deltas_is_scalar_ ? deltas[0] : deltas[i];
    int len = NNACL_MAX((int)ceil((float)(limit - start) / delta), 0);
    splits[i + 1] = splits[i] + len;
    for (int j = 0; j < len; j++) {
      *value++ = start;
      start += delta;
    }
  }
}

void RaggedRangeInt(const int32_t *starts, const int32_t *limits, const int32_t *deltas, int32_t *splits,
                    int32_t *value, RaggedRangeStruct *ragged_range) {
  splits[0] = 0;
  for (int i = 0; i < ragged_range->rows_; i++) {
    int start = ragged_range->starts_is_scalar_ ? starts[0] : starts[i];
    int limit = ragged_range->limits_is_scalar_ ? limits[0] : limits[i];
    int delta = ragged_range->deltas_is_scalar_ ? deltas[0] : deltas[i];
    NNACL_CHECK_ZERO_RETURN(delta);
    int len = NNACL_MAX((int)ceil((float)(limit - start) / delta), 0);
    splits[i + 1] = splits[i] + len;
    for (int j = 0; j < len; j++) {
      *value++ = start;
      start += delta;
    }
  }
}
