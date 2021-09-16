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

void RaggedRangeFp32(const float *starts, const float *limits, const float *deltas, int *splits, float *value,
                     const RaggedRangeParameter *param) {
  splits[0] = 0;
  for (int i = 0; i < param->rows; i++) {
    float start = param->starts_is_scalar ? starts[0] : starts[i];
    float limit = param->limits_is_scalar ? limits[0] : limits[i];
    float delta = param->deltas_is_scalar ? deltas[0] : deltas[i];
    int len = MSMAX((int)ceil((float)(limit - start) / delta), 0);
    splits[i + 1] = splits[i] + len;
    for (int j = 0; j < len; j++) {
      *value++ = start;
      start += delta;
    }
  }
}

void RaggedRangeInt(const int *starts, const int *limits, const int *deltas, int *splits, int *value,
                    const RaggedRangeParameter *param) {
  splits[0] = 0;
  for (int i = 0; i < param->rows; i++) {
    int start = param->starts_is_scalar ? starts[0] : starts[i];
    int limit = param->limits_is_scalar ? limits[0] : limits[i];
    int delta = param->deltas_is_scalar ? deltas[0] : deltas[i];
    NNACL_CHECK_ZERO_RETURN(delta);
    int len = MSMAX((int)ceil((float)(limit - start) / delta), 0);
    splits[i + 1] = splits[i] + len;
    for (int j = 0; j < len; j++) {
      *value++ = start;
      start += delta;
    }
  }
}
