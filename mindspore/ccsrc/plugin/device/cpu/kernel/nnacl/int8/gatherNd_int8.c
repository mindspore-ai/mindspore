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

#include "nnacl/int8/gatherNd_int8.h"
#include <string.h>
#include "nnacl/errorcode.h"

int GatherNdInt8(int8_t *input, int8_t *output, const int *in_offset, int area, int count, GatherQuantArg param) {
  double alpha = param.alpha_;
  int z1 = param.zp_in_;
  int z2 = param.zp_out_;
  for (int i = 0; i < count; ++i) {
    for (int j = 0; j < area; ++j) {
      int32_t tmp = round(alpha * (input[in_offset[i] + j] - z1)) + z2;
      tmp = tmp > 127 ? 127 : tmp;
      tmp = tmp < -128 ? -128 : tmp;
      output[area * i + j] = (int8_t)tmp;
    }
  }
  return NNACL_OK;
}
