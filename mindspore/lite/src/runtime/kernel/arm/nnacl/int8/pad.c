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

#include "nnacl/int8/pad.h"
#include "nnacl/common_func.h"
#include "nnacl/errorcode.h"

int PadConstant4D(const int8_t *in_data, int8_t *out_data, const int32_t *in_dims, const int32_t *out_dims,
                   const int32_t *paddings, const int tid, const int thread_num) {
  int32_t copy_size = in_dims[3];
  for (int n = 0; n < in_dims[0]; n++) {
    for (int h = tid; h < in_dims[1]; h += thread_num) {
      for (int w = 0; w < in_dims[2]; w++) {
        const int8_t *in = in_data + offset(in_dims, n, h, w, 0);
        int8_t *out = out_data + offset(out_dims, n + paddings[0], h + paddings[2], w + paddings[4], paddings[6]);
        memcpy(out, in, copy_size * sizeof(int8_t));
      }
    }
  }
  return NNACL_OK;
}
