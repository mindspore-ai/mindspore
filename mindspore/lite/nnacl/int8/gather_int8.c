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
 *
 */
#include "nnacl/int8/gather_int8.h"
#include "nnacl/op_base.h"
#include "nnacl/int8/quantize.h"
#include "nnacl/errorcode.h"

int GatherInt8(int8_t *in_data, int8_t *out_data, int outer_size, int inner_size, int limit, const int *indices,
               int indices_element_size, GatherQuantArg para) {
  double alpha = para.alpha_;
  int z1 = para.zp_in_;
  int z2 = para.zp_out_;
  int i, m, j;
  for (m = 0; m < outer_size; ++m) {
    const int8_t *inputm = in_data + inner_size * m * limit;
    int8_t *outputm = out_data + inner_size * m * indices_element_size;
    for (i = 0; i < indices_element_size; ++i) {
      if (indices[i] < 0 || indices[i] > limit) {
        return NNACL_ERR;
      }
      for (j = 0; j < inner_size; ++j) {
        int32_t tmp = round(alpha * (inputm[indices[i] * inner_size + j] - z1)) + z2;
        tmp = tmp > 127 ? 127 : tmp;
        tmp = tmp < -128 ? -128 : tmp;
        outputm[i * inner_size + j] = (int8_t)tmp;
      }
    }
  }
  return NNACL_OK;
}
