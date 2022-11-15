/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/online_fusion/split_reduce_concat_fp32.h"
#include <float.h>
#include "nnacl/reduce_fp32_simd.h"
#include "nnacl/errorcode.h"

int64_t Fp32SplitReduceSumConcatFusion(const float *src, float *dst, int64_t inner_size, int64_t mid_size,
                                       int *mid_split, int64_t mid_len, int64_t out_size) {
  const float *cur_src = src;
  float *cur_dst = dst;
  for (int64_t i = 0; i < out_size; i++) {
    for (int64_t j = 0; j < mid_len; j++) {
      int k = 0;
      SIMD_RUN_NO_SCALAR(ReduceSum, k, cur_src, cur_dst, inner_size, mid_split[j]);
      for (; k < inner_size; k++) {
        float result = cur_src[k];
        for (int64_t l = 1; l < mid_split[j]; l++) {
          result += cur_src[inner_size * l + k];
        }
        cur_dst[k] = result;
      }
      cur_src += (inner_size * mid_split[j]);
      cur_dst += inner_size;
    }
  }
  return NNACL_OK;
}
