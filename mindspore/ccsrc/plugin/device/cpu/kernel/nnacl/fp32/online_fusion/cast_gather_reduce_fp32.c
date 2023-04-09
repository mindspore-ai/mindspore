/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/online_fusion/cast_gather_reduce_fp32.h"
#include "nnacl/errorcode.h"
#include "nnacl/cast_gather_reduce_fp32_simd.h"

int64_t Fp32CastGatherReduceInt64Fusion(float *output_data, const int64_t *input_indices, const float *input_data,
                                        int32_t inner_size, int32_t input_data_inner_size, int32_t outer_start,
                                        int32_t outer_end) {
  int index = 0;
  SIMD_RUN_NO_SCALAR(Fp32CastGatherReduceInt64Fusion, index, output_data, input_indices, input_data, inner_size,
                     input_data_inner_size, outer_start, outer_end);

  if (index < input_data_inner_size) {
    for (int i = outer_start; i < outer_end; i++) {
      float *result = output_data + i * input_data_inner_size + index;
      int64_t indice0 = input_indices[i * inner_size];
      for (int k = index; k < input_data_inner_size; k++) {
        result[k] = input_data[indice0 * input_data_inner_size + k];
      }
      for (int j = 1; j < inner_size; j++) {
        int64_t indice = input_indices[i * inner_size + j];
        for (int k = index; k < input_data_inner_size; k++) {
          result[k] += input_data[indice * input_data_inner_size + k];
        }
      }
    }
  }
  return NNACL_OK;
}

int64_t Fp32CastGatherReduceInt32Fusion(float *output_data, const int32_t *input_indices, const float *input_data,
                                        int32_t inner_size, int32_t input_data_inner_size, int32_t outer_start,
                                        int32_t outer_end) {
  int index = 0;
  SIMD_RUN_NO_SCALAR(Fp32CastGatherReduceInt32Fusion, index, output_data, input_indices, input_data, inner_size,
                     input_data_inner_size, outer_start, outer_end);

  if (index < input_data_inner_size) {
    for (int i = outer_start; i < outer_end; i++) {
      float *result = output_data + i * input_data_inner_size + index;
      int32_t indice0 = input_indices[i * inner_size];
      for (int k = index; k < input_data_inner_size; k++) {
        result[k] = input_data[indice0 * input_data_inner_size + k];
      }
      for (int j = 1; j < inner_size; j++) {
        int32_t indice = input_indices[i * inner_size + j];
        for (int k = index; k < input_data_inner_size; k++) {
          result[k] += input_data[indice * input_data_inner_size + k];
        }
      }
    }
  }
  return NNACL_OK;
}
