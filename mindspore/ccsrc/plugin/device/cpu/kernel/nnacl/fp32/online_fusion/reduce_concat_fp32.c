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

#include "nnacl/fp32/online_fusion/reduce_concat_fp32.h"
#include <float.h>
#include "nnacl/reduce_concat_fp32_simd.h"
#include "nnacl/errorcode.h"

int64_t Fp32ReduceSumConcatAxisSizeAVX512Fusion(float *output_data, float **input_datas,
                                                const int64_t *reduce_axis_size, int64_t input_nums, int64_t batch,
                                                int64_t batch_tile_size, int64_t inner_tile, int64_t thread_num,
                                                int64_t task_id) {
  int64_t single_thread_tile = DOWN_DIV(batch, thread_num);
  int64_t less_tile = batch - thread_num * single_thread_tile;

  int64_t batch_start = task_id * single_thread_tile;
  if (task_id < less_tile) {
    single_thread_tile += 1;
    batch_start += task_id;
  } else {
    batch_start += less_tile;
  }
  int64_t batch_end = batch_start + single_thread_tile;
  int64_t last_inner_size = batch_tile_size - (input_nums - 1) * inner_tile;

  int res = NNACL_OK;
  if (inner_tile == C16NUM) {
    for (int i = batch_start; i < batch_end; i++) {
      float *result = output_data + i * batch_tile_size;
      for (size_t j = 0; j < input_nums - 1; j++) {
        SIMD_RUN_AVX512(Fp32ReduceSumConcatAxisSize16Fusion, res, result,
                        input_datas[j] + i * C16NUM * reduce_axis_size[j], reduce_axis_size[j]);
        result += C16NUM;
      }
      (void)memcpy(result, input_datas[input_nums - 1] + i * last_inner_size, last_inner_size * sizeof(float));
    }
  } else if (inner_tile == C32NUM) {
    for (int i = batch_start; i < batch_end; i++) {
      float *result = output_data + i * batch_tile_size;
      for (size_t j = 0; j < input_nums - 1; j++) {
        SIMD_RUN_AVX512(Fp32ReduceSumConcatAxisSize32Fusion, res, result,
                        input_datas[j] + i * C32NUM * reduce_axis_size[j], reduce_axis_size[j]);
        result += C32NUM;
      }
      (void)memcpy(result, input_datas[input_nums - 1] + i * last_inner_size, last_inner_size * sizeof(float));
    }
  } else if (inner_tile == C64NUM) {
    for (int i = batch_start; i < batch_end; i++) {
      float *result = output_data + i * batch_tile_size;
      for (size_t j = 0; j < input_nums - 1; j++) {
        SIMD_RUN_AVX512(Fp32ReduceSumConcatAxisSize64Fusion, res, result,
                        input_datas[j] + i * C64NUM * reduce_axis_size[j], reduce_axis_size[j]);
        result += C64NUM;
      }
      (void)memcpy(result, input_datas[input_nums - 1] + i * last_inner_size, last_inner_size * sizeof(float));
    }
  } else if (inner_tile == C128NUM) {
    for (int i = batch_start; i < batch_end; i++) {
      float *result = output_data + i * batch_tile_size;
      for (size_t j = 0; j < input_nums - 1; j++) {
        SIMD_RUN_AVX512(Fp32ReduceSumConcatAxisSize128Fusion, res, result,
                        input_datas[j] + i * C128NUM * reduce_axis_size[j], reduce_axis_size[j]);
        result += C128NUM;
      }
      (void)memcpy(result, input_datas[input_nums - 1] + i * last_inner_size, last_inner_size * sizeof(float));
    }
  }
  return res;
}

int64_t Fp32ReduceSumConcatFusion(float *output_data, float **input_datas, const int64_t *reduce_axis_size,
                                  int64_t input_nums, int64_t batch, int64_t batch_tile_size, int64_t inner_tile,
                                  int64_t thread_num, int64_t task_id) {
  AVX512_HARDWARE_SELF_AWARENESS_BEGIN;
  if (inner_tile == C16NUM || inner_tile == C32NUM || inner_tile == C64NUM || inner_tile == C128NUM) {
    return Fp32ReduceSumConcatAxisSizeAVX512Fusion(output_data, input_datas, reduce_axis_size, input_nums, batch,
                                                   batch_tile_size, inner_tile, thread_num, task_id);
  }
  AVX512_HARDWARE_SELF_AWARENESS_END;

  int64_t single_thread_tile = DOWN_DIV(batch, thread_num);
  int64_t less_tile = batch - thread_num * single_thread_tile;

  int64_t batch_start = task_id * single_thread_tile;
  if (task_id < less_tile) {
    batch_start += task_id;
    single_thread_tile += 1;
  } else {
    batch_start += less_tile;
  }
  int64_t batch_end = batch_start + single_thread_tile;
  for (int i = batch_start; i < batch_end; i++) {
    float *result = output_data + i * batch_tile_size;
    for (size_t j = 0; j < input_nums - 1; j++) {
      const float *input_data_ptr = input_datas[j] + i * inner_tile * reduce_axis_size[j];

      for (int k = 0; k < inner_tile; k++) {
        result[k] = input_data_ptr[k];
        for (int l = 1; l < reduce_axis_size[j]; l++) {
          result[k] += input_data_ptr[l * inner_tile + k];
        }
      }
      result += inner_tile;
    }

    int64_t inner_size2 = batch_tile_size - (input_nums - 1) * inner_tile;
    const float *input_data_ptr = input_datas[input_nums - 1] + i * inner_size2;
    (void)memcpy(result, input_data_ptr, inner_size2 * sizeof(float));
  }
  return NNACL_OK;
}
