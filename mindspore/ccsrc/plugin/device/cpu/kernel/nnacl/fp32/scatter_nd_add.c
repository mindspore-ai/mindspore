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

#include "nnacl/fp32/scatter_nd_add.h"
#include <string.h>
#include <stdio.h>
#include "nnacl/errorcode.h"
#include "nnacl/intrinsics/ms_simd_instructions.h"

int ScatterNDAdd(const void *update, void *output, int *output_unit_offsets, const ScatterNDParameter *param, int type,
                 int task_id) {
  if (update == NULL || output == NULL || output_unit_offsets == NULL || param == NULL) {
    return NNACL_NULL_PTR;
  }
  if (param->op_parameter.thread_num_ == 0) {
    return NNACL_ERR;
  }
  int unit_per_thread = UP_DIV(param->num_unit, param->op_parameter.thread_num_);
  int begin = unit_per_thread * task_id;
  int end = MSMIN(begin + unit_per_thread, param->num_unit);
  if (type == 0) {
    float *update_fp32 = (float *)update;
    float *output_fp32 = (float *)output;
    for (int i = begin; i < end; i++) {
      const float *update_data = update_fp32 + i * param->unit_size;
      float *output_data = output_fp32 + output_unit_offsets[i];
      int j = 0;

#ifdef ENABLE_ARM
      for (; j + C4NUM < param->unit_size; j += C4NUM) {
        MS_ST128_F32(output_data + j, MS_ADD128_F32(MS_LD128_F32(output_data + j), MS_LD128_F32(update_data + j)));
      }
#endif
#ifdef ENABLE_AVX
      for (; j + C8NUM < param->unit_size; j += C8NUM) {
        MS_ST256_F32(output_data + j, MS_ADD256_F32(MS_LD256_F32(output_data + j), MS_LD256_F32(update_data + j)));
      }
#endif
      for (; j < param->unit_size; j++) {
        output_data[j] += update_data[j];
      }
    }
  } else {
    int *update_int32 = (int *)update;
    int *output_int32 = (int *)output;
    for (int i = begin; i < end; i++) {
      const int *update_data = update_int32 + i * param->unit_size;
      int *output_data = output_int32 + output_unit_offsets[i];
      int j = 0;

#ifdef ENABLE_ARM
      for (; j + C4NUM < param->unit_size; j += C4NUM) {
        MS_ST128_EPI32(output_data + j,
                       MS_ADD128_EPI32(MS_LD128_EPI32(output_data + j), MS_LD128_EPI32(update_data + j)));
      }
#endif
#ifdef ENABLE_AVX
      for (; j + C8NUM < param->unit_size; j += C8NUM) {
        MS_ST256_EPI32(output_data + j,
                       MS_ADD256_EPI32(MS_LD256_EPI32(output_data + j), MS_LD256_EPI32(update_data + j)));
      }
#endif
      for (; j < param->unit_size; j++) {
        output_data[j] += update_data[j];
      }
    }
  }
  return NNACL_OK;
}
