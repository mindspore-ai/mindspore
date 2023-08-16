/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
// * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "nnacl/fp16/sparse_to_dense_fp16.h"
#include "nnacl/errorcode.h"

int SparseToDenseSetDefaultFp16(float16_t *output, float16_t default_value, SparseToDenseParameter *param,
                                int task_id) {
  if (output == NULL || param == NULL) {
    return NNACL_NULL_PTR;
  }
  if (param->op_parameter_.thread_num_ == 0) {
    return NNACL_ERR;
  }
  int unit_per_thread = UP_DIV(param->output_num, param->op_parameter_.thread_num_);
  int begin = unit_per_thread * task_id;
  int end = MSMIN(begin + unit_per_thread, param->output_num);
  for (int i = begin; i < end; i++) {
    output[i] = default_value;
  }
  return NNACL_OK;
}

int SparseToDenseFp16(int *indices_vec, const float16_t *sparse_values, float16_t default_value, float16_t *output,
                      SparseToDenseParameter *param, int task_id) {
  if (indices_vec == NULL || sparse_values == NULL || output == NULL || param == NULL) {
    return NNACL_NULL_PTR;
  }
  if (param->op_parameter_.thread_num_) {
    return NNACL_ERR;
  }
  int unit_per_thread = UP_DIV(param->index_num, param->op_parameter_.thread_num_);
  int begin = unit_per_thread * task_id;
  int end = MSMIN(begin + unit_per_thread, param->index_num);

  int stride0 = param->output_stride[0];
  int stride1 = param->output_stride[1];
  int stride2 = param->output_stride[2];

  if (param->validate_indices_ == true) {
    int index_before = -1;
    for (int i = begin; i < end; i++) {
      int *indices = indices_vec + i * DIMENSION_4D;
      int index = stride0 * indices[0] + stride1 * indices[1] + stride2 * indices[2] + indices[3];
      if (index <= index_before) {
        return NNACL_ERR;
      }
      index_before = index;
    }
  }

  if (param->is_scalar == true) {
    for (int i = begin; i < end; i++) {
      int *indices = indices_vec + i * DIMENSION_4D;
      int index = stride0 * indices[0] + stride1 * indices[1] + stride2 * indices[2] + indices[3];
      output[index] = sparse_values[0];
    }
  } else {
    for (int i = begin; i < end; i++) {
      int *indices = indices_vec + i * DIMENSION_4D;
      int index = stride0 * indices[0] + stride1 * indices[1] + stride2 * indices[2] + indices[3];
      output[index] = sparse_values[i];
    }
  }
  return NNACL_OK;
}
