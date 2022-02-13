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

#include "nnacl/fp32/embedding_lookup_fp32.h"
#include <string.h>
#include "nnacl/errorcode.h"

void l2_regulate(float *data, int size, float max_norm) {
  float sum = 0;
  for (int i = 0; i < size; ++i) {
    sum += data[i];
  }
  if (sum != 0) {
    for (int i = 0; i < size; ++i) {
      data[i] *= max_norm / sum;
    }
  }
  return;
}

int CopyData(float *input_data, const int *ids, float *output_data, int num,
             const EmbeddingLookupParameter *parameter) {
  if (ids[num] >= parameter->layer_num_ || ids[num] < 0) {
    return NNACL_ERRCODE_INDEX_OUT_OF_RANGE;
  }
  float *out_data = output_data + num * parameter->layer_size_;
  float *in_data = input_data + ids[num] * parameter->layer_size_;
  if (!parameter->is_regulated_[ids[num]]) {
    l2_regulate(in_data, parameter->layer_size_, parameter->max_norm_);
    parameter->is_regulated_[ids[num]] = true;
  }

  memcpy(out_data, in_data, sizeof(float) * (size_t)(parameter->layer_size_));
  return NNACL_OK;
}

int EmbeddingLookup(float *input_data, const int *ids, float *output_data, const EmbeddingLookupParameter *parameter,
                    int task_id) {
  if (parameter->op_parameter_.thread_num_ == 0) {
    return NNACL_PARAM_INVALID;
  }
  for (int i = task_id; i < parameter->ids_size_; i += parameter->op_parameter_.thread_num_) {
    int ret = CopyData(input_data, ids, output_data, i, parameter);
    if (ret != NNACL_OK) {
      return ret;
    }
  }
  return NNACL_OK;
}
