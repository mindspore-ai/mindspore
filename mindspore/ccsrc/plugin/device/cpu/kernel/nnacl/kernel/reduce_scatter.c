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

#include "nnacl/kernel/reduce_scatter.h"
#include <float.h>
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/reduce_scatter_parameter.h"
#include "nnacl/tensor_c_utils.h"

void InitReduceScatterData(void *data, size_t data_num, int data_type, int mode) {
  if (data_type == kNumberTypeFloat32) {
    float *float_d = (float *)data;
    if (mode == Reduce_Prod) {
      for (size_t i = 0; i < data_num; i++) float_d[i] = 1.0f;
    } else if (mode == Reduce_Max) {
      for (size_t i = 0; i < data_num; i++) float_d[i] = FLT_MIN;
    } else if (mode == Reduce_Min) {
      for (size_t i = 0; i < data_num; i++) float_d[i] = FLT_MAX;
    } else {
      memset(data, 0, data_num * sizeof(float));
    }
  }
  return;
}

int DoReduceScatter(void *in_data, void *reduce_data, size_t data_size, int data_type, int mode, float rank_size) {
  if (data_type == kNumberTypeFloat32) {
    float *in = (float *)in_data;
    float *out = (float *)reduce_data;

    if (mode == Reduce_Sum) {
      for (size_t i = 0; i < data_size; i++) out[i] += in[i];
    } else if (mode == Reduce_Mean) {
      for (size_t i = 0; i < data_size; i++) out[i] += (in[i] / rank_size);
    } else if (mode == Reduce_Max) {
      for (size_t i = 0; i < data_size; i++) out[i] = in[i] > out[i] ? in[i] : out[i];
    } else if (mode == Reduce_Min) {
      for (size_t i = 0; i < data_size; i++) out[i] = in[i] < out[i] ? in[i] : out[i];
    } else {
      return NNACL_REDUCE_SCATTER_MODE_INVALID;
    }
  } else {
    return NNACL_REDUCE_SCATTER_DATA_TYPE_INVALID;
  }
  return NNACL_OK;
}

int ReduceScatterCompute(KernelBase *self) {
  TensorC *input = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  TensorC *output = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output);
  int rank = ((ReduceScatterParameter *)self->param_)->rank_size_;
  int mode = ((ReduceScatterParameter *)self->param_)->mode_;
  size_t in_data_size = GetSize(input);
  size_t in_ele_size = GetElementNum(input);
  size_t out_data_size = GetSize(output);
  int data_type = input->data_type_;

  void *reduce_data = self->env_->Alloc(self->env_->allocator_, in_data_size);
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(reduce_data);
  InitReduceScatterData(reduce_data, in_ele_size, data_type, mode);

  for (int i = 0; i < rank; i++) {
    /* update in_tensor by rank id */
    DoReduceScatter(input->data_, reduce_data, in_ele_size, data_type, mode, rank);
  }

  for (int i = 0; i < rank; i++) {
    /* update out_tensor by rank id */
    memcpy(output->data_, (uint8_t *)reduce_data + i * out_data_size, out_data_size);
  }

  self->env_->Free(self->env_->allocator_, reduce_data);
  return NNACL_OK;
}

KernelBase *CreateReduceScatter(OpParameter *param, int data_type) {
  ReduceScatterStruct *reduce_scatter = (ReduceScatterStruct *)malloc(sizeof(ReduceScatterStruct));
  NNACL_CHECK_NULL_RETURN_NULL(reduce_scatter);
  reduce_scatter->base_.Release = DefaultRelease;
  reduce_scatter->base_.Prepare = DefaultPrepare1In1Out;
  reduce_scatter->base_.Resize = DefaultResize;
  reduce_scatter->base_.Compute = ReduceScatterCompute;
  return (KernelBase *)reduce_scatter;
}

REG_KERNEL_CREATOR(PrimType_ReduceScatter, kNumberTypeFloat32, CreateReduceScatter)
