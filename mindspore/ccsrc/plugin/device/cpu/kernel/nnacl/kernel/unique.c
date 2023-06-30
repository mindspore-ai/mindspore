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

#include "nnacl/kernel/unique.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/fp32/unique_fp32.h"
#include "nnacl/tensor_c_utils.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/unique_fp16.h"
#endif

int UniqueCompute(KernelBase *self) {
  TensorC *input = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  TensorC *output0 = self->out_[Index0];
  NNACL_CHECK_NULL_RETURN_ERR(output0);
  TensorC *output1 = self->out_[Index1];
  NNACL_CHECK_NULL_RETURN_ERR(output1);

  int num = GetElementNum(input);
  int output0_len = 0;

#ifdef ENABLE_FP16
  if (input->data_type_ == kNumberTypeFloat16) {
    UniqueFp16((float16_t *)input->data_, num, (float16_t *)output0->data_, &output0_len, (int *)output1->data_);
  }
#endif
  if (input->data_type_ == kNumberTypeInt32) {
    UniqueInt((int *)input->data_, num, (int *)output0->data_, &output0_len, (int *)output1->data_);
  }
  if (input->data_type_ == kNumberTypeFloat32) {
    Unique((float *)input->data_, num, (float *)output0->data_, &output0_len, (int *)output1->data_);
  }

  output0->shape_changed_ = (output0->shape_[output0->shape_size_ - 1] != output0_len);
  output0->shape_[output0->shape_size_ - 1] = output0_len;
  return NNACL_OK;
}

KernelBase *CreateUnique(OpParameter *param, int data_type) {
  UniqueStruct *unique = (UniqueStruct *)malloc(sizeof(UniqueStruct));
  NNACL_CHECK_NULL_RETURN_NULL(unique);
  unique->data_type_ = data_type;
  unique->base_.Release = DefaultRelease;
  unique->base_.Prepare = DefaultPrepare1In2Out;
  unique->base_.Resize = DefaultResize;
  unique->base_.Compute = UniqueCompute;
  return (KernelBase *)unique;
}

REG_KERNEL_CREATOR(PrimType_Unique, kNumberTypeInt32, CreateUnique)
REG_KERNEL_CREATOR(PrimType_Unique, kNumberTypeFloat32, CreateUnique)
REG_KERNEL_CREATOR(PrimType_Unique, kNumberTypeFloat16, CreateUnique)
