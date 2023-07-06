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

#include "nnacl/kernel/size.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/tensor_c_utils.h"

int SizeCompute(KernelBase *self) {
  TensorC *in_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in_tensor);
  TensorC *out_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(out_tensor);
  int *out_data = (int *)out_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(out_data);
  out_data[Index0] = GetElementNum(in_tensor);
  return NNACL_OK;
}

KernelBase *CreateSize(OpParameter *param, int data_type) {
  SizeStruct *size = (SizeStruct *)malloc(sizeof(SizeStruct));
  NNACL_CHECK_NULL_RETURN_NULL(size);
  size->base_.Release = DefaultRelease;
  size->base_.Prepare = DefaultPrepare1In1Out;
  size->base_.Resize = DefaultResize;
  size->base_.Compute = SizeCompute;
  return (KernelBase *)size;
}

REG_KERNEL_CREATOR(PrimType_Size, kNumberTypeInt32, CreateSize)
REG_KERNEL_CREATOR(PrimType_Size, kNumberTypeFloat32, CreateSize)
REG_KERNEL_CREATOR(PrimType_Size, kNumberTypeFloat16, CreateSize)
