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

#include "nnacl/kernel/zeros_like.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/common_func.h"
#include "nnacl/tensor_c_utils.h"

int ZerosLikeCompute(KernelBase *self) {
  NNACL_CHECK_NULL_RETURN_ERR(self);
  TensorC *output = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output);
  NNACL_CHECK_NULL_RETURN_ERR(output->data_);
  (void)memset(output->data_, 0, GetSize(output));
  return NNACL_OK;
}

KernelBase *CreateZerosLike(OpParameter *param, int data_type) {
  ZerosLikeStruct *zeros_like = (ZerosLikeStruct *)malloc(sizeof(ZerosLikeStruct));
  NNACL_CHECK_NULL_RETURN_NULL(zeros_like);
  zeros_like->base_.Release = DefaultRelease;
  zeros_like->base_.Prepare = DefaultPrepare1In1Out;
  zeros_like->base_.Resize = DefaultResize;
  zeros_like->base_.Compute = ZerosLikeCompute;
  return (KernelBase *)zeros_like;
}

REG_KERNEL_CREATOR(PrimType_ZerosLike, kNumberTypeInt32, CreateZerosLike)
REG_KERNEL_CREATOR(PrimType_ZerosLike, kNumberTypeFloat32, CreateZerosLike)
REG_KERNEL_CREATOR(PrimType_ZerosLike, kNumberTypeFloat16, CreateZerosLike)
