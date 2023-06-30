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

#include "nnacl/kernel/ones_like.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/tensor_c_utils.h"

#define ApproximateOnesLike(output, data_size) \
  for (size_t i = 0; i < data_size; ++i) {     \
    output[i] = 1;                             \
  }

int OnesLikeCompute(KernelBase *self) {
  NNACL_CHECK_NULL_RETURN_ERR(self);
  TensorC *output_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);
  void *output_ptr = output_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_ptr);
  size_t num = (size_t)GetElementNum(output_tensor);

  if (output_tensor->data_type_ == kNumberTypeFloat32) {
    float *output = (float *)output_ptr;
    ApproximateOnesLike(output, num);
    return NNACL_OK;
  }
#ifdef ENABLE_FP16
  if (output_tensor->data_type_ == kNumberTypeFloat16) {
    float16_t *output = (float16_t *)output_ptr;
    ApproximateOnesLike(output, num);
    return NNACL_OK;
  }
#endif
  if (output_tensor->data_type_ == kNumberTypeInt32) {
    int *output = (int *)output_ptr;
    ApproximateOnesLike(output, num);
    return NNACL_OK;
  }
  return NNACL_UNSUPPORTED_DATA_TYPE;
}

KernelBase *CreateOnesLike(OpParameter *param, int data_type) {
  OnesLikeStruct *ones_like = (OnesLikeStruct *)malloc(sizeof(OnesLikeStruct));
  NNACL_CHECK_NULL_RETURN_NULL(ones_like);
  ones_like->data_type_ = data_type;
  ones_like->base_.Release = DefaultRelease;
  ones_like->base_.Prepare = DefaultPrepare1In1Out;
  ones_like->base_.Resize = DefaultResize;
  ones_like->base_.Compute = OnesLikeCompute;
  return (KernelBase *)ones_like;
}

REG_KERNEL_CREATOR(PrimType_OnesLike, kNumberTypeInt32, CreateOnesLike)
REG_KERNEL_CREATOR(PrimType_OnesLike, kNumberTypeFloat32, CreateOnesLike)
REG_KERNEL_CREATOR(PrimType_OnesLike, kNumberTypeFloat16, CreateOnesLike)
