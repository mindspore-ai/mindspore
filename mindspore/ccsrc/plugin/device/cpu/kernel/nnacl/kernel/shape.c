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

#include "nnacl/kernel/shape.h"
#include "nnacl/kernel/default_kernel_base.h"

int shape_compute(struct KernelBase *self) {
  TensorC *in = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in);
  TensorC *out = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(out);
  NNACL_CHECK_NULL_RETURN_ERR(out->data_);

  for (size_t i = 0; i < in->shape_size_; i++) {
    ((int *)out->data_)[i] = in->shape_[i];
  }
  return NNACL_OK;
}

KernelBase *CreateShape(OpParameter *param, int data_type) {
  ShapeStruct *shape = (ShapeStruct *)malloc(sizeof(ShapeStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(shape);
  shape->base_.release_ = DefaultRelease;
  shape->base_.prepare_ = DefaultPrepare1In1Out;
  shape->base_.resize_ = DefaultResize;
  shape->base_.compute_ = shape_compute;
  return (KernelBase *)shape;
}

REG_KERNEL_CREATOR(PrimType_Shape, kNumberTypeInt32, CreateShape)
REG_KERNEL_CREATOR(PrimType_Shape, kNumberTypeBool, CreateShape)
REG_KERNEL_CREATOR(PrimType_Shape, kNumberTypeFloat16, CreateShape)
REG_KERNEL_CREATOR(PrimType_Shape, kNumberTypeFloat32, CreateShape)
REG_KERNEL_CREATOR(PrimType_Shape, kNumberTypeInt8, CreateShape)
REG_KERNEL_CREATOR(PrimType_Shape, kNumberTypeUInt8, CreateShape)
REG_KERNEL_CREATOR(PrimType_Shape, kNumberTypeInt64, CreateShape)
