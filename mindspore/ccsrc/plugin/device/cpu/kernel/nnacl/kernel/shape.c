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

int ShapeCompute(struct KernelBase *self) {
  ShapeStruct *shape = (ShapeStruct *)self;
  memcpy(self->out_[OUTPUT_INDEX]->data_, self->in_[FIRST_INPUT]->shape_, shape->shape_size_);
  return NNACL_OK;
}

int ShapeResize(KernelBase *self) {
  NNACL_CHECK_NULL_RETURN_ERR(self->in_[FIRST_INPUT]);
  NNACL_CHECK_NULL_RETURN_ERR(self->out_[OUTPUT_INDEX]);
  ShapeStruct *shape = (ShapeStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(shape);
  shape->shape_size_ = self->in_[FIRST_INPUT]->shape_size_ * sizeof(int);
  return NNACL_OK;
}

KernelBase *CreateShape(OpParameter *param, int data_type) {
  ShapeStruct *shape = (ShapeStruct *)malloc(sizeof(ShapeStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(shape);
  shape->base_.Release = DefaultRelease;
  shape->base_.Prepare = DefaultPrepare1In1Out;
  shape->base_.Resize = ShapeResize;
  shape->base_.Compute = ShapeCompute;
  return (KernelBase *)shape;
}

REG_KERNEL_CREATOR(PrimType_Shape, kNumberTypeInt32, CreateShape)
REG_KERNEL_CREATOR(PrimType_Shape, kNumberTypeBool, CreateShape)
REG_KERNEL_CREATOR(PrimType_Shape, kNumberTypeFloat16, CreateShape)
REG_KERNEL_CREATOR(PrimType_Shape, kNumberTypeFloat32, CreateShape)
REG_KERNEL_CREATOR(PrimType_Shape, kNumberTypeInt8, CreateShape)
REG_KERNEL_CREATOR(PrimType_Shape, kNumberTypeUInt8, CreateShape)
REG_KERNEL_CREATOR(PrimType_Shape, kNumberTypeInt64, CreateShape)
