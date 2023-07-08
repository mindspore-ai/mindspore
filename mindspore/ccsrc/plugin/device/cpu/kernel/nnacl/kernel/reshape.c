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

#include "nnacl/kernel/reshape.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/tensor_c_utils.h"

int kMinCostPerThread = 16384;

int ParallelReshape(void *param, int task_id, float l, float r) {
  NNACL_CHECK_NULL_RETURN_ERR(param);
  ReshapeStruct *reshape = (ReshapeStruct *)param;

  uint8_t *in_start = (uint8_t *)(reshape->base_.in_[0]->data_) + task_id * reshape->block_size_;
  uint8_t *out_start = (uint8_t *)(reshape->base_.out_[0]->data_) + task_id * reshape->block_size_;
  int copy_size = reshape->block_size_;
  if (task_id == (reshape->base_.thread_nr_ - 1)) {
    copy_size = reshape->total_size_ - task_id * reshape->block_size_;
  }
  (void)memcpy(out_start, in_start, copy_size);
  return NNACL_OK;
}

int ReshapeResize(struct KernelBase *self) {
  NNACL_CHECK_NULL_RETURN_ERR(self);
  ReshapeStruct *reshape = (ReshapeStruct *)self;
  reshape->total_size_ = GetSize(self->in_[0]);
  if (reshape->total_size_ == 0) {
    return NNACL_OK;
  }

  self->thread_nr_ = MSMIN(self->thread_nr_, UP_DIV(reshape->total_size_, kMinCostPerThread));
  if (self->thread_nr_ < 1) {
    self->thread_nr_ = 1;
  }
  NNACL_CHECK_ZERO_RETURN_ERR(self->thread_nr_);
  reshape->block_size_ = UP_DIV(reshape->total_size_, self->thread_nr_);
  NNACL_CHECK_ZERO_RETURN_ERR(reshape->block_size_);
  self->thread_nr_ = UP_DIV(reshape->total_size_, reshape->block_size_);

  return NNACL_OK;
}

int ReshapeCompute(struct KernelBase *self) {
  return self->env_->ParallelLaunch(self->env_->thread_pool_, ParallelReshape, self, self->thread_nr_);
}

KernelBase *CreateReshape(OpParameter *param, int data_type) {
  ReshapeStruct *reshape = (ReshapeStruct *)malloc(sizeof(ReshapeStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(reshape);
  reshape->base_.Release = DefaultRelease;
  reshape->base_.Prepare = DefaultPrepare1In1Out;
  reshape->base_.Resize = ReshapeResize;
  reshape->base_.Compute = ReshapeCompute;
  return (KernelBase *)reshape;
}

REG_KERNEL_CREATOR(PrimType_Reshape, kNumberTypeInt32, CreateReshape)
REG_KERNEL_CREATOR(PrimType_Reshape, kNumberTypeFloat32, CreateReshape)
REG_KERNEL_CREATOR(PrimType_Reshape, kNumberTypeFloat16, CreateReshape)
REG_KERNEL_CREATOR(PrimType_Reshape, kNumberTypeBool, CreateReshape)
REG_KERNEL_CREATOR(PrimType_Flatten, kNumberTypeInt32, CreateReshape)
REG_KERNEL_CREATOR(PrimType_Flatten, kNumberTypeFloat16, CreateReshape)
REG_KERNEL_CREATOR(PrimType_Flatten, kNumberTypeFloat32, CreateReshape)
REG_KERNEL_CREATOR(PrimType_FlattenGrad, kNumberTypeFloat16, CreateReshape)
REG_KERNEL_CREATOR(PrimType_FlattenGrad, kNumberTypeFloat32, CreateReshape)
REG_KERNEL_CREATOR(PrimType_ExpandDims, kNumberTypeInt32, CreateReshape)
REG_KERNEL_CREATOR(PrimType_ExpandDims, kNumberTypeFloat16, CreateReshape)
REG_KERNEL_CREATOR(PrimType_ExpandDims, kNumberTypeFloat32, CreateReshape)
REG_KERNEL_CREATOR(PrimType_ExpandDims, kNumberTypeBool, CreateReshape)
REG_KERNEL_CREATOR(PrimType_ExpandDims, kNumberTypeInt8, CreateReshape)
REG_KERNEL_CREATOR(PrimType_Squeeze, kNumberTypeFloat32, CreateReshape)
REG_KERNEL_CREATOR(PrimType_Squeeze, kNumberTypeFloat16, CreateReshape)
REG_KERNEL_CREATOR(PrimType_Squeeze, kNumberTypeInt32, CreateReshape)
REG_KERNEL_CREATOR(PrimType_Squeeze, kNumberTypeBool, CreateReshape)
REG_KERNEL_CREATOR(PrimType_Unsqueeze, kNumberTypeFloat16, CreateReshape)
REG_KERNEL_CREATOR(PrimType_Unsqueeze, kNumberTypeFloat32, CreateReshape)
REG_KERNEL_CREATOR(PrimType_Unsqueeze, kNumberTypeUInt8, CreateReshape)
REG_KERNEL_CREATOR(PrimType_Unsqueeze, kNumberTypeInt32, CreateReshape)
REG_KERNEL_CREATOR(PrimType_Unsqueeze, kNumberTypeInt64, CreateReshape)
REG_KERNEL_CREATOR(PrimType_Unsqueeze, kNumberTypeBool, CreateReshape)
