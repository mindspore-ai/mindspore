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

#include "nnacl/kernel/slice.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/base/slice_base.h"
#include "nnacl/nnacl_common.h"

int SliceLaunch(void *cdata, int task_id, float l, float r) {
  SliceStruct *slice = (SliceStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(slice);
  void *in_data = slice->base_.in_[FIRST_INPUT]->data_;
  void *out_data = slice->base_.out_[OUTPUT_INDEX]->data_;
  DoSlice(in_data, out_data, slice, task_id, slice->base_.thread_nr_, slice->data_type_size_);
  return NNACL_OK;
}

int SliceResize(KernelBase *self) {
  SliceStruct *slice = (SliceStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(slice);

  InitSliceStruct(slice, self->in_[Index0], self->in_[Index1], self->in_[Index2]);

  if (slice->param_length_ < DIMENSION_8D) {
    PadSliceParameterTo8D(slice);
  }
  return NNACL_OK;
}

int SliceCompute(KernelBase *self) {
  TensorC *in_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in_tensor);
  TensorC *out_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(out_tensor);

  SliceStruct *slice = (SliceStruct *)self;
  if (slice->size_[Index5] < self->thread_nr_) {
    DoSliceNoParallel(in_tensor->data_, out_tensor->data_, slice, slice->data_type_size_);
    return NNACL_OK;
  }

  int ret = self->env_->ParallelLaunch(self->env_->thread_pool_, SliceLaunch, self, self->thread_nr_);
  if (ret != NNACL_OK) {
    return ret;
  }
  return NNACL_OK;
}

KernelBase *CreateSlice(OpParameter *param, int data_type) {
  SliceStruct *slice = (SliceStruct *)malloc(sizeof(SliceStruct));
  NNACL_CHECK_NULL_RETURN_NULL(slice);
  slice->data_type_size_ = DataTypeCSize(data_type);
  slice->base_.Release = DefaultRelease;
  slice->base_.Prepare = DefaultPrepare3In1Out;
  slice->base_.Resize = SliceResize;
  slice->base_.Compute = SliceCompute;
  return (KernelBase *)slice;
}

REG_KERNEL_CREATOR(PrimType_SliceFusion, kNumberTypeInt32, CreateSlice)
REG_KERNEL_CREATOR(PrimType_SliceFusion, kNumberTypeFloat32, CreateSlice)
REG_KERNEL_CREATOR(PrimType_SliceFusion, kNumberTypeFloat16, CreateSlice)
