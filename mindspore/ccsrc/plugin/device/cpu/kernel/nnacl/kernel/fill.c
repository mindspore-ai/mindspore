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

#include "nnacl/kernel/fill.h"
#include "nnacl/fill_parameter.h"
#include "nnacl/op_base.h"
#include "nnacl/nnacl_common.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/base/fill_base.h"
#include "nnacl/kernel/default_kernel_base.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/fill_fp16.h"
#endif

int FillResize(struct KernelBase *self) {
  FillStruct *fill = (FillStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(fill);
  fill->base_.thread_nr_ = fill->base_.UpdateThread(TC_PTYPE(PrimType_Fill), 0, 1,
                                                    GetSize(fill->base_.out_[OUTPUT_INDEX]), fill->base_.thread_nr_);

  NNACL_CHECK_NULL_RETURN_ERR(fill->base_.out_[OUTPUT_INDEX]);
  fill->data_size_ = (int)GetElementNum(fill->base_.out_[OUTPUT_INDEX]);
  fill->thread_sz_count_ = MSMIN(fill->base_.thread_nr_, fill->data_size_);
  if (fill->thread_sz_count_ != 0) {
    fill->thread_sz_stride_ = UP_DIV(fill->data_size_, fill->thread_sz_count_);
  }
  return NNACL_OK;
}

int FillImpl(void *cdata, int task_id, float l, float r) {
  FillStruct *fill = (FillStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(fill);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(task_id, fill->thread_sz_stride_, NNACL_ERR);
  int size = MSMIN(fill->thread_sz_stride_, fill->data_size_ - task_id * fill->thread_sz_stride_);
  NNACL_CHECK_FALSE(size <= 0, NNACL_OK);
  int offset = task_id * fill->thread_sz_stride_;
  int ret = NNACL_OK;
  switch (fill->base_.in_[FIRST_INPUT]->data_type_) {
#ifdef ENABLE_FP16
    case kNumberTypeFloat16:
      ret = FillFp16((float16_t *)fill->out_ptr_ + offset, size, ((float16_t *)fill->src_data_)[FIRST_INPUT]);
      break;
#endif
    case kNumberTypeFloat32:
      ret = FillFp32((float *)fill->out_ptr_ + offset, size, ((float *)fill->src_data_)[FIRST_INPUT]);
      break;
    case kNumberTypeInt32:
      ret = FillInt32((int *)fill->out_ptr_ + offset, size, ((int *)fill->src_data_)[FIRST_INPUT]);
      break;
    case kNumberTypeBool:
      ret = FillBool((bool *)fill->out_ptr_ + offset, size, ((bool *)fill->src_data_)[FIRST_INPUT]);
      break;
    default:
      return NNACL_FILL_DATA_TYPE_INVALID;
  }
  return ret;
}

int FillCompute(struct KernelBase *self) {
  FillStruct *fill = (FillStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(fill);

  fill->src_data_ = (void *)fill->base_.in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(fill->src_data_);
  fill->out_ptr_ = (void *)fill->base_.out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(fill->out_ptr_);

  return self->env_->ParallelLaunch(self->env_->thread_pool_, FillImpl, fill, fill->base_.thread_nr_);
}

KernelBase *CreateFill(OpParameter *param, int data_type) {
  FillStruct *fill = (FillStruct *)malloc(sizeof(FillStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(fill);
  fill->base_.Prepare = DefaultPrepare2In1Out;
  fill->base_.Resize = FillResize;
  fill->base_.Release = DefaultRelease;
  fill->base_.Compute = FillCompute;
  return (KernelBase *)fill;
}

REG_KERNEL_CREATOR(PrimType_Fill, kNumberTypeBool, CreateFill);
REG_KERNEL_CREATOR(PrimType_Fill, kNumberTypeInt32, CreateFill);
REG_KERNEL_CREATOR(PrimType_Fill, kNumberTypeFloat32, CreateFill);
REG_KERNEL_CREATOR(PrimType_Fill, kNumberTypeFloat16, CreateFill);

REG_KERNEL_CREATOR(PrimType_FillV2, kNumberTypeBool, CreateFill);
REG_KERNEL_CREATOR(PrimType_FillV2, kNumberTypeInt32, CreateFill);
REG_KERNEL_CREATOR(PrimType_FillV2, kNumberTypeFloat32, CreateFill);
REG_KERNEL_CREATOR(PrimType_FillV2, kNumberTypeFloat16, CreateFill);
