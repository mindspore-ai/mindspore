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

#include "nnacl/kernel/ragged_range.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/fp32/ragged_range_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/ragged_range_fp16.h"
#endif

int RaggedRangeCompute(KernelBase *self) {
  RaggedRangeStruct *ragged_range = (RaggedRangeStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(ragged_range);

  TensorC *input0 = self->in_[Index0];
  TensorC *input1 = self->in_[Index1];
  TensorC *input2 = self->in_[Index2];
  TensorC *output0 = self->out_[Index0];
  TensorC *output1 = self->out_[Index1];

  if (input0->data_type_ == kNumberTypeFloat32) {
    RaggedRangeFp32((float *)input0->data_, (float *)input1->data_, (float *)input2->data_, (int *)output0->data_,
                    (float *)output1->data_, ragged_range);
  } else if (input0->data_type_ == kNumberTypeInt32) {
    RaggedRangeInt((int *)input0->data_, (int *)input1->data_, (int *)input2->data_, (int *)output0->data_,
                   (int *)output1->data_, ragged_range);
  } else if (input0->data_type_ == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
    RaggedRangeFp16((float16_t *)input0->data_, (float16_t *)input1->data_, (float16_t *)input2->data_,
                    (int *)output0->data_, (float16_t *)output1->data_, ragged_range);
#endif
  } else {
    return NNACL_UNSUPPORTED_DATA_TYPE;
  }
  return NNACL_OK;
}

int RaggedRangeResize(KernelBase *self) {
  RaggedRangeStruct *ragged_range = (RaggedRangeStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(ragged_range);

  ragged_range->rows_ = self->out_[OUTPUT_INDEX]->shape_[Index0] - 1;
  ragged_range->starts_is_scalar_ = self->in_[FIRST_INPUT]->shape_size_ == 0;
  ragged_range->limits_is_scalar_ = self->in_[SECOND_INPUT]->shape_size_ == 0;
  ragged_range->deltas_is_scalar_ = self->in_[THIRD_INPUT]->shape_size_ == 0;
  return NNACL_OK;
}

KernelBase *CreateRaggedRange(OpParameter *param, int data_type) {
  RaggedRangeStruct *ragged_range = (RaggedRangeStruct *)malloc(sizeof(RaggedRangeStruct));
  NNACL_CHECK_NULL_RETURN_NULL(ragged_range);
  ragged_range->base_.Release = DefaultRelease;
  ragged_range->base_.Prepare = DefaultPrepare3In2Out;
  ragged_range->base_.Resize = RaggedRangeResize;
  ragged_range->base_.Compute = RaggedRangeCompute;
  return (KernelBase *)ragged_range;
}

REG_KERNEL_CREATOR(PrimType_RaggedRange, kNumberTypeInt32, CreateRaggedRange)
REG_KERNEL_CREATOR(PrimType_RaggedRange, kNumberTypeFloat16, CreateRaggedRange)
REG_KERNEL_CREATOR(PrimType_RaggedRange, kNumberTypeFloat32, CreateRaggedRange)
