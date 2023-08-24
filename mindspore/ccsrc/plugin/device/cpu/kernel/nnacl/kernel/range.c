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

#include "nnacl/kernel/range.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/range_parameter.h"
#include "nnacl/fp32/range_fp32.h"
#include "nnacl/tensor_c_utils.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/range_fp16.h"
#endif

int RangeCompute(KernelBase *self) {
  NNACL_CHECK_NULL_RETURN_ERR(self);

  TensorC *input = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  TensorC *output = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output);
  int output_num = GetElementNum(output);

  if (self->in_size_ == THREE_TENSOR) {
    TensorC *delta = self->in_[THIRD_INPUT];
    NNACL_CHECK_NULL_RETURN_ERR(delta);

    if (input->data_type_ == kNumberTypeFloat32) {
      Range((float *)output->data_, *(float *)input->data_, *(float *)delta->data_, output_num);
    } else if (input->data_type_ == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
      RangeFp16((float16_t *)output->data_, *(float16_t *)input->data_, *(float16_t *)delta->data_, output_num);
#endif
    } else if (input->data_type_ == kNumberTypeInt32) {
      RangeInt((int *)output->data_, *(int *)input->data_, *(int *)delta->data_, output_num);
    } else {
      return NNACL_UNSUPPORTED_DATA_TYPE;
    }
  } else {
    if (input->data_type_ == kNumberTypeInt32) {
      RangeParameter *param = (RangeParameter *)self->param_;
      NNACL_CHECK_NULL_RETURN_ERR(param);
      RangeInt((int *)output->data_, param->start_, param->delta_, output_num);
    } else {
      return NNACL_UNSUPPORTED_DATA_TYPE;
    }
  }
  return NNACL_OK;
}

KernelBase *CreateRange(OpParameter *param, int data_type) {
  RangeStruct *range = (RangeStruct *)malloc(sizeof(RangeStruct));
  NNACL_CHECK_NULL_RETURN_NULL(range);
  range->base_.Release = DefaultRelease;
  range->base_.Prepare = DefaultPrepare1In1Out;
  range->base_.Resize = DefaultResize;
  range->base_.Compute = RangeCompute;
  return (KernelBase *)range;
}

REG_KERNEL_CREATOR(PrimType_Range, kNumberTypeInt32, CreateRange)
REG_KERNEL_CREATOR(PrimType_Range, kNumberTypeFloat32, CreateRange)
REG_KERNEL_CREATOR(PrimType_Range, kNumberTypeFloat16, CreateRange)
