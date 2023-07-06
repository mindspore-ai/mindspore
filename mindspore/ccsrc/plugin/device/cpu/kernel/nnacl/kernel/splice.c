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

#include "nnacl/kernel/splice.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/splice_parameter.h"
#include "nnacl/fp32/splice_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/splice_fp16.h"
#endif

int SpliceCompute(struct KernelBase *self) {
  TensorC *input = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  TensorC *output = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output);

  NNACL_CHECK_FALSE(input->shape_size_ != output->shape_size_, NNACL_SPLICE_SHAPE_INVALID);
  NNACL_CHECK_FALSE(input->shape_size_ != DIMENSION_3D, NNACL_SPLICE_SHAPE_INVALID);
  NNACL_CHECK_FALSE(output->shape_size_ != DIMENSION_3D, NNACL_SPLICE_SHAPE_INVALID);

  SpliceParameter *param = (SpliceParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);

  int src_row = input->shape_[Index1];
  int src_col = input->shape_[Index2];
  int dst_row = output->shape_[Index1];
  int dst_col = output->shape_[Index2];

  NNACL_CHECK_FALSE(src_col * param->context_dim_ != dst_col, NNACL_SPLICE_SHAPE_INVALID);
  NNACL_CHECK_FALSE(param->context_dim_ * dst_row != param->forward_indexes_dim_, NNACL_SPLICE_SHAPE_INVALID);

  for (int i = 0; i < param->forward_indexes_dim_; ++i) {
    if (param->forward_indexes_[i] >= src_row) {
      return NNACL_SPLICE_SHAPE_INVALID;
    }
  }

  void *input_data = input->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_data);
  void *output_data = output->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_data);

#ifdef ENABLE_FP16
  if (input->data_type_ == kNumberTypeFloat16) {
    SpliceFp16((float16_t *)input_data, src_row, src_col, param, (float16_t *)output_data, dst_row, dst_col);
    return NNACL_OK;
  }
#endif

  SpliceFp32((float *)input_data, src_row, src_col, param, (float *)output_data, dst_row, dst_col);
  return NNACL_OK;
}

KernelBase *CreateSplice(OpParameter *param, int data_type) {
  SpliceStruct *splice = (SpliceStruct *)malloc(sizeof(SpliceStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(splice);
  splice->base_.Release = DefaultRelease;
  splice->base_.Prepare = DefaultPrepare1In1Out;
  splice->base_.Resize = DefaultResize;
  splice->base_.Compute = SpliceCompute;
  return (KernelBase *)splice;
}

REG_KERNEL_CREATOR(PrimType_Splice, kNumberTypeFloat32, CreateSplice)
REG_KERNEL_CREATOR(PrimType_Splice, kNumberTypeFloat16, CreateSplice)
