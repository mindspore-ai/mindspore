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

#include "nnacl/kernel/non_zero.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/tensor_c_utils.h"

int NonZeroCompute(KernelBase *self) {
  TensorC *input = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  TensorC *output = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output);

  NNACL_CHECK_FALSE(input->shape_size_ != DIMENSION_2D, NNACL_NON_ZERO_SHAPE_INVALID);

  bool *input_data = (bool *)input->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_data);
  int *output_data = (int *)output->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_data);

  int non_zero_nums = output->shape_[Index1];
  int non_zero_count = 0;

  int *coordiate_values = (int *)self->env_->Alloc(self->env_->allocator_, input->shape_size_ * sizeof(int));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(coordiate_values);

  for (int i = 0; i < GetElementNum(input); i += 1) {
    if (input_data[i]) {
      for (size_t j = 0; j < input->shape_size_; j++) {
        output_data[non_zero_count + (int)j * non_zero_nums] = coordiate_values[j];
      }
      non_zero_count++;
    }
    for (size_t idx = input->shape_size_; idx >= 1; --idx) {
      if (coordiate_values[idx - 1] != input->shape_[idx - 1] - 1) {
        coordiate_values[idx - 1] = coordiate_values[idx - 1] + 1;
        break;
      }
      coordiate_values[idx - 1] = 0;
    }
  }

  return NNACL_OK;
}

KernelBase *CreateNonZero(OpParameter *param, int data_type) {
  NonZeroStruct *non_zero = (NonZeroStruct *)malloc(sizeof(NonZeroStruct));
  NNACL_CHECK_NULL_RETURN_NULL(non_zero);
  non_zero->base_.Release = DefaultRelease;
  non_zero->base_.Prepare = DefaultPrepare2In1Out;
  non_zero->base_.Resize = DefaultResize;
  non_zero->base_.Compute = NonZeroCompute;
  return (KernelBase *)non_zero;
}

REG_KERNEL_CREATOR(PrimType_NonZero, kNumberTypeBool, CreateNonZero)
