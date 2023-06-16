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

#include "nnacl/kernel/triu.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/common_func.h"
#include "nnacl/fp32/triu_tril_fp32.h"

int triu_compute(KernelBase *self) {
  TriuStruct *triu = (TriuStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(triu);

  int ret = TriuTrilGetKValue(self, &triu->k_);
  if (ret != NNACL_OK) {
    return ret;
  }

  TensorC *input_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  for (size_t i = 0; i < input_tensor->shape_size_; i++) {
    if (input_tensor->shape_[i] <= 0) {
      return NNACL_INPUT_TENSOR_ERROR;
    }
  }

  int input_hw_dims = Num2;
  NNACL_CHECK_FALSE(input_tensor->shape_size_ < DIMENSION_2D, NNACL_TRIU_INPUT_DIMS_INVALID);

  int64_t mul = 1;
  for (size_t i = 0; i < input_tensor->shape_size_ - input_hw_dims; i++) {
    mul *= input_tensor->shape_[i];
  }
  int64_t height = input_tensor->shape_[input_tensor->shape_size_ - Num2];
  int64_t width = input_tensor->shape_[input_tensor->shape_size_ - Num1];

  void *src_data = input_tensor->data_;
  void *dst_data = self->out_[OUTPUT_INDEX]->data_;
  int type_size = DataTypeCSize(input_tensor->data_type_);
  NNACL_CHECK_ZERO_RETURN_ERR(type_size);

  switch (type_size) {
    case sizeof(int64_t): {
      TriuByte8(src_data, dst_data, triu->k_, height, width, mul);
      break;
    }
    case sizeof(int32_t): {
      TriuByte4(src_data, dst_data, triu->k_, height, width, mul);
      break;
    }
    case sizeof(int16_t): {
      TriuByte2(src_data, dst_data, triu->k_, height, width, mul);
      break;
    }
    case sizeof(int8_t): {
      TriuByte1(src_data, dst_data, triu->k_, height, width, mul);
      break;
    }
    default:
      return NNACL_UNSUPPORTED_DATA_TYPE;
  }
  return NNACL_OK;
}

KernelBase *CreateTriu(OpParameter *param, int data_type) {
  TriuStruct *triu = (TriuStruct *)malloc(sizeof(TriuStruct));
  NNACL_CHECK_NULL_RETURN_NULL(triu);
  triu->base_.release = default_release;
  triu->base_.prepare = default_prepare_1in_1out;
  triu->base_.resize = default_resize;
  triu->base_.compute = triu_compute;
  return (KernelBase *)triu;
}

REG_KERNEL_CREATOR(PrimType_Triu, kNumberTypeDouble, CreateTriu)
REG_KERNEL_CREATOR(PrimType_Triu, kNumberTypeFloat, CreateTriu)
REG_KERNEL_CREATOR(PrimType_Triu, kNumberTypeFloat64, CreateTriu)
REG_KERNEL_CREATOR(PrimType_Triu, kNumberTypeFloat32, CreateTriu)
REG_KERNEL_CREATOR(PrimType_Triu, kNumberTypeFloat16, CreateTriu)
REG_KERNEL_CREATOR(PrimType_Triu, kNumberTypeInt, CreateTriu)
REG_KERNEL_CREATOR(PrimType_Triu, kNumberTypeInt64, CreateTriu)
REG_KERNEL_CREATOR(PrimType_Triu, kNumberTypeInt32, CreateTriu)
REG_KERNEL_CREATOR(PrimType_Triu, kNumberTypeInt16, CreateTriu)
REG_KERNEL_CREATOR(PrimType_Triu, kNumberTypeInt8, CreateTriu)
REG_KERNEL_CREATOR(PrimType_Triu, kNumberTypeUInt64, CreateTriu)
REG_KERNEL_CREATOR(PrimType_Triu, kNumberTypeUInt32, CreateTriu)
REG_KERNEL_CREATOR(PrimType_Triu, kNumberTypeUInt16, CreateTriu)
REG_KERNEL_CREATOR(PrimType_Triu, kNumberTypeUInt8, CreateTriu)
REG_KERNEL_CREATOR(PrimType_Triu, kNumberTypeBool, CreateTriu)
