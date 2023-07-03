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

#include "nnacl/kernel/tril.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/common_func.h"
#include "nnacl/fp32/triu_tril_fp32.h"

int TrilCompute(KernelBase *self) {
  TrilStruct *tril = (TrilStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(tril);

  int ret = TriuTrilGetKValue(self, &tril->k_);
  if (ret != NNACL_OK) {
    return ret;
  }

  int64_t mul, height, width;
  ret = TriuTrilGetCalculateNum(self, &mul, &height, &width);
  if (ret != NNACL_OK) {
    return ret;
  }

  void *src_data = self->in_[FIRST_INPUT]->data_;
  void *dst_data = self->out_[OUTPUT_INDEX]->data_;
  int type_size = DataTypeCSize(self->in_[FIRST_INPUT]->data_type_);
  NNACL_CHECK_ZERO_RETURN_ERR(type_size);

  switch (type_size) {
    case sizeof(int64_t): {
      TrilByte8(src_data, dst_data, tril->k_, height, width, mul);
      break;
    }
    case sizeof(int32_t): {
      TrilByte4(src_data, dst_data, tril->k_, height, width, mul);
      break;
    }
    case sizeof(int16_t): {
      TrilByte2(src_data, dst_data, tril->k_, height, width, mul);
      break;
    }
    case sizeof(int8_t): {
      TrilByte1(src_data, dst_data, tril->k_, height, width, mul);
      break;
    }
    default:
      return NNACL_UNSUPPORTED_DATA_TYPE;
  }
  return NNACL_OK;
}

KernelBase *CreateTril(OpParameter *param, int data_type) {
  TrilStruct *tril = (TrilStruct *)malloc(sizeof(TrilStruct));
  NNACL_CHECK_NULL_RETURN_NULL(tril);
  tril->base_.Release = DefaultRelease;
  tril->base_.Prepare = DefaultPrepare1In1Out;
  tril->base_.Resize = DefaultResize;
  tril->base_.Compute = TrilCompute;
  return (KernelBase *)tril;
}

REG_KERNEL_CREATOR(PrimType_Tril, kNumberTypeDouble, CreateTril)
REG_KERNEL_CREATOR(PrimType_Tril, kNumberTypeFloat, CreateTril)
REG_KERNEL_CREATOR(PrimType_Tril, kNumberTypeFloat64, CreateTril)
REG_KERNEL_CREATOR(PrimType_Tril, kNumberTypeFloat32, CreateTril)
REG_KERNEL_CREATOR(PrimType_Tril, kNumberTypeFloat16, CreateTril)
REG_KERNEL_CREATOR(PrimType_Tril, kNumberTypeInt, CreateTril)
REG_KERNEL_CREATOR(PrimType_Tril, kNumberTypeInt64, CreateTril)
REG_KERNEL_CREATOR(PrimType_Tril, kNumberTypeInt32, CreateTril)
REG_KERNEL_CREATOR(PrimType_Tril, kNumberTypeInt16, CreateTril)
REG_KERNEL_CREATOR(PrimType_Tril, kNumberTypeInt8, CreateTril)
REG_KERNEL_CREATOR(PrimType_Tril, kNumberTypeUInt64, CreateTril)
REG_KERNEL_CREATOR(PrimType_Tril, kNumberTypeUInt32, CreateTril)
REG_KERNEL_CREATOR(PrimType_Tril, kNumberTypeUInt16, CreateTril)
REG_KERNEL_CREATOR(PrimType_Tril, kNumberTypeUInt8, CreateTril)
REG_KERNEL_CREATOR(PrimType_Tril, kNumberTypeBool, CreateTril)
