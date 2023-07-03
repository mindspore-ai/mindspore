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

int TriuCompute(KernelBase *self) {
  TriuStruct *triu = (TriuStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(triu);

  void *src_data = self->in_[FIRST_INPUT]->data_;
  void *dst_data = self->out_[OUTPUT_INDEX]->data_;
  int type_size = DataTypeCSize(self->in_[FIRST_INPUT]->data_type_);
  NNACL_CHECK_ZERO_RETURN_ERR(type_size);

  int ret = TriuTrilGetKValue(self, &triu->k_);
  if (ret != NNACL_OK) {
    return ret;
  }

  int64_t mul, height, width;
  ret = TriuTrilGetCalculateNum(self, &mul, &height, &width);
  if (ret != NNACL_OK) {
    return ret;
  }

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
  triu->base_.Release = DefaultRelease;
  triu->base_.Prepare = DefaultPrepare1In1Out;
  triu->base_.Resize = DefaultResize;
  triu->base_.Compute = TriuCompute;
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
