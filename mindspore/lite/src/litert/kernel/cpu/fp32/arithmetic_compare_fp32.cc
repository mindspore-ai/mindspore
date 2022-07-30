/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/cpu/fp32/arithmetic_compare_fp32.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/fp32/arithmetic_compare_fp32.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Equal;
using mindspore::schema::PrimitiveType_Greater;
using mindspore::schema::PrimitiveType_GreaterEqual;
using mindspore::schema::PrimitiveType_Less;
using mindspore::schema::PrimitiveType_LessEqual;
using mindspore::schema::PrimitiveType_NotEqual;

namespace mindspore::kernel {
void ArithmeticCompareCPUKernel::InitRunFunction(int primitive_type) {
  ARITHMETIC_COMEPARE_FUNC_INFO_FP32 fun_table[] = {
    {PrimitiveType_Equal, ElementEqualFp32, ElementEqualInt32, ElementOptEqualFp32, ElementOptEqualInt32},
    {PrimitiveType_NotEqual, ElementNotEqualFp32, ElementNotEqualInt32, ElementOptNotEqualFp32,
     ElementOptNotEqualInt32},
    {PrimitiveType_Less, ElementLessFp32, ElementLessInt32, ElementOptLessFp32, ElementOptLessInt32},
    {PrimitiveType_LessEqual, ElementLessEqualFp32, ElementLessEqualInt32, ElementOptLessEqualFp32,
     ElementOptLessEqualInt32},
    {PrimitiveType_Greater, ElementGreaterFp32, ElementGreaterInt32, ElementOptGreaterFp32, ElementOptGreaterInt32},
    {PrimitiveType_GreaterEqual, ElementGreaterEqualFp32, ElementGreaterEqualInt32, ElementOptGreaterEqualFp32,
     ElementOptGreaterEqualInt32}};
  size_t length = sizeof(fun_table) / sizeof(ARITHMETIC_COMEPARE_FUNC_INFO_FP32);
  for (size_t i = 0; i < length; i++) {
    if (fun_table[i].primitive_type_ == primitive_type) {
      func_fp32_ = fun_table[i].func_;
      func_int32_ = fun_table[i].int_func_;
      opt_func_fp32_ = fun_table[i].opt_func_;
      opt_func_int32_ = fun_table[i].opt_int_func_;
      return;
    }
  }
}

int ArithmeticCompareCPUKernel::DoExecute(const void *input0, const void *input1, void *output, int64_t size) {
  int ret = RET_OK;
  if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
    if (scalar_opt_) {
      CHECK_NULL_RETURN(opt_func_fp32_);
      ret = opt_func_fp32_(reinterpret_cast<const float *>(input0), reinterpret_cast<const float *>(input1),
                           reinterpret_cast<uint8_t *>(output), size, param_);
    } else {
      CHECK_NULL_RETURN(func_fp32_);
      ret = func_fp32_(reinterpret_cast<const float *>(input0), reinterpret_cast<const float *>(input1),
                       reinterpret_cast<uint8_t *>(output), size);
    }
  } else if (in_tensors_[0]->data_type() == kNumberTypeInt || in_tensors_[0]->data_type() == kNumberTypeInt32) {
    if (scalar_opt_) {
      CHECK_NULL_RETURN(opt_func_int32_);
      ret = opt_func_int32_(reinterpret_cast<const int *>(input0), reinterpret_cast<const int *>(input1),
                            reinterpret_cast<uint8_t *>(output), size, param_);
    } else {
      CHECK_NULL_RETURN(func_int32_);
      ret = func_int32_(reinterpret_cast<const int *>(input0), reinterpret_cast<const int *>(input1),
                        reinterpret_cast<uint8_t *>(output), size);
    }
  } else {
    MS_LOG(ERROR) << "Error Operator type " << kNumberTypeInt32;
    return RET_ERROR;
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Equal, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Equal, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_NotEqual, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_NotEqual, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Less, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Less, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LessEqual, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_LessEqual, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Greater, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Greater, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_GreaterEqual, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_GreaterEqual, LiteKernelCreator<ArithmeticCompareCPUKernel>)
}  // namespace mindspore::kernel
