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
#include "src/runtime/kernel/arm/fp16/arithmetic_self_fp16.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "src/kernel_registry.h"
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/fp16/arithmetic_self_fp16.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
typedef struct {
  int primitive_type_;
  ArithmeticSelfFp16Func func_;
} TYPE_FUNC_INFO;
}  // namespace

ArithmeticSelfFp16Func ArithmeticSelfFp16CPUKernel::GetArithmeticSelfFp16Fun(int primitive_type) {
  TYPE_FUNC_INFO type_func_table[] = {{mindspore::schema::PrimitiveType_Abs, ElementAbsFp16},
                                      {mindspore::schema::PrimitiveType_Cos, ElementCosFp16},
                                      {mindspore::schema::PrimitiveType_Log, ElementLogFp16},
                                      {mindspore::schema::PrimitiveType_Square, ElementSquareFp16},
                                      {mindspore::schema::PrimitiveType_Sqrt, ElementSqrtFp16},
                                      {mindspore::schema::PrimitiveType_Rsqrt, ElementRsqrtFp16},
                                      {mindspore::schema::PrimitiveType_Sin, ElementSinFp16},
                                      {mindspore::schema::PrimitiveType_LogicalNot, ElementLogicalNotFp16},
                                      {mindspore::schema::PrimitiveType_Floor, ElementFloorFp16},
                                      {mindspore::schema::PrimitiveType_Ceil, ElementCeilFp16},
                                      {mindspore::schema::PrimitiveType_Round, ElementRoundFp16},
                                      {mindspore::schema::PrimitiveType_Neg, ElementNegativeFp16},
                                      {mindspore::schema::PrimitiveType_Reciprocal, ElementReciprocalFp16}};
  for (size_t i = 0; i < sizeof(type_func_table) / sizeof(TYPE_FUNC_INFO); i++) {
    if (type_func_table[i].primitive_type_ == primitive_type) {
      return type_func_table[i].func_;
    }
  }
  return nullptr;
}

int ArithmeticSelfFp16CPUKernel::DoExecute(int task_id) {
  int elements_num = in_tensors_.at(0)->ElementsNum();
  int stride = UP_DIV(elements_num, op_parameter_->thread_num_);
  int offset = task_id * stride;
  int count = MSMIN(stride, elements_num - offset);
  if (count <= 0) {
    return RET_OK;
  }
  if (fp16_func_ == nullptr) {
    MS_LOG(ERROR) << "Run function is null! ";
    return RET_ERROR;
  }
  auto ret = fp16_func_(input_fp16_ptr_ + offset, output_fp16_ptr_ + offset, count);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run failed, illegal input! ";
  }
  return ret;
}

void ArithmeticSelfFp16CPUKernel::FreeInputAndOutput() {
  if (in_tensors_.at(0)->data_type() == kNumberTypeFloat32) {
    context_->allocator->Free(input_fp16_ptr_);
    input_fp16_ptr_ = nullptr;
  }
  if (out_tensors_.at(0)->data_type() == kNumberTypeFloat32) {
    context_->allocator->Free(output_fp16_ptr_);
    output_fp16_ptr_ = nullptr;
  }
}

int ArithmeticSelfFp16CPUKernel::Run() {
  auto input_tensor = in_tensors_.at(0);
  auto output_tensor = out_tensors_.at(0);
  input_fp16_ptr_ = ConvertInputFp32toFp16(input_tensor, context_);
  output_fp16_ptr_ = MallocOutputFp16(output_tensor, context_);
  if (input_fp16_ptr_ == nullptr || output_fp16_ptr_ == nullptr) {
    FreeInputAndOutput();
    MS_LOG(ERROR) << "input or output is nullptr";
    return RET_ERROR;
  }
  auto ret = ParallelLaunch(this->context_->thread_pool_, ArithmeticSelfRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticSelfRun error error_code[" << ret << "]";
  }
  if (out_tensors_.at(0)->data_type() == kNumberTypeFloat32) {
    Float16ToFloat32(output_fp16_ptr_, reinterpret_cast<float *>(output_tensor->MutableData()),
                     output_tensor->ElementsNum());
  }
  FreeInputAndOutput();
  return ret;
}

kernel::LiteKernel *CpuArithmeticSelfFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                       const std::vector<lite::Tensor *> &outputs,
                                                       OpParameter *parameter, const lite::InnerContext *ctx,
                                                       const kernel::KernelKey &desc,
                                                       const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) ArithmeticSelfFp16CPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ArithmeticSelfFp16CPUKernel fail!";
    if (parameter != nullptr) {
      free(parameter);
    }
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << parameter->name_
                  << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Abs, CpuArithmeticSelfFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Cos, CpuArithmeticSelfFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Log, CpuArithmeticSelfFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Square, CpuArithmeticSelfFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Sqrt, CpuArithmeticSelfFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Rsqrt, CpuArithmeticSelfFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Sin, CpuArithmeticSelfFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_LogicalNot, CpuArithmeticSelfFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Floor, CpuArithmeticSelfFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Ceil, CpuArithmeticSelfFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Round, CpuArithmeticSelfFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Neg, CpuArithmeticSelfFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Reciprocal, CpuArithmeticSelfFp16KernelCreator)
}  // namespace mindspore::kernel
