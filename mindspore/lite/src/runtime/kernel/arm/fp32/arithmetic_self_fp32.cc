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
#include "src/runtime/kernel/arm/fp32/arithmetic_self_fp32.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32/arithmetic_self.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
typedef struct {
  int primitive_type_;
  ArithmeticSelfFunc func_;
} TYPE_FUNC_INFO;
}  // namespace

ArithmeticSelfFunc ArithmeticSelfCPUKernel::GetArithmeticSelfFun(int primitive_type) {
  TYPE_FUNC_INFO type_func_table[] = {{mindspore::schema::PrimitiveType_Abs, ElementAbs},
                                      {mindspore::schema::PrimitiveType_Cos, ElementCos},
                                      {mindspore::schema::PrimitiveType_Log, ElementLog},
                                      {mindspore::schema::PrimitiveType_Square, ElementSquare},
                                      {mindspore::schema::PrimitiveType_Sqrt, ElementSqrt},
                                      {mindspore::schema::PrimitiveType_Rsqrt, ElementRsqrt},
                                      {mindspore::schema::PrimitiveType_Sin, ElementSin},
                                      {mindspore::schema::PrimitiveType_LogicalNot, ElementLogicalNot},
                                      {mindspore::schema::PrimitiveType_Floor, ElementFloor},
                                      {mindspore::schema::PrimitiveType_Ceil, ElementCeil},
                                      {mindspore::schema::PrimitiveType_Round, ElementRound},
                                      {mindspore::schema::PrimitiveType_Neg, ElementNegative}};
  for (size_t i = 0; i < sizeof(type_func_table); i++) {
    if (type_func_table[i].primitive_type_ == primitive_type) {
      return type_func_table[i].func_;
    }
  }
  return nullptr;
}

int ArithmeticSelfCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArithmeticSelfCPUKernel::ReSize() { return RET_OK; }

int ArithmeticSelfCPUKernel::DoExecute(int task_id) {
  int elements_num = in_tensors_.at(0)->ElementsNum();
  int stride = UP_DIV(elements_num, op_parameter_->thread_num_);
  int offset = task_id * stride;
  int count = MSMIN(stride, elements_num - offset);
  if (count <= 0) {
    return RET_OK;
  }
  if (func_ == nullptr) {
    MS_LOG(ERROR) << "Run function is null! ";
    return RET_ERROR;
  }
  float *input_ptr = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  float *output_ptr = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  auto ret = func_(input_ptr + offset, output_ptr + offset, count);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run failed, illegal input! ";
  }
  return ret;
}

int ArithmeticSelfRun(void *cdata, int task_id) {
  auto kernel = reinterpret_cast<ArithmeticSelfCPUKernel *>(cdata);
  auto ret = kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticSelfRuns error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int ArithmeticSelfCPUKernel::Run() {
  auto ret = ParallelLaunch(this->context_->thread_pool_, ArithmeticSelfRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticSelfRun error error_code[" << ret << "]";
  }
  return ret;
}

kernel::LiteKernel *CpuArithmeticSelfFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                       const std::vector<lite::Tensor *> &outputs,
                                                       OpParameter *parameter, const lite::InnerContext *ctx,
                                                       const kernel::KernelKey &desc,
                                                       const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) ArithmeticSelfCPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ArithmeticSelfCPUKernel fail!";
    free(parameter);
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Abs, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Cos, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Log, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Square, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Sqrt, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Rsqrt, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Sin, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogicalNot, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Floor, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Ceil, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Round, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Neg, CpuArithmeticSelfFp32KernelCreator)
}  // namespace mindspore::kernel
