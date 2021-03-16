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
#include "nnacl/fp32/arithmetic_self_fp32.h"

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
                                      {mindspore::schema::PrimitiveType_Neg, ElementNegative},
                                      {mindspore::schema::PrimitiveType_Reciprocal, ElementReciprocal},
                                      {mindspore::schema::PrimitiveType_Erf, ElementErf}};
  for (size_t i = 0; i < sizeof(type_func_table) / sizeof(TYPE_FUNC_INFO); i++) {
    if (type_func_table[i].primitive_type_ == primitive_type) {
      return type_func_table[i].func_;
    }
  }
  return nullptr;
}

ArithmeticSelfBoolFunc ArithmeticSelfCPUKernel::GetArithmeticSelfBoolFun(int primitive_type) {
  if (primitive_type == mindspore::schema::PrimitiveType_LogicalNot) {
    return ElementLogicalNotBool;
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
  int ret = RET_ERROR;
  if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
    if (func_ == nullptr) {
      MS_LOG(ERROR) << "Run function is null! ";
      return RET_ERROR;
    }
    float *input_ptr = reinterpret_cast<float *>(in_tensors_.at(0)->data_c());
    float *output_ptr = reinterpret_cast<float *>(out_tensors_.at(0)->data_c());
    ret = func_(input_ptr + offset, output_ptr + offset, count);
  } else if (in_tensors_[0]->data_type() == kNumberTypeBool) {
    if (func_bool_ == nullptr) {
      MS_LOG(ERROR) << "Run function is null! ";
      return RET_ERROR;
    }
    bool *input_ptr = reinterpret_cast<bool *>(in_tensors_.at(0)->data_c());
    bool *output_ptr = reinterpret_cast<bool *>(out_tensors_.at(0)->data_c());
    ret = func_bool_(input_ptr + offset, output_ptr + offset, count);
  } else {
    MS_LOG(ERROR) << "Unsupported type: " << in_tensors_[0]->data_type() << ".";
    return RET_ERROR;
  }
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Abs, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Cos, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Log, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Square, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Sqrt, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Rsqrt, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Sin, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogicalNot, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_LogicalNot, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Floor, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Ceil, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Round, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Neg, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Reciprocal, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Erf, LiteKernelCreator<ArithmeticSelfCPUKernel>)
}  // namespace mindspore::kernel
