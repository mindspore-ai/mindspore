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
#include "src/runtime/kernel/arm/fp32/arithmetic_compare_fp32.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32/arithmetic_compare.h"

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
namespace {
typedef struct {
  int primitive_type_;
  ArithmeticCompareFp32Func func_;
} TYPE_FUNC_INFO;
}  // namespace

ArithmeticCompareFp32Func ArithmeticCompareCPUKernel::GetArithmeticCompareFun(int primitive_type) {
  TYPE_FUNC_INFO type_func_table[] = {
    {PrimitiveType_Equal, ElementEqualFp32},     {PrimitiveType_NotEqual, ElementNotEqualFp32},
    {PrimitiveType_Less, ElementLessFp32},       {PrimitiveType_LessEqual, ElementLessEqualFp32},
    {PrimitiveType_Greater, ElementGreaterFp32}, {PrimitiveType_GreaterEqual, ElementGreaterEqualFp32}};
  for (size_t i = 0; i < sizeof(type_func_table); i++) {
    if (type_func_table[i].primitive_type_ == primitive_type) {
      return type_func_table[i].func_;
    }
  }
  return nullptr;
}

int ArithmeticCompareCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArithmeticCompareCPUKernel::ReSize() { return RET_OK; }

int ArithmeticCompareCPUKernel::DoExecute(int task_id) {
  if (in_tensors_.at(0)->shape() != in_tensors_.at(1)->shape()) {
    MS_LOG(ERROR) << "Compare op must inputs have the same shape, support broadcast later! ";
    return RET_ERROR;
  }
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
  // two inputs have the same shape, support broadcast later
  auto *input0_ptr = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto *input1_ptr = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  auto *output_ptr = reinterpret_cast<uint8_t *>(out_tensors_.at(0)->MutableData());
  auto ret = func_(input0_ptr + offset, input1_ptr + offset, output_ptr + offset, count);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run failed, illegal input! ";
  }
  return ret;
}

int ArithmeticCompareRun(void *cdata, int task_id) {
  auto kernel = reinterpret_cast<ArithmeticCompareCPUKernel *>(cdata);
  auto ret = kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticSelfRuns error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int ArithmeticCompareCPUKernel::Run() {
  auto ret = ParallelLaunch(this->context_->thread_pool_, ArithmeticCompareRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticSelfRun error error_code[" << ret << "]";
  }
  return ret;
}

kernel::LiteKernel *CpuArithmeticCompareFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                          const std::vector<lite::Tensor *> &outputs,
                                                          OpParameter *parameter, const lite::InnerContext *ctx,
                                                          const kernel::KernelKey &desc,
                                                          const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) ArithmeticCompareCPUKernel(parameter, inputs, outputs, ctx, primitive);
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Equal, CpuArithmeticCompareFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_NotEqual, CpuArithmeticCompareFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Less, CpuArithmeticCompareFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LessEqual, CpuArithmeticCompareFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Greater, CpuArithmeticCompareFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_GreaterEqual, CpuArithmeticCompareFp32KernelCreator)
}  // namespace mindspore::kernel
