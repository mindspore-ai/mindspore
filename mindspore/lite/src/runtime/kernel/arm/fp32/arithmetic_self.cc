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

#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32/arithmetic_self.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ArithmeticSelfCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int ArithmeticSelfCPUKernel::ReSize() {
  data_size_ = in_tensors_[0]->ElementsNum();
  thread_sz_count_ = MSMIN(thread_count_, data_size_);
  thread_sz_stride_ = UP_DIV(data_size_, thread_sz_count_);
  return RET_OK;
}

int ArithmeticSelfRuns(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto g_kernel = reinterpret_cast<ArithmeticSelfCPUKernel *>(cdata);
  auto ret = g_kernel->DoArithmeticSelf(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticSelfRuns error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int ArithmeticSelfCPUKernel::DoArithmeticSelf(int task_id) {
  int size = MSMIN(thread_sz_stride_, data_size_ - task_id * thread_sz_stride_);
  if (size <= 0) {
    return RET_OK;
  }
  int offset = task_id * thread_sz_stride_;
  if (arithmeticSelf_run_) {
    auto ret = arithmeticSelf_run_(in_ptr_ + offset, out_ptr_ + offset, size);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Run failed, illegal input! ";
      return ret;
    }
  } else {
    MS_LOG(ERROR) << "Run function is null! ";
    return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticSelfCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  auto input_tensor = in_tensors_.at(0);
  auto out_tensor = out_tensors_.at(0);
  in_ptr_ = reinterpret_cast<float *>(input_tensor->Data());
  out_ptr_ = reinterpret_cast<float *>(out_tensor->Data());
  ret = LiteBackendParallelLaunch(ArithmeticSelfRuns, this, thread_sz_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticSelfRun error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuArithmeticSelfFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                       const std::vector<lite::tensor::Tensor *> &outputs,
                                                       OpParameter *opParameter, const lite::Context *ctx,
                                                       const kernel::KernelKey &desc,
                                                       const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Creator failed, opParameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) ArithmeticSelfCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  MS_ASSERT(kernel != nullptr);
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Abs, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Cos, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Exp, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Log, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Square, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Sqrt, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Rsqrt, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Sin, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogicalNot, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Floor, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Ceil, CpuArithmeticSelfFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Round, CpuArithmeticSelfFp32KernelCreator)
}  // namespace mindspore::kernel
