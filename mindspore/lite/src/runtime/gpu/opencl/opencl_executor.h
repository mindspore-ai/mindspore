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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_OPENCL_EXECUTOR_H_
#define MINDSPORE_LITE_SRC_RUNTIME_OPENCL_EXECUTOR_H_

#include <vector>
#include "src/runtime/gpu/opencl/opencl_runtime.h"
#include "src/runtime/allocator.h"
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "src/executor.h"
#include "include/lite_session.h"

namespace mindspore::lite::opencl {
class OpenCLExecutor : public Executor {
 public:
  OpenCLExecutor() : Executor() { allocator_ = ocl_runtime.GetInstance()->GetAllocator(); }

  int Prepare(const std::vector<kernel::LiteKernel *> &kernels) override { return RET_OK; }

  int Run(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
          const std::vector<kernel::LiteKernel *> &kernels, mindspore::Allocator *allocator = nullptr,
          const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr) override;
  int RunOrTune(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                const std::vector<kernel::LiteKernel *> &kernels, mindspore::Allocator *allocator = nullptr,
                const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr, bool is_tune = false);

 protected:
  InnerContext *context = nullptr;
  OpenCLAllocator *allocator_ = nullptr;
  OpenCLRuntimeWrapper ocl_runtime;
};
}  // namespace mindspore::lite::opencl
#endif
