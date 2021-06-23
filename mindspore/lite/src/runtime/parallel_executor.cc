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

#include <utility>
#include "src/runtime/parallel_executor.h"
#include "src/lite_kernel_util.h"

namespace mindspore::lite {
ParallelExecutor::~ParallelExecutor() { delete thread_pool_; }
int ParallelExecutor::Prepare(const std::vector<mindspore::kernel::LiteKernel *> &kernels,
                              const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                              const lite::InnerContext *ctx) {
  thread_pool_ = ActorThreadPool::CreateThreadPool(1, max_thread_num_, KThreadSpin);
  if (thread_pool_ == nullptr) {
    MS_LOG(ERROR) << "Memory error: fail to new ThreadPool";
    return RET_ERROR;
  }
  return RET_OK;
}

static int RunKernel(void *data, int index, float lhs_scale, float rhs_scale) {
  auto *executor = reinterpret_cast<ParallelExecutor *>(data);
  auto kernel = executor->GetReadyKernel(index);
  auto ret = kernel->Execute();
  executor->SetResult(index, ret);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "run kernel failed, name: " << kernel->name();
    return 0;
  }
  return 0;
}

int ParallelExecutor::Run(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                          const std::vector<kernel::LiteKernel *> &kernels, mindspore::Allocator *allocator,
                          const KernelCallBack &before, const KernelCallBack &after) {
  MS_ASSERT(allocator != nullptr);
  for (auto &inTensor : in_tensors) {
    if (inTensor == nullptr) {
      MS_LOG(ERROR) << "Graph input tensor is nullptr";
      return RET_ERROR;
    }
    if (inTensor->format() != mindspore::NHWC) {
      MS_LOG(ERROR) << "Model input tensor should be NHWC";
      return RET_ERROR;
    }
  }
  kernel::LiteKernelUtil::InitTensorInitRefCount(kernels);

  for (auto kernel : kernels) {
    if (kernel->in_kernels().empty()) {
      readyKernels.emplace_back(kernel);
      continue;
    }
    refCount[kernel] = kernel->in_kernels().size();
  }
  std::vector<kernel::LiteKernel *> newReadyKernels;
  while (!readyKernels.empty()) {
    results.resize(readyKernels.size(), RET_OK);
    if (thread_pool_->ParallelLaunch(RunKernel, this, readyKernels.size()) != 0) {
      MS_LOG(ERROR) << "ParallelLaunch failed ";
      return RET_ERROR;
    }

    if (std::find_if(results.begin(), results.end(), [](const int &ret) { return (ret != 0); }) != results.end()) {
      return RET_ERROR;
    }
    newReadyKernels.clear();
    for (auto completed : readyKernels) {
      for (auto out : completed->out_kernels()) {
        auto iter = refCount.find(out);
        if (iter == refCount.end()) {
          continue;
        }
        (iter->second)--;
        if (iter->second <= 0) {
          newReadyKernels.emplace_back(iter->first);
          refCount.erase(iter);
        }
      }
    }
    readyKernels.clear();
    readyKernels = std::move(newReadyKernels);
  }
  return RET_OK;
}
}  // namespace mindspore::lite
