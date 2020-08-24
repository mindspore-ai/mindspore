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
using mindspore::predict::ThreadPool;
using mindspore::predict::TvmEnv;
#define MAX_THREAD_NUM 8
namespace mindspore::lite {
ParallelExecutor::~ParallelExecutor() {
  delete pool;
  pool = nullptr;
}
int ParallelExecutor::Prepare(std::vector<mindspore::kernel::LiteKernel *> &kernels) {
  pool = new ThreadPool();
  if (pool == nullptr) {
    MS_LOG(ERROR) << "Memory error: fail to new ThreadPool";
    return RET_ERROR;
  }
  pool->ConfigMaxThreadNum(MAX_THREAD_NUM);
  pool->ConfigThreadPool(NO_BIND, MAX_THREAD_NUM);
  return RET_OK;
}

static int RunKernel(int index, TvmEnv *env, void *data) {
  ParallelExecutor *executor = reinterpret_cast<ParallelExecutor *>(data);
  auto kernel = executor->GetReadyKernel(index);
  auto ret = kernel->Run();
  executor->SetResult(index, ret);
  if (0 != ret) {
    MS_LOG(ERROR) << "run kernel failed, name: " << kernel->name();
    return 0;
  }

  for (auto input_kernel : kernel->in_kernels()) {
    MS_ASSERT(input_kernel != nullptr);
    if (input_kernel->is_model_output()) {
      continue;
    }
    ret = input_kernel->DecOutTensorRefCount();
    if (0 != ret) {
      MS_LOG(WARNING) << "DecOutTensorRefCount for kernel" << kernel->name() << " failed";
    }
  }
  return 0;
}

int ParallelExecutor::Run(std::vector<tensor::Tensor *> &in_tensors, std::vector<tensor::Tensor *> &out_tensors,
                          std::vector<kernel::LiteKernel *> &kernels, Allocator *allocator,
                          const session::KernelCallBack &before, const session::KernelCallBack &after) {
  MS_ASSERT(nullptr != allocator);
  for (auto &inTensor : in_tensors) {
    if (inTensor == nullptr) {
      MS_LOG(ERROR) << "Graph input tensor is nullptr";
      return RET_ERROR;
    }
    if (inTensor->GetFormat() != schema::Format_NHWC) {
      MS_LOG(ERROR) << "Model input tensor should be NHWC";
      return RET_ERROR;
    }
  }
  kernel::LiteKernelUtil::InitTensorRefCount(kernels);

  for (auto kernel : kernels) {
    if (kernel->in_kernels().size() == 0) {
      readyKernels.emplace_back(kernel);
      continue;
    }
    refCount[kernel] = kernel->in_kernels().size();
  }
  std::vector<kernel::LiteKernel *> newReadyKernels;
  while (readyKernels.size() > 0) {
    results.resize(readyKernels.size(), RET_OK);
    pool->LaunchWork(RunKernel, this, readyKernels.size());

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

      for (auto input_kernel : completed->in_kernels()) {
        MS_ASSERT(input_kernel != nullptr);
        if (input_kernel->is_model_output()) {
          continue;
        }
        auto ret = input_kernel->DecOutTensorRefCount();
        if (0 != ret) {
          MS_LOG(WARNING) << "DecOutTensorRefCount for kernel" << completed->name() << " failed";
          return -1;
        }
      }
    }
    readyKernels.clear();
    readyKernels = std::move(newReadyKernels);
  }

  return RET_OK;
}

}  // namespace mindspore::lite
