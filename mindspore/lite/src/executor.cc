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

#include "src/executor.h"
#include <queue>
#include "include/errorcode.h"

namespace mindspore::lite {
int Executor::CheckInputs(const std::vector<Tensor *> &in_tensors) {
  for (auto &inTensor : in_tensors) {
    if (inTensor == nullptr) {
      MS_LOG(ERROR) << "Graph input tensor is nullptr";
      return RET_ERROR;
    }
    if (inTensor->data_type() != kObjectTypeTensorType && inTensor->data_c() == nullptr) {
      MS_LOG(ERROR) << "Graph input tensor data is nullptr " << in_tensors;
      return RET_ERROR;
    }
    auto shape = inTensor->shape();
    bool valid = all_of(shape.begin(), shape.end(), [](int i) { return i >= 0; });
    if (!valid) {
      MS_LOG(ERROR) << "The shape of input tensor contains negative dimension,"
                    << "check the model and assign the input shape with method Resize().";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int Executor::Run(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                  const std::vector<kernel::LiteKernel *> &kernels, mindspore::Allocator *allocator,
                  const KernelCallBack &before, const KernelCallBack &after) {
  MS_ASSERT(nullptr != allocator);
  auto ret = this->CheckInputs(in_tensors);
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "CheckInputs failed";
    return ret;
  }
  // clear ref_count
  for (auto *kernel : kernels) {
    for (auto *tensor : kernel->in_tensors()) {
      tensor->set_ref_count(0);
    }
  }
  std::queue<kernel::LiteKernel *> kernel_queue;
  for (auto kernel : kernels) {
    if (kernel->IsReady(kernel->in_tensors())) {
      kernel_queue.push(kernel);
    }
  }
  while (!kernel_queue.empty()) {
    auto cur_kernel = kernel_queue.front();
    kernel_queue.pop();
    MS_ASSERT(nullptr != cur_kernel);
    ret = cur_kernel->PreProcess();
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "PreProcess kernel failed, name: " << cur_kernel->name();
      return ret;
    }
    ret = cur_kernel->Run(before, after);
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "run kernel failed, name: " << cur_kernel->name();
      return ret;
    }
    ret = cur_kernel->PostProcess();
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "PostProcess kernel failed, name: " << cur_kernel->name();
      return ret;
    }
    for (auto &out_kernel : cur_kernel->out_kernels()) {
      if (out_kernel->IsReady(out_kernel->in_tensors())) {
        kernel_queue.push(out_kernel);
      }
    }
  }
  return RET_OK;
}

int CpuExecutor::Run(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                     const std::vector<kernel::LiteKernel *> &kernels, Allocator *allocator,
                     const KernelCallBack &before, const KernelCallBack &after) {
  MS_ASSERT(nullptr != allocator);
  //  not check input for merge. too hard
  if (kernels.front()->Type() != schema::PrimitiveType_Merge) {
    auto ret = this->CheckInputs(in_tensors);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "CheckInputs failed";
      return ret;
    }
  }
#ifdef SUPPORT_TRAIN
  for (auto out_tensor : out_tensors) {  // increase RefCount of output tensors, such that Run will not free them
    out_tensor->set_ref_count(out_tensor->ref_count() + 1);
  }
#endif
  for (auto *kernel : kernels) {
    MS_ASSERT(nullptr != kernel);
    auto ret = kernel->PreProcess();
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "PreProcess kernel failed, name: " << kernel->name();
      return ret;
    }
    ret = kernel->Run(before, after);
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "run kernel failed, name: " << kernel->name();
      return ret;
    }
    ret = kernel->PostProcess();
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "PostProcess kernel failed, name: " << kernel->name();
      return ret;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
