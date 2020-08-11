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

#include "src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "src/runtime/opencl/opencl_executor.h"
#include "src/runtime/opencl/opencl_runtime.h"

namespace mindspore::kernel {

SubGraphOpenCLKernel::~SubGraphOpenCLKernel() { UnInit(); }

int SubGraphOpenCLKernel::Init() {
  allocator_ = lite::opencl::OpenCLRuntime::GetInstance()->GetAllocator();
  MS_LOG(DEBUG) << "input num=" << inputs_.size() << ", output num=" << outputs_.size();
  for (const auto tensor : inputs_) {
    tensor->set_allocator(allocator_);
  }
  for (const auto tensor : outputs_) {
    tensor->set_allocator(allocator_);
  }
  // Map buffer for write, it is not necessary for fine-grained
  for (auto &tensor : inputs_) {
    void *data = tensor->Data();
    // It is required with coarse-grained SVM
    if (data != nullptr) {
      data = allocator_->MapBuffer(data, CL_MAP_WRITE, nullptr, true);
      tensor->SetData(data);
    } else {
      MS_LOG(ERROR) << "SubGraphOpenCLKernel input nullptr!";
    }
  }
  return 0;
}

int SubGraphOpenCLKernel::UnInit() {
  for (auto &tensor : outputs_) {
    allocator_->UnmapBuffer(tensor->Data());
  }
  for (const auto tensor : inputs_) {
    if (tensor != nullptr) {
      tensor->FreeData();
    }
  }
  for (const auto tensor : outputs_) {
    if (tensor != nullptr) {
      tensor->FreeData();
    }
  }
  return 0;
}

int SubGraphOpenCLKernel::InferShape() { return 0; }

int SubGraphOpenCLKernel::ReSize() { return 0; }

int SubGraphOpenCLKernel::Run() {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  for (auto &tensor : inputs_) {
    allocator_->UnmapBuffer(tensor->Data());
  }

  lite::opencl::OpenCLExecutor executor;
  executor.Run(inputs_, outputs_, nodes_, allocator_);
  ocl_runtime->SyncCommandQueue();
  for (auto &tensor : outputs_) {
    void *data = allocator_->MapBuffer(tensor->Data(), CL_MAP_READ, nullptr, true);
    tensor->SetData(data);
  }
  return 0;
}

}  // namespace mindspore::kernel
