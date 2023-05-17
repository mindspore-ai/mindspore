/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <vector>

#include "extendrt/execution_plan.h"

#include "litert/lite_kernel.h"
#include "litert/kernel_exec_util.h"
#include "executor/sub_graph_kernel.h"

namespace mindspore::infer {
ExecutionPlan::~ExecutionPlan() {
  FreeInputIsolateMap();
  FreeOutputIsolateMap();

  for (auto tensor : inputs_) {
    if (tensor != nullptr) {
      delete tensor;
    }
  }
  for (auto tensor : outputs_) {
    if (tensor != nullptr) {
      delete tensor;
    }
  }
}

std::vector<abstract::Kernel *> ExecutionPlan::ToKernelList() { return kernel_list_; }

bool ExecutionPlan::PrepareKernels() {
  for (auto kernel : kernels_) {
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "ExecutionPlan::PrepareKernels get nullptr execution flow.";
      return false;
    }
    auto subgraph_kernel = dynamic_cast<kernel::SubGraphKernel *>(kernel);
    for (auto &node : subgraph_kernel->nodes()) {
      auto ret = node->Prepare();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "ExecutionPlan::PrepareKernels node: " << node->name() << " prepare failed.";
        return false;
      }
    }
    auto ret = subgraph_kernel->Prepare();
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "ExecutionPlan::PrepareKernels prepare subgraph failed.";
      return false;
    }
    if (!MallocTensorData(subgraph_kernel)) {
      MS_LOG(ERROR) << "ExecutionPlan::PrepareKernels malloc memory for kernel: " << subgraph_kernel->name()
                    << " failed.";
      return false;
    }
    kernel_list_.emplace_back(kernel);
  }
  return true;
}

bool ExecutionPlan::MallocTensorData(abstract::Kernel *subgraph_kernel) {
  auto subgraph = dynamic_cast<kernel::SubGraphKernel *>(subgraph_kernel);
  if (subgraph->desc().arch != kernel::KERNEL_ARCH::kCPU) {
    MS_LOG(ERROR)
      << "ExecutionPlan::MallocTensorData subgraph target is not CPU, cannot malloc data with default allocator";
    return false;
  }
  auto kernel_list = subgraph->nodes();
  for (auto kernel_ : kernel_list) {
    for (auto tensor : kernel_->in_tensors()) {
      if (tensor->category() == lite::VAR) {
        auto ref_count = tensor->init_ref_count();
        tensor->set_init_ref_count(ref_count + 1);
      }
    }
  }
  return true;
}
}  // namespace mindspore::infer
