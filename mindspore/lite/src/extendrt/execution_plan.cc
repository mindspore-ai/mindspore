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
#include "litert/sub_graph_kernel.h"

namespace mindspore::infer {
ExecutionPlan::~ExecutionPlan() {
  if (input_isolate_map_ != nullptr) {
    delete input_isolate_map_;
    input_isolate_map_ = nullptr;
  }
  if (output_isolate_map_) {
    delete output_isolate_map_;
    output_isolate_map_ = nullptr;
  }

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

std::vector<abstract::Kernel *> ExecutionPlan::ToKernelList() {
  std::vector<abstract::Kernel *> kernels;
  for (auto flow : execution_flows_) {
    if (flow == nullptr) {
      MS_LOG(ERROR) << "ExecutionPlan::ToKernelList get nullptr execution flow.";
      return std::vector<abstract::Kernel *>{};
    }
    auto kernel = flow->ConstructFusionKernel();
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "ExecutionPlan::ToKernelList construct execution flow to Sub Graph Kernel failed.";
      return std::vector<abstract::Kernel *>{};
    }
    kernels.emplace_back(kernel);
  }
  return kernels;
}
}  // namespace mindspore::infer
