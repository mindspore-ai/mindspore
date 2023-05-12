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

#include "src/extendrt/execution_flow.h"
#include <vector>
#include "src/litert/kernel_exec_util.h"
#include "src/executor/sub_graph_kernel.h"

namespace mindspore::infer {
ExecutionFlow::~ExecutionFlow() {
  for (auto kernel : kernels_) {
    delete kernel;
  }
}

abstract::Kernel *ExecutionFlow::ConstructFusionKernel() {
  kernel::KernelExecUtil::FindAllInoutKernels(kernels_);
  kernel::SubGraphType cur_sub_graph_type = kernel::kCpuFP32SubGraph;
  MS_LOG(INFO) << "cur_sub_graph_type: " << cur_sub_graph_type;
  // SCHEMA_VERSION::SCHEMA_CUR = 0
  auto subgraph_kernel =
    kernel::KernelExecUtil::CreateSubGraphKernel(kernels_, &inputs_, &outputs_, cur_sub_graph_type, *context_, 0);
  if (subgraph_kernel == nullptr) {
    MS_LOG(ERROR) << "CreateSubGraphKernel failed, cur_sub_graph_type: " << cur_sub_graph_type;
    return nullptr;
  }
  return subgraph_kernel;
}

std::string ExecutionFlow::Dump() const {
  std::ostringstream oss;
  oss << "inputs: [" << std::endl;
  for (const auto &input : inputs_) {
    oss << input->ToString() << std::endl;
  }
  oss << "]" << std::endl;
  oss << "outputs: [" << std::endl;
  for (const auto &output : outputs_) {
    oss << output->ToString() << std::endl;
  }
  oss << "]" << std::endl;
  oss << "kernels: [" << std::endl;
  for (const auto &kernel : kernels_) {
    oss << kernel->ToString() << std::endl << std::endl;
  }
  oss << "]" << std::endl;
  return oss.str();
}
}  // namespace mindspore::infer
