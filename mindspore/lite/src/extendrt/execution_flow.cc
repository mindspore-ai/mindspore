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
  kernels_.clear();
}

kernel::SubGraphKernel *ExecutionFlow::ConstructFusionKernel() {
  kernel::KernelExecUtil::FindAllInoutKernels(kernels_);
  if (kernels_.size() == 0) {
    MS_LOG(ERROR) << "CreateSubGraphKernel failed, kernels size is 0";
    return nullptr;
  }
  kernel::SubGraphType cur_sub_graph_type = this->GetSubGraphType(kernels_[0]);
  MS_LOG(INFO) << "cur_sub_graph_type: " << cur_sub_graph_type;
  // SCHEMA_VERSION::SCHEMA_CUR = 0
  // extendrt subgraph will be implemented later.
  auto subgraph_kernel =
    kernel::KernelExecUtil::CreateSubGraphKernel(kernels_, &inputs_, &outputs_, cur_sub_graph_type, *context_, 0);
  if (subgraph_kernel == nullptr) {
    MS_LOG(ERROR) << "CreateSubGraphKernel failed, cur_sub_graph_type: " << cur_sub_graph_type;
    return nullptr;
  }
  subgraph_kernel->SetTensors(this->tensors_);
  this->kernels_.clear();
  this->tensors_.clear();
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

mindspore::kernel::SubGraphType ExecutionFlow::GetSubGraphType(abstract::Kernel *kernel) {
  if (kernel == nullptr) {
    return kernel::kNotSubGraph;
  }

  auto provider = kernel->desc().provider;
  auto data_type = kernel->desc().data_type;
  auto arch = kernel->desc().arch;

  if (provider != kernel::kBuiltin) {
    // if support custom kernel, should return kernel::kCustomSubGraph
    MS_LOG(ERROR) << "not support non-build-in kernel";
    return kernel::kNotSubGraph;
  }

  // normal float compute sub graph
  switch (arch) {
    case kernel::kCPU: {
      if (data_type == kNumberTypeFloat16) {
        return kernel::kCpuFP16SubGraph;
      }
      if (data_type == kNumberTypeFloat32) {
        return kernel::kCpuFP32SubGraph;
      }
      return kernel::kCpuFP32SubGraph;
    }
    case kernel::kGPU: {
      if (data_type == kNumberTypeFloat16) {
        return kernel::kGpuFp16SubGraph;
      }
      if (data_type == kNumberTypeFloat32) {
        return kernel::kGpuFp32SubGraph;
      }
      return kernel::kGpuFp32SubGraph;
    }
    case kernel::kACL:
      return kernel::kAclSubGraph;
    default:
      return kernel::kNotSubGraph;
  }
}
}  // namespace mindspore::infer
