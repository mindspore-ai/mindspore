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

#include "src/extendrt/graph_compiler/single_graph_scheduler.h"

namespace mindspore {
namespace infer {
ExecutionFlowPtr SingleGraphScheduler::Schedule(const CompileResultPtr node_list) {
  // no need to infer, infer is invoked in converter and delived to here
  // select kernel
  // fp16/fp32 weight, transpose weight
  // append kernel with transpose
  // optimize transpose
  // infershape
  return nullptr;
}

KernelType SingleGraphScheduler::SelectKernel(const CompileNode *compile_node) { return kKernelTypeUnknown; }

bool SingleGraphScheduler::HandleWeightForKernel(const CompileNode *compile_node, const KernelType kernel_type) {
  return false;
}

bool SingleGraphScheduler::AppendKernelToPlan(kernel::KernelExec *kernel) { return false; }

bool SingleGraphScheduler::OptimizeTranspose(const std::vector<kernel::KernelExec *> &kernels) { return false; }

bool InferShape(const std::vector<kernel::KernelExec *> &kernels) { return false; }
}  // namespace infer
}  // namespace mindspore
