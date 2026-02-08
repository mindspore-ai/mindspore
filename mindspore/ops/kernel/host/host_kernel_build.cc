/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "kernel/host/host_kernel_build.h"

#include <vector>
#include <string>
#include "kernel/host/host_kernel_mod.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "utils/log_adapter.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace kernel {
KernelModPtr HostOpBuild(const std::shared_ptr<AnfNode> &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto prim = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(prim);
  MS_LOG(INFO) << "Build host op [" << prim->name() << "]";

  auto kernel_mod_ptr = HostKernelFactory::Get(prim->name());
  if (kernel_mod_ptr == nullptr) {
    MS_LOG(ERROR) << "Host can't find Kernel[" << prim->name() << "]";
    return nullptr;
  }

  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);
  if (!std::static_pointer_cast<KernelMod>(kernel_mod_ptr)->Init(prim, input_kernel_tensors, output_kernel_tensors)) {
    MS_LOG_WITH_NODE(EXCEPTION, anf_node)
      << "#dmsg#Kernel build failed:#dmsg#Initialize host kernel op[" << anf_node->fullname_with_scope() << "] failed."
      << trace::DumpSourceLines(anf_node);
  }
  return kernel_mod_ptr;
}
}  // namespace kernel
}  // namespace mindspore
