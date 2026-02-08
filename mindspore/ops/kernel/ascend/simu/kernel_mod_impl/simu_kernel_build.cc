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

#include "kernel/ascend/simu/kernel_mod_impl/simu_kernel_build.h"
#include <string>
#include <memory>
#include <vector>
#include "kernel/ascend/simu/kernel_mod_impl/simu_kernel.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"

namespace mindspore {
namespace kernel {
KernelModPtr SimuOpBuild(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto prim = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(prim);
  MS_LOG(INFO) << "Build simu op [" << prim->name() << "]";

  auto kernel_mod_ptr = SimuKernelFactory::Get(prim->name());
  if (kernel_mod_ptr == nullptr) {
    MS_LOG(ERROR) << "Simu op can't find kernel[" << prim->name() << "]";
    return nullptr;
  }

  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);
  if (!kernel_mod_ptr->Init(prim, input_kernel_tensors, output_kernel_tensors)) {
    MS_LOG_WITH_NODE(EXCEPTION, anf_node)
      << "#dmsg#Kernel build failed:#dmsg#Initialize simu kernel op[" << anf_node->fullname_with_scope() << "] failed.";
  }
  return kernel_mod_ptr;
}
}  // namespace kernel
}  // namespace mindspore
