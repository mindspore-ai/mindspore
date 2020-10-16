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

#include "backend/optimizer/graph_kernel/graph_kernel_cse.h"

#include <memory>
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/kernel_info.h"

namespace mindspore {
namespace opt {

bool GraphKernelBackendCSE::CheckEqualKernelBuildInfo(const AnfNodePtr &main, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(main);
  MS_EXCEPTION_IF_NULL(node);
  auto main_kernel_info = dynamic_cast<device::KernelInfo *>(main->kernel_info());
  auto node_kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  if (main_kernel_info == nullptr && node_kernel_info == nullptr) {
    return true;
  }

  if (main_kernel_info != nullptr && node_kernel_info != nullptr) {
    auto main_build_info = main_kernel_info->GetMutableSelectKernelBuildInfo();
    auto node_build_info = node_kernel_info->GetMutableSelectKernelBuildInfo();
    if (main_build_info == nullptr && node_build_info == nullptr) {
      return true;
    }

    if (main_build_info == nullptr || node_build_info == nullptr) {
      return false;
    }

    if (main_build_info->fusion_type() != node_build_info->fusion_type() ||
        main_build_info->processor() != node_build_info->processor()) {
      return false;
    }

    return main_build_info->IsSimilarityKernelBuildInfo(*node_build_info);
  }
  return false;
}

bool GraphKernelCSE::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto graphkernel_backend_cse = std::make_shared<GraphKernelBackendCSE>();
  return graphkernel_backend_cse->Cse(func_graph, func_graph->manager());
}
}  // namespace opt
}  // namespace mindspore
