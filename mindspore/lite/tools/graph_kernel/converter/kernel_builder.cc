/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/graph_kernel/converter/kernel_builder.h"

#include <string>
#include <memory>
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "tools/graph_kernel/converter/akg/ascend_kernel_builder.h"
#include "tools/graph_kernel/converter/akg/cpu_kernel_builder.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore::graphkernel {
AkgKernelBuilderPtr GetAkgBuilder(const std::string &target) {
  if (target == kCPUDevice) {
    return std::make_shared<CpuKernelBuilder>();
  }
  if (target == kAscendDevice) {
    return std::make_shared<AscendKernelBuilder>();
  }
  MS_LOG(EXCEPTION) << "GraphKernel does not support " << target << " akg builder.";
  return nullptr;
}

bool KernelBuilder::Run(const FuncGraphPtr &func_graph) {
  auto node_list = GkUtils::GetGraphKernelNodes(func_graph);
  if (node_list.empty()) {
    return false;
  }
  auto builder = GetAkgBuilder(Callback::Instance()->GetTargetFromContext());
  if (!builder->CompileJsonsInAnfnodes(node_list)) {
    MS_LOG(EXCEPTION) << "Graph kernel compile fail";
  }
  auto manager = Manage(func_graph, true);
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &node : node_list) {
    auto cnode = node->cast<CNodePtr>();
    auto custom_cnode = builder->CreateCustomOp(func_graph, cnode);
    if (custom_cnode == nullptr) {
      MS_LOG(EXCEPTION) << "Create custom op fail for " << cnode->fullname_with_scope();
    }
    manager->Replace(node, custom_cnode);
  }
  return true;
}
}  // namespace mindspore::graphkernel
