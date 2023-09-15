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

#define USE_DEPRECATED_API
#include "tools/graph_kernel/converter/kernel_builder.h"

#include <string>
#include <memory>
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "tools/graph_kernel/converter/akg/ascend_kernel_builder.h"
#include "tools/graph_kernel/converter/akg/cpu_kernel_builder.h"
#include "tools/graph_kernel/converter/akg/gpu_kernel_builder.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "mindspore/ccsrc/kernel/kernel_build_info.h"
#include "include/backend/kernel_info.h"

namespace mindspore::graphkernel {
AkgKernelBuilderPtr GetAkgBuilder(const std::string &target) {
  if (target == kCPUDevice) {
    return std::make_shared<CpuKernelBuilder>();
  }
  if (target == kGPUDevice) {
    return std::make_shared<GpuKernelBuilder>();
  }
  if (target == kAscendDevice) {
    return std::make_shared<AscendKernelBuilder>();
  }
  MS_LOG(EXCEPTION) << "GraphKernel does not support " << target << " akg builder.";
  return nullptr;
}

bool KernelBuilder::Run(const FuncGraphPtr &func_graph) {
  auto node_list = GkUtils::GetGraphKernelNodes(func_graph);
  auto device_type = Callback::Instance()->GetTargetFromContext();
  if (node_list.empty()) {
    MS_LOG(WARNING)
      << "No GraphKernel nodes found in the func_graph, possibly because the input model file does not have any "
         "operators that can be fused or the model has inputs with dynamic shapes.";
    return false;
  }
  auto builder = GetAkgBuilder(device_type);
  if (!builder->CompileJsonsInAnfnodes(node_list)) {
    MS_LOG(EXCEPTION) << "Graph kernel compile fail";
  }
  auto manager = Manage(func_graph, true);
  MS_EXCEPTION_IF_NULL(manager);
  ParameterPtr akg_node = nullptr;
  for (auto &node : node_list) {
    auto cnode = node->cast<CNodePtr>();
    auto custom_cnode = builder->CreateCustomOp(func_graph, cnode);
    if (custom_cnode == nullptr) {
      MS_LOG(EXCEPTION) << "Create custom op fail for " << cnode->fullname_with_scope();
    }
    if (!builder->GenerateAkgKernelNodes(func_graph, custom_cnode, cnode)) {
      MS_LOG(EXCEPTION) << "Copy kernel.o to tensor data fail for " << cnode->fullname_with_scope();
    }
    custom_cnode->set_kernel_info(node->kernel_info_ptr());
    manager->Replace(node, custom_cnode);
    if (akg_node != nullptr) {
      manager->AddEdge(custom_cnode, akg_node);
    }
  }
  return true;
}
}  // namespace mindspore::graphkernel
