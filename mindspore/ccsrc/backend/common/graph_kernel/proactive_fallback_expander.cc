/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "backend/common/graph_kernel/proactive_fallback_expander.h"

#include <unordered_set>
#include <vector>
#include <string>
#include <memory>

#include "ir/anf.h"
#include "utils/ms_context.h"
#include "kernel/graph_kernel_info.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "backend/common/expander/fallback/fallback_irbuilder.h"

namespace mindspore::graphkernel {
const std::unordered_set<std::string> &ProactiveFallbackExpander::GetFallbackOps() {
  static const std::unordered_set<std::string> fallback_ops_list_ = {"AddExt", "SubExt", "SumExt"};
  return fallback_ops_list_;
}

bool ProactiveFallbackExpander::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->get_return());
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto nodes = TopoSort(func_graph->get_return());
  const auto &need_fallback_ops = GetFallbackOps();
  for (const auto &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    const std::string &prim_name = GetCNodePrimitive(cnode)->name();
    if (need_fallback_ops.find(prim_name) == need_fallback_ops.end()) {
      continue;
    }
    MS_LOG(DEBUG) << "Start Fallback node: " << cnode->fullname_with_scope();
    auto func = [](const CNodePtr &cnode) -> bool {
      MS_EXCEPTION_IF_NULL(cnode);
      for (size_t i = 1; i < cnode->size(); i++) {
        const auto &input = cnode->input(i);
        if (!input->isa<ValueNode>()) {
          continue;
        }
        auto input_kernel_info = input->kernel_info_ptr();
        if (input_kernel_info == nullptr) {
          input_kernel_info = std::make_shared<device::KernelInfo>();
          input->set_kernel_info(input_kernel_info);
        }
        if (input_kernel_info->has_build_info()) {
          continue;
        }
        auto info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
        MS_EXCEPTION_IF_NULL(info_builder);
        auto vnode = input->cast<ValueNodePtr>();
        auto value = vnode->value();
        MS_EXCEPTION_IF_NULL(value);
        if (value->isa<tensor::Tensor>()) {
          auto tensor = value->cast<tensor::TensorPtr>();
          info_builder->SetOutputsFormat(std::vector<std::string>{kOpFormat_DEFAULT});
          info_builder->SetOutputsDeviceType(std::vector<TypeId>{tensor->Dtype()->type_id()});
        } else if (value->isa<Scalar>()) {
          auto scalar = value->cast<ScalarPtr>();
          info_builder->SetOutputsFormat(std::vector<std::string>{kOpFormat_DEFAULT});
          info_builder->SetOutputsDeviceType(std::vector<TypeId>{scalar->type()->type_id()});
        } else {
          return false;
        }
        AnfAlgo::SetSelectKernelBuildInfo(info_builder->Build(), input.get());
      }
      auto kernel_info = cnode->kernel_info_ptr();
      if (kernel_info == nullptr) {
        cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
      }
      auto kernel_info_setter = GraphKernelInfoManager::Instance().GetGraphKernelInfo(kAscendDevice);
      MS_EXCEPTION_IF_NULL(kernel_info_setter);
      kernel_info_setter->SetKernelInfo(cnode, KernelType::UNKNOWN_KERNEL_TYPE);
      return true;
    };
    expander::FallbackIRBuilder ib(prim_name, cnode->func_graph(), func);
    const auto *handle = expander::IRBuilderFactory::Instance().GetBuilder(prim_name);
    if (handle == nullptr) {
      MS_LOG(EXCEPTION) << "No fallback handle for node: " << cnode->fullname_with_scope();
      return false;
    }
    auto output = ib.Run(cnode, *handle);
    (void)mng->Replace(cnode, output);
  }
  return true;
}

}  // namespace mindspore::graphkernel
