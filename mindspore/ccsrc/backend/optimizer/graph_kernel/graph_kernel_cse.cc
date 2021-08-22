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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/kernel_info.h"

namespace mindspore {
namespace opt {
namespace {
bool IsCNodePrimitveEqual(const CNodePtr &main, const CNodePtr &node, const std::vector<PrimitivePtr> &black_list) {
  auto main_primitive = AnfAlgo::GetCNodePrimitive(main);
  auto node_primitive = AnfAlgo::GetCNodePrimitive(node);
  if (main_primitive != nullptr && node_primitive != nullptr) {
    // Some ops such as Reshape is not real op, cse these type will not get gain. And for ops fusion, keep these op
    // alone can prevent some redundant output case (input -> reshape -> output).
    if (main_primitive->name() != node_primitive->name() ||
        std::any_of(black_list.begin(), black_list.end(),
                    [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); })) {
      return false;
    }

    auto main_attrs = main_primitive->attrs();
    auto node_attrs = node_primitive->attrs();

    std::vector<std::string> exclude_attrs{"IsFeatureMapOutput", "IsFeatureMapInputList", "pri_format"};
    for (auto &attr : exclude_attrs) {
      main_attrs.erase(attr);
      node_attrs.erase(attr);
    }

    if (main_attrs.size() != node_attrs.size()) {
      return false;
    }

    auto all = std::all_of(main_attrs.begin(), main_attrs.end(),
                           [&node_attrs](const std::pair<std::string, ValuePtr> &item) -> bool {
                             if (item.second == nullptr) {
                               return false;
                             }
                             auto iter = node_attrs.find(item.first);
                             if (iter == node_attrs.end()) {
                               return false;
                             }
                             return *item.second == *iter->second;
                           });
    return all;
  }

  return *main->inputs()[0] == *node->inputs()[0];
}
}  // namespace

bool GraphKernelBackendCSE::CheckEqualKernelBuildInfo(const AnfNodePtr &main, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(main);
  MS_EXCEPTION_IF_NULL(node);

  if (!AnfAlgo::IsNodeInGraphKernel(main)) {
    return BackendCSE::CheckEqualKernelBuildInfo(main, node);
  }

  auto main_kernel_info = static_cast<device::KernelInfo *>(main->kernel_info());
  auto node_kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
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

    if (main_build_info->processor() != node_build_info->processor()) {
      return false;
    }

    return main_build_info->IsSimilarityKernelBuildInfo(*node_build_info);
  }
  return false;
}

bool GraphKernelBackendCSE::CheckEqualCnodeInputs(const AnfNodePtr &main, const AnfNodePtr &node) const {
  auto c_main = main->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(c_main);
  auto c_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(c_node);

  if (!AnfAlgo::IsNodeInGraphKernel(c_main)) {
    return BackendCSE::CheckEqualCnodeInputs(main, node);
  }

  const auto &inp1 = c_main->inputs();
  const auto &inp2 = c_node->inputs();
  if (inp1.size() != inp2.size()) {
    return false;
  }
  for (size_t j = 1; j < inp1.size(); j++) {
    auto inp1_j = inp1[j];
    auto inp2_j = inp2[j];
    MS_EXCEPTION_IF_NULL(inp1_j);
    MS_EXCEPTION_IF_NULL(inp2_j);
    if (!(*inp1_j == *inp2_j)) {
      return false;
    }
  }
  return IsCNodePrimitveEqual(c_main, c_node, black_list_);
}

bool GraphKernelCSE::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto graphkernel_backend_cse = std::make_shared<GraphKernelBackendCSE>(black_list_);
  return graphkernel_backend_cse->Cse(func_graph, func_graph->manager());
}
}  // namespace opt
}  // namespace mindspore
