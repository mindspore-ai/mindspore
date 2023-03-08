/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "backend/common/graph_kernel/graph_kernel_cse.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "include/backend/kernel_info.h"
#include "utils/ms_utils.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/graph_kernel/reshape_reduce_for_cse.h"

namespace mindspore::graphkernel {
namespace {
bool IsCNodePrimitveEqual(const CNodePtr &main, const CNodePtr &node, const std::vector<PrimitivePtr> &black_list) {
  auto main_primitive = common::AnfAlgo::GetCNodePrimitive(main);
  auto node_primitive = common::AnfAlgo::GetCNodePrimitive(node);
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
    std::vector<std::string> exclude_attrs{"IsFeatureMapOutput", "IsFeatureMapInputList", "pri_format",  "input_names",
                                           "output_names",       "in_strategy",           "out_strategy"};
    for (auto &attr : exclude_attrs) {
      auto main_attrs_iter = main_attrs.find(attr);
      if (main_attrs_iter != main_attrs.end()) {
        (void)main_attrs.erase(main_attrs_iter);
      }
      auto node_attrs_iter = node_attrs.find(attr);
      if (node_attrs_iter != node_attrs.end()) {
        (void)node_attrs.erase(node_attrs_iter);
      }
    }
    return common::IsAttrsEqual(main_attrs, node_attrs);
  }
  return *main->inputs()[0] == *node->inputs()[0];
}
}  // namespace

bool GraphKernelBackendCSE::CheckEqualKernelBuildInfo(const AnfNodePtr &main, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(main);
  MS_EXCEPTION_IF_NULL(node);

  if (!common::AnfAlgo::IsNodeInGraphKernel(main)) {
    return BackendCSE::CheckEqualKernelBuildInfo(main, node);
  }

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

bool CseForSingleGraph(const std::shared_ptr<GraphKernelBackendCSE> &graphkernel_backend_cse,
                       const std::shared_ptr<ReshapeReduceForCSE> &reshape_reduce, const FuncGraphPtr &fg) {
  bool continue_cse = true;
  bool has_change = false;
  while (continue_cse) {
    has_change = graphkernel_backend_cse->Cse(fg, fg->manager()) || has_change;
    continue_cse = reshape_reduce->Run(fg);
    has_change = has_change || continue_cse;
  }
  return has_change;
}

bool GraphKernelCSE::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto graphkernel_backend_cse = std::make_shared<GraphKernelBackendCSE>(black_list_);
  auto reshape_reduce = std::make_shared<ReshapeReduceForCSE>();
  auto changed = CseForSingleGraph(graphkernel_backend_cse, reshape_reduce, func_graph);
  auto nodes = TopoSort(func_graph->get_return());
  for (auto node : nodes) {
    auto graph_kernel_fg = GetCNodeFuncGraph(node);
    if (graph_kernel_fg != nullptr) {
      changed = CseForSingleGraph(graphkernel_backend_cse, reshape_reduce, graph_kernel_fg) || changed;
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel
