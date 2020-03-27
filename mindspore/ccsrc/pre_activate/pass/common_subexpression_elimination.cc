/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "pre_activate/pass/common_subexpression_elimination.h"
#include <memory>
#include "device/kernel_info.h"

namespace mindspore {
namespace opt {
namespace {
bool CheckEqualKernelBuildInfo(const AnfNodePtr &main, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(main);
  MS_EXCEPTION_IF_NULL(node);
  auto main_kernel_info = main->kernel_info();
  auto node_kernel_info = node->kernel_info();
  if (main_kernel_info == nullptr && node_kernel_info == nullptr) {
    return true;
  }
  if (main_kernel_info != nullptr && node_kernel_info != nullptr) {
    return *main_kernel_info == *node_kernel_info;
  }
  return false;
}
}  // namespace

bool BackendCSE::CheckReplace(const AnfNodePtr &main, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(main);
  MS_EXCEPTION_IF_NULL(node);

  bool replace = false;
  if (main->isa<ValueNode>() && node->isa<ValueNode>()) {
    auto main_value = GetValueNode(main);
    auto node_value = GetValueNode(node);
    if (main_value->isa<Primitive>() && node_value->isa<Primitive>()) {
      replace = false;
    } else {
      replace = (AbsOf(main) == AbsOf(node)) && (*main_value == *node_value);
    }
  } else if (main->isa<CNode>() && node->isa<CNode>()) {
    if (!CheckEqualKernelBuildInfo(main, node)) {
      replace = false;
    } else {
      auto c_main = main->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(c_main);
      auto c_node = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(c_node);
      const auto &inp1 = c_main->inputs();
      const auto &inp2 = c_node->inputs();
      if (inp1.size() == inp2.size()) {
        bool appsame = true;
        for (size_t j = 0; j < inp1.size(); j++) {
          MS_EXCEPTION_IF_NULL(inp1[j]);
          MS_EXCEPTION_IF_NULL(inp2[j]);
          if (!(*inp1[j] == *inp2[j])) {
            appsame = false;
            break;
          }
        }
        replace = appsame;
      }
    }
  }
  return replace;
}

bool CommonSubexpressionElimination::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto backend_cse = std::make_shared<BackendCSE>();
  return backend_cse->Cse(func_graph, func_graph->manager());
}
}  // namespace opt
}  // namespace mindspore
