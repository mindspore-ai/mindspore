/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/common_subexpression_elimination.h"

#include <memory>
#include <vector>
#include <map>
#include <utility>
#include "runtime/device/kernel_info.h"
#include "mindspore/core/ops/core_ops.h"
#include "utils/ms_context.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
using KernelWithIndex = std::pair<AnfNodePtr, int64_t>;

bool CheckIgnoreCase(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (common::AnfAlgo::GetCNodeName(node) != kTransDataOpName) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  bool need_ignore = true;
  auto input_size = cnode->inputs().size() - 1;
  for (size_t k = 0; k < input_size; ++k) {
    auto input = common::AnfAlgo::VisitKernelWithReturnType(common::AnfAlgo::GetInputNode(cnode, k), 0).first;
    if (input != nullptr && input->isa<CNode>()) {
      need_ignore = false;
      break;
    }
  }
  return need_ignore;
}

void EliminateDuplicatedTupleGetItem(const FuncGraphPtr &graph, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(manager);

  // key: (getitem_input, getitem_index), value: getitem_list
  std::map<KernelWithIndex, std::vector<AnfNodePtr>> getitem_dup_map;
  const auto &node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (!node->isa<CNode>() || !IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto getitem_cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(getitem_cnode);
    KernelWithIndex input_with_index{getitem_cnode->input(kRealInputNodeIndexInTupleGetItem),
                                     GetGetitemIndex(getitem_cnode)};
    if (getitem_dup_map.count(input_with_index) == 0) {
      getitem_dup_map.emplace(input_with_index, std::vector<AnfNodePtr>{node});
    } else {
      getitem_dup_map[input_with_index].push_back(node);
    }
  }

  // remove duplicated
  for (auto &item : getitem_dup_map) {
    auto &getitem_list = item.second;
    if (getitem_list.size() > 1) {
      auto first_getitem = getitem_list[0];
      std::for_each(getitem_list.begin() + 1, getitem_list.end(), [first_getitem, manager](const AnfNodePtr &getitem) {
        (void)manager->Replace(getitem, first_getitem);
      });
    }
  }
}
}  // namespace

bool BackendCSE::CheckEqualKernelBuildInfo(const AnfNodePtr &main, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(main);
  MS_EXCEPTION_IF_NULL(node);
  if (main->isa<CNode>()) {
    auto main_name = common::AnfAlgo::GetCNodeName(main);
    if (main_name == prim::kPrimTensorMove->name() || main_name == prim::kPrimMemCpyAsync->name()) {
      return false;
    }
  }
  auto main_kernel_info = dynamic_cast<device::KernelInfo *>(main->kernel_info());
  auto node_kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  if (main_kernel_info == nullptr && node_kernel_info == nullptr) {
    return true;
  }
  if (main_kernel_info != nullptr && node_kernel_info != nullptr) {
    return *main_kernel_info == *node_kernel_info;
  }
  return false;
}

bool BackendCSE::CheckEqualCnodeInputs(const AnfNodePtr &main, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(main);
  MS_EXCEPTION_IF_NULL(node);
  auto c_main = main->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(c_main);
  auto c_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(c_node);
  const auto &inp1 = c_main->inputs();
  const auto &inp2 = c_node->inputs();
  if (inp1.size() != inp2.size()) {
    return false;
  }
  for (size_t j = 0; j < inp1.size(); j++) {
    auto inp1_j = GetReplicatedNode(inp1[j]);
    auto inp2_j = GetReplicatedNode(inp2[j]);
    MS_EXCEPTION_IF_NULL(inp1_j);
    MS_EXCEPTION_IF_NULL(inp2_j);
    if (!(*inp1_j == *inp2_j)) {
      return false;
    }
  }
  return true;
}

bool BackendCSE::CheckValueNode(const ValueNodePtr &main, const ValueNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(main);
  MS_EXCEPTION_IF_NULL(node);

  auto main_value = main->value();
  MS_EXCEPTION_IF_NULL(main_value);
  auto node_value = node->value();
  MS_EXCEPTION_IF_NULL(node_value);
  if (main_value->isa<Primitive>() && node_value->isa<Primitive>()) {
    return false;
  } else if (main_value->isa<tensor::Tensor>() && node_value->isa<tensor::Tensor>()) {
    return (AbsOf(main) == AbsOf(node)) && CheckEqualKernelBuildInfo(main, node);
  }
  return (AbsOf(main) == AbsOf(node)) && (*main_value == *node_value);
}

bool BackendCSE::CheckCNode(const CNodePtr &main, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(main);
  MS_EXCEPTION_IF_NULL(node);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->get_param<bool>(MS_CTX_ENABLE_LOOP_SINK) && CheckIgnoreCase(main)) {
    return false;
  }
  if (HasHiddenSideEffect(main) || HasHiddenSideEffect(node)) {
    return false;
  }
  if (!CheckEqualKernelBuildInfo(main, node)) {
    return false;
  }
  return CheckEqualCnodeInputs(main, node);
}

bool BackendCSE::CheckReplace(const AnfNodePtr &main, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(main);
  MS_EXCEPTION_IF_NULL(node);

  // attrs of nop node inserted by backend maybe omitted, so two nodes have same inputs will have different outputs
  auto main_abs = main->abstract();
  auto node_abs = node->abstract();
  if (main_abs != nullptr && node_abs != nullptr && !(*main_abs == *node_abs)) {
    return false;
  }

  if (main->isa<ValueNode>() && node->isa<ValueNode>()) {
    return CheckValueNode(main->cast<ValueNodePtr>(), node->cast<ValueNodePtr>());
  } else if (main->isa<CNode>() && node->isa<CNode>()) {
    return CheckCNode(main->cast<CNodePtr>(), node->cast<CNodePtr>());
  }
  return false;
}

bool BackendCSE::Cse(const FuncGraphPtr graph, const FuncGraphManagerPtr manager) {
  MS_EXCEPTION_IF_NULL(manager);
  Init();
  auto ret = BuildOrderGroupForOneGraph(graph, manager);
  if (ret) {
    DoReplace(manager);
    EliminateDuplicatedTupleGetItem(graph, manager);
  }
  return ret;
}

bool CommonSubexpressionElimination::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto backend_cse = std::make_shared<BackendCSE>();
  MS_EXCEPTION_IF_NULL(backend_cse);
  return backend_cse->Cse(func_graph, func_graph->manager());
}
}  // namespace opt
}  // namespace mindspore
