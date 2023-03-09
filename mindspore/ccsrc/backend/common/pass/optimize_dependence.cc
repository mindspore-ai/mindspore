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

#include "backend/common/pass/optimize_dependence.h"
#include <memory>
#include <vector>
#include <string>
#include "include/backend/optimizer/helper.h"
#include "mindspore/core/ops/core_ops.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "utils/trace_base.h"
#include "include/backend/kernel_graph.h"

namespace mindspore {
namespace opt {
constexpr auto kSingleInputIndex = 1;
constexpr auto kIsolatedDependRealInputIndex = 0;
constexpr auto kIsolatedDependVirtualInputIndex = 1;
namespace {
CNodePtr CreateNewDependNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                             const std::vector<AnfNodePtr> &new_depend_inputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  if (kernel_graph == nullptr) {
    auto new_depend = func_graph->NewCNode(new_depend_inputs);
    MS_EXCEPTION_IF_NULL(new_depend);
    new_depend->set_abstract(cnode->abstract());
    new_depend->set_scope(cnode->scope());
    return new_depend;
  }
  auto new_depend = kernel_graph->NewCNode(cnode);
  MS_EXCEPTION_IF_NULL(new_depend);
  new_depend->set_inputs(new_depend_inputs);
  return new_depend;
}

CNodePtr CheckIsolatedVirtualNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::GetCNodeName(cnode) != prim::kPrimDepend->name() &&
      common::AnfAlgo::GetCNodeName(cnode) != prim::kPrimLoad->name()) {
    return nullptr;
  }
  auto virtual_input_op = common::AnfAlgo::GetInputNode(cnode, kIsolatedDependVirtualInputIndex);
  if (!HasAbstractMonad(virtual_input_op)) {
    return nullptr;
  }
  auto real_input_op = common::AnfAlgo::GetInputNode(cnode, kIsolatedDependRealInputIndex);
  MS_EXCEPTION_IF_NULL(real_input_op);
  if (!real_input_op->isa<CNode>()) {
    return nullptr;
  }
  auto real_input_cnode = real_input_op->cast<CNodePtr>();
  return real_input_cnode;
}

AnfNodePtr EliminateIsolatedVirtualNodeInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                             const CNodePtr &eliminate_node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(eliminate_node);
  auto replace_node = eliminate_node->input(kSingleInputIndex);
  std::vector<AnfNodePtr> new_depend_inputs = cnode->inputs();
  new_depend_inputs[kIsolatedDependRealInputIndex + 1] = replace_node;
  auto new_depend = CreateNewDependNode(func_graph, cnode, new_depend_inputs);
  (void)func_graph->manager()->Replace(cnode, new_depend);
  return new_depend;
}

AnfNodePtr GetReplaceNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto replace_cnode = cnode;
  // Process updatestate and depend as isolated node env.
  auto isolated_cnode = CheckIsolatedVirtualNode(replace_cnode);
  if (isolated_cnode != nullptr) {
    replace_cnode = isolated_cnode;
  }
  string op_name = common::AnfAlgo::GetCNodeName(replace_cnode);
  // Currently we only eliminate transdata or cast nodes.
  if (op_name != kTransDataOpName && op_name != prim::kPrimCast->name()) {
    return nullptr;
  }
  if (!IsNotRealUsedByOthers(func_graph, replace_cnode)) {
    return nullptr;
  }
  CheckCNodeInputSize(replace_cnode, kSingleInputIndex);
  if (isolated_cnode != nullptr) {
    auto new_depend_node = EliminateIsolatedVirtualNodeInput(func_graph, cnode, replace_cnode);
    return new_depend_node;
  }
  return cnode->input(kSingleInputIndex);
}

AnfNodePtr ReplaceMakeTuple(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::GetCNodeName(cnode) != prim::kPrimMakeTuple->name()) {
    return nullptr;
  }
  std::vector<AnfNodePtr> new_make_tuple_inputs = {common::AnfAlgo::GetCNodePrimitiveNode(cnode)};
  bool need_update = false;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t index = 0; index < input_num; ++index) {
    auto input = common::AnfAlgo::GetInputNode(cnode, index);
    AnfNodePtr replace_input = GetReplaceNode(func_graph, input);
    // If replace input is not null, it will be the input of the TransData or Cast.
    if (replace_input == nullptr) {
      new_make_tuple_inputs.push_back(input);
      continue;
    }
    new_make_tuple_inputs.push_back(replace_input);
    need_update = true;
  }
  if (need_update) {
    auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
    CNodePtr new_make_tuple = nullptr;
    if (kernel_graph == nullptr) {
      new_make_tuple = func_graph->NewCNode(new_make_tuple_inputs);
    } else {
      new_make_tuple = kernel_graph->NewCNode(cnode);
    }
    MS_EXCEPTION_IF_NULL(new_make_tuple);
    new_make_tuple->set_inputs(new_make_tuple_inputs);
    auto manager = func_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    manager->Replace(cnode, new_make_tuple);
    return new_make_tuple;
  }
  return nullptr;
}
}  // namespace

const BaseRef OptimizeDependence::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({X, Xs});
}

std::vector<size_t> SearchTransDataAndCast(const CNodePtr &cnode) {
  // Search Depend and UpdateState only.
  if (!cnode->IsApply(prim::kPrimDepend) && !cnode->IsApply(prim::kPrimUpdateState)) {
    return {};
  }
  // Find inputs which is Cast or TransData.
  std::vector<size_t> result;
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto &input = cnode->input(i);
    if (common::AnfAlgo::CheckPrimitiveType(input, prim::kPrimCast) ||
        common::AnfAlgo::CheckPrimitiveType(input, prim::kPrimTransData) ||
        common::AnfAlgo::CheckPrimitiveType(input, prim::kPrimMakeTuple)) {
      (void)result.emplace_back(i);
    }
  }
  return result;
}

const AnfNodePtr OptimizeDependence::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = dyn_cast<CNode>(node);
  if (cnode == nullptr) {
    return nullptr;
  }
  // Search inputs to be replaced.
  auto candidate_inputs = SearchTransDataAndCast(cnode);
  if (candidate_inputs.empty()) {
    return nullptr;
  }
  // Get new nodes which will act as new inputs of Depend or UpdateState.
  std::vector<AnfNodePtr> new_inputs = cnode->inputs();
  bool inputs_changed = false;
  for (auto index : candidate_inputs) {
    if (index >= new_inputs.size()) {
      MS_LOG(EXCEPTION) << "Index is out of the size of " << cnode->DebugString() << trace::DumpSourceLines(cnode);
    }
    auto replace_node = GetConvertNode(func_graph, cnode, index);
    if (replace_node != nullptr) {
      new_inputs[index] = replace_node;
      inputs_changed = true;
    }
  }
  if (!inputs_changed) {
    return nullptr;
  }
  // Create a new Depend node to replace the old one if inputs changed.
  auto new_depend = CreateNewDependNode(func_graph, cnode, new_inputs);
  (void)func_graph->manager()->Replace(cnode, new_depend);
  return nullptr;
}

const AnfNodePtr OptimizeDependence::GetConvertNode(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                    const size_t index) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto depend_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(depend_cnode);
  auto replacing_node = depend_cnode->input(index);
  MS_EXCEPTION_IF_NULL(replacing_node);
  if (!replacing_node->isa<CNode>()) {
    return nullptr;
  }
  auto replacing_cnode = replacing_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(replacing_cnode);
  // Deal with the make_tuple with TransData or Cast inputs.
  auto make_tuple_replace_node = ReplaceMakeTuple(graph, replacing_cnode);
  if (make_tuple_replace_node != nullptr) {
    return make_tuple_replace_node;
  }
  AnfNodePtr replace_node = GetReplaceNode(graph, replacing_cnode);
  return replace_node;
}
}  // namespace opt
}  // namespace mindspore
