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

#include "backend/optimizer/pass/optimize_dependence.h"
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include "backend/optimizer/common/helper.h"
#include "base/core_ops.h"
#include "utils/utils.h"
#include "backend/session/kernel_graph.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
constexpr auto kSingleInputIndex = 1;
namespace {
AnfNodePtr GetReplaceNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  string op_name = AnfAlgo::GetCNodeName(cnode);
  // Currently we only eliminate transdata or cast nodes.
  if (op_name != kTransDataOpName && op_name != prim::kPrimCast->name()) {
    return nullptr;
  }
  if (!IsNotRealUsedByOthers(func_graph, cnode)) {
    return nullptr;
  }
  CheckCNodeInputSize(cnode, kSingleInputIndex);
  return cnode->input(kSingleInputIndex);
}

AnfNodePtr ReplaceMakeTuple(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::GetCNodeName(cnode) != prim::kPrimMakeTuple->name()) {
    return nullptr;
  }
  std::vector<AnfNodePtr> new_make_tuple_inputs = {AnfAlgo::GetCNodePrimitiveNode(cnode)};
  bool need_update = false;
  size_t input_num = AnfAlgo::GetInputTensorNum(cnode);
  for (size_t index = 0; index < input_num; ++index) {
    auto input = AnfAlgo::GetInputNode(cnode, index);
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

std::pair<AnfNodePtr, size_t> SearchTransDataAndCast(const AnfNodePtr &node, bool is_first_node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return std::pair<AnfNodePtr, size_t>(nullptr, 0);
  }
  // get real input of depend and update state.
  size_t replace_input_index = 0;
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimDepend)) {
    replace_input_index = is_first_node ? kDependAttachNodeIndex : kRealInputIndexInDepend;
  } else if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimUpdateState)) {
    replace_input_index = is_first_node ? kUpdateStateStateInput : kUpdateStateRealInput;
  } else {
    return std::pair<AnfNodePtr, size_t>(nullptr, 0);
  }
  // check whether real input is cast or trans data
  auto real_input = node->cast<CNodePtr>()->input(replace_input_index);
  if (AnfAlgo::CheckPrimitiveType(real_input, prim::kPrimCast) ||
      AnfAlgo::CheckPrimitiveType(real_input, prim::KPrimTransData) ||
      AnfAlgo::CheckPrimitiveType(real_input, prim::kPrimMakeTuple)) {
    return std::pair<AnfNodePtr, size_t>(node, replace_input_index);
  }
  return SearchTransDataAndCast(real_input, false);
}

const AnfNodePtr OptimizeDependence::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  // Get the cnode with repalce input index
  auto cnode_with_input_index = SearchTransDataAndCast(node, true);
  if (cnode_with_input_index.first == nullptr) {
    return nullptr;
  }
  size_t replace_index = cnode_with_input_index.second;
  auto depend_cnode = cnode_with_input_index.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(depend_cnode);
  // Get new node which will act as new input of depend or UpdateState.
  std::vector<AnfNodePtr> new_depend_inputs = depend_cnode->inputs();
  auto replace_node = GetConvertNode(func_graph, depend_cnode, replace_index);
  if (replace_node == nullptr) {
    return nullptr;
  }
  new_depend_inputs[replace_index] = replace_node;
  // Because depend's input has been changed, so a new depend(UpdateState) node will be created to replaced the old one.
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  CNodePtr new_depend = nullptr;
  if (kernel_graph == nullptr) {
    new_depend = func_graph->NewCNode(new_depend_inputs);
    MS_EXCEPTION_IF_NULL(new_depend);
    new_depend->set_abstract(depend_cnode->abstract());
    new_depend->set_scope(depend_cnode->scope());
  } else {
    new_depend = kernel_graph->NewCNode(depend_cnode);
    MS_EXCEPTION_IF_NULL(new_depend);
    new_depend->set_inputs(new_depend_inputs);
  }
  func_graph->manager()->Replace(depend_cnode, new_depend);
  return nullptr;
}

const AnfNodePtr OptimizeDependence::GetConvertNode(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                    const size_t index) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto depend_cnode = node->cast<CNodePtr>();
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
