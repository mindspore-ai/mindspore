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

#include "pre_activate/pass/optimize_dependence.h"
#include <memory>
#include <vector>
#include <string>
#include "pre_activate/common/helper.h"
#include "operator/ops.h"
#include "utils/utils.h"
#include "session/kernel_graph.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
constexpr auto kSingleInputIndex = 1;
namespace {
AnfNodePtr GetReplaceNode(const AnfNodePtr &node) {
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
  CheckCNodeInputSize(cnode, kSingleInputIndex + 1);
  return cnode->input(kSingleInputIndex);
}

AnfNodePtr ReplaceMakeTuple(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::GetCNodeName(cnode) != prim::kPrimMakeTuple->name()) {
    return nullptr;
  }
  std::vector<AnfNodePtr> new_make_tuple_inputs;
  bool need_update = false;
  for (const auto &input : cnode->inputs()) {
    AnfNodePtr replace_input = GetReplaceNode(input);
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

const AnfNodePtr OptimizeDependence::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  if (AnfAlgo::GetCNodeName(node) != prim::kPrimControlDepend->name() &&
      AnfAlgo::GetCNodeName(node) != prim::kPrimDepend->name()) {
    return nullptr;
  }
  size_t index = 0;
  auto depend_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(depend_cnode);
  std::vector<AnfNodePtr> new_depend_inputs = {depend_cnode->input(kAnfPrimitiveIndex)};
  if (AnfAlgo::GetCNodeName(node) == prim::kPrimDepend->name()) {
    index = 1;
    new_depend_inputs.push_back(depend_cnode->input(kRealInputIndexInDepend));
  }
  if (AnfAlgo::GetInputTensorNum(depend_cnode) < 2) {
    MS_LOG(EXCEPTION) << "The depend node input size is at less size 2,but got "
                      << AnfAlgo::GetInputTensorNum(depend_cnode) << depend_cnode->DebugString();
  }

  while (index < AnfAlgo::GetInputTensorNum(depend_cnode)) {
    auto replacing_node = AnfAlgo::GetInputNode(depend_cnode, index);
    ++index;
    MS_EXCEPTION_IF_NULL(replacing_node);
    if (!replacing_node->isa<CNode>()) {
      new_depend_inputs.push_back(replacing_node);
      continue;
    }
    auto replacing_cnode = replacing_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(replacing_cnode);
    // Deal with the make_tuple with TransData or Cast inputs.
    auto make_tuple_replace_node = ReplaceMakeTuple(func_graph, replacing_cnode);
    if (make_tuple_replace_node != nullptr) {
      new_depend_inputs.push_back(make_tuple_replace_node);
      continue;
    }
    AnfNodePtr replace_node = GetReplaceNode(replacing_cnode);
    if (replace_node == nullptr) {
      new_depend_inputs.push_back(replacing_node);
      MS_LOG(DEBUG) << "Can not find the TransData or Cast with single output node. Depend node: "
                    << node->DebugString();
      continue;
    }
    new_depend_inputs.push_back(replace_node);
  }
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  CNodePtr new_depend = nullptr;
  if (kernel_graph == nullptr) {
    new_depend = func_graph->NewCNode(new_depend_inputs);
    MS_EXCEPTION_IF_NULL(new_depend);
    new_depend->set_abstract(node->abstract());
    new_depend->set_scope(node->scope());
  } else {
    new_depend = kernel_graph->NewCNode(depend_cnode);
    MS_EXCEPTION_IF_NULL(new_depend);
    new_depend->set_inputs(new_depend_inputs);
  }
  return new_depend;
}
}  // namespace opt
}  // namespace mindspore
