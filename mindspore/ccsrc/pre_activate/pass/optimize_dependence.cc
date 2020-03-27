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
const BaseRef OptimizeDependence::DefinePattern() const {
  VarPtr X = std::make_shared<Var>("X");
  MS_EXCEPTION_IF_NULL(X);
  VarPtr Y = std::make_shared<Var>("Y");
  MS_EXCEPTION_IF_NULL(Y);
  return VectorRef({prim::kPrimDepend, X, Y});
}

const AnfNodePtr OptimizeDependence::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  auto depend_cnode = node->cast<CNodePtr>();
  if (depend_cnode->inputs().size() < kDependInputNum) {
    return nullptr;
  }
  auto replacing_node = depend_cnode->input(kDependInputNum - 1);
  MS_EXCEPTION_IF_NULL(replacing_node);
  if (!replacing_node->isa<CNode>()) {
    return nullptr;
  }
  auto replacing_cnode = replacing_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(replacing_cnode);
  // Currently we only optimize transdata or cast nodes.
  string replacing_cnode_op_name = AnfAlgo::GetCNodeName(replacing_cnode);
  if (replacing_cnode_op_name != kTransDataOpName && replacing_cnode_op_name != prim::kPrimCast->name()) {
    return nullptr;
  }
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  // Check whether the replacing node has only one input and one output.
  if (replacing_cnode->inputs().size() != kSingleInputIndex + 1) {
    return nullptr;
  }
  if (manager->node_users().find(replacing_node) == manager->node_users().end()) {
    MS_LOG(EXCEPTION) << "The node should be used by at least another node input";
  }
  if (manager->node_users()[replacing_node].size() > 1) {
    return nullptr;
  }
  std::vector<AnfNodePtr> new_depend_inputs = {depend_cnode->input(kAnfPrimitiveIndex),
                                               depend_cnode->input(kRealInputIndexInDepend),
                                               replacing_cnode->input(kSingleInputIndex)};
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  CNodePtr new_depend;
  if (kernel_graph == nullptr) {
    new_depend = func_graph->NewCNode(new_depend_inputs);
  } else {
    new_depend = kernel_graph->NewCNode(depend_cnode);
    MS_EXCEPTION_IF_NULL(new_depend);
    new_depend->set_inputs(new_depend_inputs);
  }
  new_depend->set_abstract(node->abstract());
  return new_depend;
}
}  // namespace opt
}  // namespace mindspore
