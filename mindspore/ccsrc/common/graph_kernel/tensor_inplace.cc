/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "common/graph_kernel/tensor_inplace.h"

#include <utility>
#include <vector>
#include "common/graph_kernel/model/op_node.h"
#include "common/graph_kernel/model/op_register.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "common/graph_kernel/graph_kernel_helper.h"

namespace mindspore::graphkernel {
namespace {
// when A and B's relation is case1 or case2, return true; else return false
// case 1: B is A's only user
// ********
// * A    *
// * |    *
// * |    *
// * |    *
// * B    *
// ********
// case 2: B is one of A's users and A's other users depend on B
// *************
// * A         *
// * |  \  \   *
// * |   C ..D *
// * |   /  /  *
// * |  E  /   *
// * | / /     *
// * B         *
// *************
bool NoUsersAfterCurNode(const AnfNodePtr &input, const AnfNodePtr &cur_node, const FuncGraphManagerPtr &mng) {
  if (!input->isa<CNode>()) return false;
  mindspore::HashSet<AnfNodePtr> predecessors;
  std::function<void(AnfNodePtr)> dfs;
  dfs = [&predecessors, &dfs](const AnfNodePtr &node) {
    if (!node->isa<CNode>() || predecessors.count(node) > 0) return;
    (void)predecessors.insert(node);
    auto cnode = node->cast<CNodePtr>();
    auto inputs = cnode->inputs();
    for (size_t i = 1; i < inputs.size(); i++) {
      dfs(inputs[i]);
    }
  };
  dfs(cur_node);
  auto users = mng->node_users()[input];
  return std::all_of(users.begin(), users.end(),
                     [&predecessors](const std::pair<AnfNodePtr, int> &p) { return predecessors.count(p.first) > 0; });
}

// check node types of a graph kernel
bool CheckComputeType(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph->get_return());
  auto all_nodes = TopoSort(func_graph->get_return());
  for (auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimReturn)) continue;
    if (auto cnode = node->cast<CNodePtr>()) {
      auto node_prim = inner::OpRegistry::Instance().NewOp(GetValueNode<PrimitivePtr>(cnode->input(0))->name());
      if (node_prim == nullptr || node_prim->compute_type() != inner::PrimOp::ComputeType::ELEMWISE) {
        return false;
      }
    }
  }
  return true;
}

// collect output nodes of a subgraph
std::vector<AnfNodePtr> SubGraphOutputs(const FuncGraphPtr &sub_graph) {
  std::vector<AnfNodePtr> outs;
  auto output = sub_graph->output();
  if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
    auto output_cnode = output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(output_cnode);
    (void)outs.insert(outs.end(), output_cnode->inputs().begin() + 1, output_cnode->inputs().end());
  } else {
    (void)outs.emplace_back(output);
  }
  return outs;
}

// whether two nodes have same shape and type
bool IsSameShapeTypeFormat(const AnfNodePtr &x, const AnfNodePtr &y) {
  if (!x->isa<CNode>() || !y->isa<CNode>()) return false;
  if (common::AnfAlgo::GetOutputInferShape(x, 0) != common::AnfAlgo::GetOutputInferShape(y, 0)) return false;
  return common::AnfAlgo::GetOutputInferDataType(x, 0) == common::AnfAlgo::GetOutputInferDataType(y, 0);
}
}  // namespace

bool TensorInplace::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->get_return());
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto todos = TopoSort(func_graph->get_return());
  bool changed = false;
  for (auto &node : todos) {
    if (common::AnfAlgo::IsGraphKernel(node)) {
      auto sub_func_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
      MS_EXCEPTION_IF_NULL(sub_func_graph);
      if (!CheckComputeType(sub_func_graph)) continue;
      auto outs = SubGraphOutputs(sub_func_graph);
      if (outs.size() > 1) continue;
      auto cnode = node->cast<CNodePtr>();
      for (size_t i = 1; i < cnode->inputs().size(); i++) {
        if (NoUsersAfterCurNode(cnode->input(i), node, mng) && IsSameShapeTypeFormat(cnode->input(i), node)) {
          // input - output pair suitable for inplace assign is found
          changed = true;
          InplaceAssignerInfo new_op_info;  // output info
          new_op_info.op_node = outs[0]->cast<CNodePtr>();
          new_op_info.real_output_num = 1;
          new_op_info.inplace_to_origin_input = i - 1;
          // modify graph kernel's abstract, kernelBuildInfo and insert assign
          ProcessOriginCNode(cnode, {{new_op_info, cnode->input(i)}});
          // reconnet output's user to input
          ProcessOriginCNodeUser(func_graph, cnode, {{new_op_info, cnode->input(i)}}, mng);
          break;
        }
      }
    }
  }
  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}
}  // namespace mindspore::graphkernel
