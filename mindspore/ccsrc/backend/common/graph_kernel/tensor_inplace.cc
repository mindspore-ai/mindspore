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

#include "backend/common/graph_kernel/tensor_inplace.h"

#include <algorithm>
#include <utility>
#include <vector>
#include "backend/common/graph_kernel/model/op_node.h"
#include "backend/common/graph_kernel/model/op_register.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"

namespace mindspore::graphkernel {
namespace {
mindspore::HashSet<AnfNodePtr> GetPredecessors(const AnfNodePtr &cur_node) {
  mindspore::HashSet<AnfNodePtr> predecessors;
  std::function<void(AnfNodePtr)> dfs;
  dfs = [&predecessors, &dfs](const AnfNodePtr &node) {
    if (!node->isa<CNode>() || predecessors.count(node) > 0) {
      return;
    }
    (void)predecessors.insert(node);
    auto cnode = node->cast<CNodePtr>();
    auto inputs = cnode->inputs();
    for (size_t i = 1; i < inputs.size(); i++) {
      dfs(inputs[i]);
    }
  };
  dfs(cur_node);
  return predecessors;
}

// check if all nodes are elemwise/broadcast/virtual nodes
bool ElemwiseBroadcastChecker(const AnfNodePtr &node) {
  if (auto cnode = node->cast<CNodePtr>()) {
    // virtual node
    if (IsPrimitiveCNode(node, prim::kPrimReturn) || IsPrimitiveCNode(node, prim::kPrimMakeTuple) ||
        IsPrimitiveCNode(node, prim::kPrimReshape)) {
      return true;
    }
    // OneHot is treated as a kind of broadcast node
    if (IsPrimitiveCNode(node, prim::kPrimOneHot)) {
      return true;
    }
    // elemwise/broadcast
    auto node_prim = inner::OpRegistry::Instance().NewOp(GetValueNode<PrimitivePtr>(cnode->input(0))->name());
    return node_prim != nullptr && (node_prim->compute_type() == inner::PrimOp::ComputeType::ELEMWISE ||
                                    node_prim->compute_type() == inner::PrimOp::ComputeType::BROADCAST);
  }
  return true;
}

// check if it's a reduce graph kernel without multi-filters;
// case 1: a reduce kernel without multi-filters
// **********
// * A      *
// * |      *
// * B      *
// * |      *
// * Reduce *
// * |(out1)*
// **********
// case 2: a reduce kernel with multi-filters
// ******************
// * A              *
// * |   \          *
// * |     B        *
// * Reduce  \(out2)*
// * |(out1)        *
// ******************
bool ReduceChecker(const std::vector<AnfNodePtr> &nodes, const FuncGraphPtr &func_graph) {
  auto reduce_node = std::find_if(nodes.begin(), nodes.end(), [](const AnfNodePtr &node) {
    if (auto cnode = node->cast<CNodePtr>()) {
      auto node_prim = inner::OpRegistry::Instance().NewOp(GetValueNode<PrimitivePtr>(cnode->input(0))->name());
      return node_prim != nullptr && node_prim->compute_type() == inner::PrimOp::ComputeType::REDUCE;
    }
    return false;
  });
  if (reduce_node != nodes.end()) {
    auto mng = func_graph->manager();
    if (mng == nullptr) {
      mng = Manage(func_graph, false);
      func_graph->set_manager(mng);
    }
    auto &users = mng->node_users()[*reduce_node];
    auto predecessors = GetPredecessors(*reduce_node);
    // atomic add case
    auto assign_node = std::find_if(users.begin(), users.end(), [](const std::pair<AnfNodePtr, int> &p) {
      return IsPrimitiveCNode(p.first, prim::kPrimAssign);
    });
    if (assign_node != users.end()) {
      (void)predecessors.insert(assign_node->first);
    }
    for (const auto &n : predecessors) {
      auto &n_users = mng->node_users()[n];
      for (const auto &n_user : n_users) {
        if (!IsPrimitiveCNode(n_user.first, prim::kPrimReturn) &&
            !IsPrimitiveCNode(n_user.first, prim::kPrimMakeTuple) && predecessors.count(n_user.first) == 0) {
          return false;
        }
      }
    }
    return true;
  }
  return false;
}

// supported kernel: 1. elemwise/broadcast kernel 2.reduce kernel without multi-filters
bool CheckComputeType(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph->get_return());
  auto all_nodes = TopoSort(func_graph->get_return());
  // graph kernel of type elemwise or broadcast can pass check
  if (std::all_of(all_nodes.begin(), all_nodes.end(), ElemwiseBroadcastChecker)) {
    return true;
  }
  // graph kernel of reduce without multi-filters can pass check
  if (ReduceChecker(all_nodes, func_graph)) {
    return true;
  }
  return false;
}

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
// ***********
// * A       *
// * | \\    *
// * |  C..D *
// * |  / /  *
// * | E /   *
// * |//     *
// * B       *
// ***********
bool NoUsersAfterCurNode(const AnfNodePtr &input, const AnfNodePtr &cur_node, const FuncGraphManagerPtr &mng) {
  if (!input->isa<CNode>()) {
    return false;
  }
  mindspore::HashSet<AnfNodePtr> predecessors = GetPredecessors(cur_node);
  auto &users = mng->node_users()[input];
  return std::all_of(users.begin(), users.end(),
                     [&predecessors](const std::pair<AnfNodePtr, int> &p) { return predecessors.count(p.first) > 0; });
}

// input-output pair suitable for inplace assign should satisfy:
// 1. output is an elemwise op
// 2. in graph kernel, input and all input's users depend on output
mindspore::HashMap<size_t, std::vector<std::pair<AnfNodePtr, size_t>>> FindInputOutputPairs(
  const FuncGraphPtr &sub_graph) {
  auto isElemwise = [](const AnfNodePtr &node) {
    if (auto cnode = node->cast<CNodePtr>()) {
      auto node_prim = inner::OpRegistry::Instance().NewOp(GetValueNode<PrimitivePtr>(cnode->input(0))->name());
      if (node_prim != nullptr && node_prim->compute_type() == inner::PrimOp::ComputeType::ELEMWISE) {
        return true;
      }
    }
    return false;
  };

  std::vector<std::pair<AnfNodePtr, size_t>> outs;
  auto output = sub_graph->output();
  if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
    auto output_cnode = output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(output_cnode);
    for (size_t i = 1; i < output_cnode->inputs().size(); i++) {
      if (isElemwise(output_cnode->input(i))) {
        (void)outs.emplace_back(std::make_pair(output_cnode->input(i), i - 1));
      }
    }
  } else {
    if (isElemwise(output)) {
      (void)outs.emplace_back(std::make_pair(output, 0));
    }
  }

  std::vector<mindspore::HashSet<AnfNodePtr>> predecessors;
  (void)std::transform(outs.cbegin(), outs.cend(), std::back_inserter(predecessors),
                       [](const std::pair<AnfNodePtr, size_t> &out) { return GetPredecessors(out.first); });

  auto params = sub_graph->parameters();
  auto mng_sub = sub_graph->manager();
  if (mng_sub == nullptr) {
    mng_sub = Manage(sub_graph, false);
    sub_graph->set_manager(mng_sub);
  }
  mindspore::HashMap<size_t, std::vector<std::pair<AnfNodePtr, size_t>>> in_out_pairs;
  for (size_t index = 0; index < params.size(); index++) {
    if (!params[index]->isa<Parameter>()) {
      continue;
    }
    auto &users = mng_sub->node_users()[params[index]];
    for (size_t j = 0; j < predecessors.size(); j++) {
      auto reliable = std::all_of(users.begin(), users.end(), [&predecessors, j](const std::pair<AnfNodePtr, int> &p) {
        return predecessors[j].count(p.first) > 0;
      });
      if (!reliable) {
        continue;
      }
      if (in_out_pairs.count(index) > 0) {
        in_out_pairs[index].push_back(outs[j]);
      } else {
        in_out_pairs[index] = {outs[j]};
      }
    }
  }
  return in_out_pairs;
}

// whether two nodes have same shape and type
// currently we only support inplace for tensor of type float32 to avoid precision problem
bool CheckShapeType(const AnfNodePtr &x, const AnfNodePtr &y) {
  if (!x->isa<CNode>() || !y->isa<CNode>()) {
    return false;
  }
  if (common::AnfAlgo::GetOutputInferShape(x, 0) != common::AnfAlgo::GetOutputInferShape(y, 0)) {
    return false;
  }
  return common::AnfAlgo::GetOutputInferDataType(x, 0) == TypeId::kNumberTypeFloat32 &&
         common::AnfAlgo::GetOutputInferDataType(y, 0) == TypeId::kNumberTypeFloat32;
}
}  // namespace

bool TensorInplace::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->get_return());
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto todos = TopoSort(func_graph->get_return());
  bool tensor_inplace_changed = false;
  for (auto &node : todos) {
    if (common::AnfAlgo::IsGraphKernel(node) && !IsBufferStitchNode(node)) {
      auto sub_func_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
      MS_EXCEPTION_IF_NULL(sub_func_graph);
      // currently support 1. elemwise/broadcast kernel 2.reduce kernel without multi-filters
      if (!CheckComputeType(sub_func_graph)) {
        continue;
      }
      auto in_out_pairs = FindInputOutputPairs(sub_func_graph);
      auto cnode = node->cast<CNodePtr>();
      for (size_t i = 1; i < cnode->inputs().size(); i++) {
        if (in_out_pairs.count(i - 1) == 0) {
          continue;
        }
        if (NoUsersAfterCurNode(cnode->input(i), node, mng)) {
          // input - output pair suitable for inplace assign
          auto outs = in_out_pairs[i - 1];
          auto candidate =
            std::find_if(outs.begin(), outs.end(), [&cnode, i](const std::pair<AnfNodePtr, size_t> &node) {
              return CheckShapeType(cnode->input(i), node.first);
            });
          if (candidate != outs.end()) {
            tensor_inplace_changed = true;
            InplaceAssignerInfo new_op_info;  // output info
            new_op_info.op_node = candidate->first->cast<CNodePtr>();
            new_op_info.real_output_num = AnfAlgo::GetOutputTensorNum(cnode);
            new_op_info.real_output_index = candidate->second;
            new_op_info.inplace_to_origin_input = SizeToInt(i) - 1;
            // modify graph kernel's abstract, kernelBuildInfo and insert assign
            ProcessOriginCNode(cnode, {{new_op_info, cnode->input(i)}});
            // reconnect output's user to input
            ProcessOriginCNodeUser(func_graph, cnode, {{new_op_info, cnode->input(i)}}, mng);
            break;
          }
        }
      }
    }
  }
  if (tensor_inplace_changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return tensor_inplace_changed;
}
}  // namespace mindspore::graphkernel
