/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "backend/common/graph_kernel/symbol_engine_extender.h"

#include <memory>
#include <map>
#include <queue>
#include <stack>
#include <unordered_set>
#include <string>
#include <vector>
#include "ir/anf.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/graph_kernel/core/graph_builder.h"
#include "symbolic_shape/symbol_engine.h"
#include "include/common/symbol_engine/symbol_engine_impl.h"
#include "backend/common/graph_kernel/adapter/symbol_engine_builder.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"

namespace mindspore::graphkernel {

bool IsBeginOp(const AnfNodePtr &node, const SymbolEnginePtr &main_engine) {
  if (main_engine->IsDependShape(node) && common::AnfAlgo::IsDynamicShape(node) &&
      !common::AnfAlgo::IsDynamicRankNode(node)) {
    MS_LOG(DEBUG) << "A begin op: " << node->DebugString();
    return true;
  }
  return false;
}

bool IsClusterableOp(const AnfNodePtr &node, const SymbolEnginePtr &main_engine) {
  if (!node->isa<CNode>()) {
    return false;
  }
  if (IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
    return false;
  }
  return main_engine->IsDependValue(node);
}

AnfNodePtrList FindNodesDependOnValue(const AnfNodePtr &base_node, const SymbolEnginePtr &main_engine) {
  std::queue<AnfNodePtr> todo;
  todo.push(base_node);
  std::unordered_set<AnfNodePtr> black_set;
  std::unordered_set<AnfNodePtr> nodes_set;
  while (!todo.empty()) {
    auto node = todo.front();
    todo.pop();
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) continue;
    // Preserve the control flow introduced by UpdateState operation
    if (IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      MS_LOG(DEBUG) << "Skipping input of node: " << node->fullname_with_scope();
      for (size_t i = 1; i < cnode->size(); ++i) {
        auto input_node = cnode->input(i);
        if (input_node->cast<CNodePtr>() == nullptr) {
          continue;
        }
        MS_LOG(DEBUG) << "--- input of UpdateState: " << input_node->fullname_with_scope();
        black_set.insert(input_node);
      }
      continue;
    }
    MS_LOG(DEBUG) << "Find nodes for cnode: " << cnode->DebugString();
    for (size_t i = 1; i < cnode->size(); ++i) {
      auto input_node = cnode->input(i);
      if (input_node->cast<CNodePtr>() == nullptr) {
        continue;
      }
      MS_LOG(DEBUG) << "--- " << i - 1 << " th input: " << input_node->fullname_with_scope();
      MS_LOG(DEBUG) << "------ "
                    << "IsDependValue: " << main_engine->IsDependValue(input_node);
      if (IsPrimitiveCNode(input_node, prim::kPrimUpdateState)) {
        todo.push(input_node);
        nodes_set.insert(input_node);
        black_set.insert(input_node);
        continue;
      }
      if (nodes_set.find(input_node) == nodes_set.end() && black_set.find(input_node) == black_set.end()) {
        todo.push(input_node);
        nodes_set.insert(input_node);
        MS_LOG(DEBUG) << "------ is a candidate";
      }
    }
  }

  auto include_func = [&main_engine, &base_node, &black_set](const AnfNodePtr &node) {
    if (node == base_node) {
      return FOLLOW;
    }
    if (black_set.find(node) != black_set.end()) {
      return EXCLUDE;
    }
    if (IsClusterableOp(node, main_engine)) {
      MS_LOG(DEBUG) << "--- is a candidate: " << node->fullname_with_scope();
      return FOLLOW;
    } else {
      return EXCLUDE;
    }
  };
  auto res = TopoSort(base_node, SuccIncoming, include_func);
  return res;
}

/**
 * @brief Extends the given cnode to include shape calc part.
 * @param node The kernel.
 * @param main_engine The main SymbolEngine.
 * @return true if the subgraph was changed, false otherwise.
 */
bool ExtendNode(const AnfNodePtr &node, const SymbolEnginePtr &main_engine, const FuncGraphPtr &main_fg) {
  ClusterConfig config;
  config.inline_sub_func_graph = false;
  config.only_output_basenode = true;
  config.sort_parameter = true;

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto nodes = FindNodesDependOnValue(node, main_engine);

  if (nodes.size() > 1) {
    MS_LOG(DEBUG) << "The size of list of nodes to be clustered: " << nodes.size();
    config.base_node = node;
    cnode->AddAttr("real kernel", MakeValue<int32_t>(1));
    // Check if the symbol engine supports inferring for the graph, if not, skip cluster of this graph
    auto [fg, inputs, outputs] = BuildSingleGraphFromNodes(nodes, config);
    auto symbol_engine = symshape::SymbolEngineImpl::Build(fg);
    if (!symbol_engine->SupportInfer()) {
      MS_LOG(INFO) << "symbol engine doesn't support infer shape of node: " << node->fullname_with_scope();
      return false;
    }
    auto new_cnode = ReplaceNodesWithGraphKernelFuncGraph(main_fg, fg, inputs, outputs);
    auto fuse_op_name = GkUtils::ExtractGraphKernelName(nodes, "", "extended");
    fg->set_attr(kAttrKernelPacketNode, MakeValue(fuse_op_name));
    new_cnode->AddAttr(kAttrKernelPacketNode, MakeValue(true));
    return true;
  }
  return false;
}

bool SymbolEngineExtender::Run(const FuncGraphPtr &func_graph) {
  // Find the manager for the FuncGraph.
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  SymbolEnginePtr main_engine = func_graph->symbol_engine();
  MS_EXCEPTION_IF_NULL(main_engine);
  // Find all cnodes.
  auto cnodes = TopoSort(func_graph->output(), SuccIncoming, [](const AnfNodePtr &node) {
    if (node->isa<CNode>()) {
      return FOLLOW;
    }
    return EXCLUDE;
  });

  bool changed = false;
  // Process each subgraph.
  for (auto cnode : cnodes) {
    if (!IsBeginOp(cnode, main_engine)) {
      continue;
    }
    if (ExtendNode(cnode, main_engine, func_graph)) {
      changed = true;
    }
  }

  // Update the manager.
  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}

bool ConvertCallToPrim::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  bool changed = false;
  auto todos = TopoSort(func_graph->output());
  for (auto node : todos) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || !cnode->HasAttr(fg_name_)) {
      continue;
    }
    auto sub_fg = GetCNodeFuncGraph(node);
    if (sub_fg != nullptr) {
      MS_LOG(DEBUG) << "ConvertCallToPrim: find a " << fg_name_ << " node: " << cnode->fullname_with_scope();
      AnfNodePtrList new_inputs = node->cast<CNodePtr>()->inputs();
      auto new_prim = std::make_shared<Primitive>(prim_name_, sub_fg->attrs());
      new_inputs[0] = NewValueNode(new_prim);
      new_prim->AddAttr(kAttrFuncGraph, sub_fg);
      auto newnode = func_graph->NewCNode(new_inputs);
      newnode->CloneCNodeInfo(cnode);
      mng->Replace(node, newnode);
      changed = true;
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel
