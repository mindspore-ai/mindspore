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

#include <algorithm>
#include <memory>
#include <map>
#include <queue>
#include <stack>
#include <unordered_set>
#include <string>
#include <vector>
#include "ir/anf.h"
#include "ir/manager.h"
#include "utils/anf_utils.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/graph_kernel/core/graph_builder.h"
#include "symbolic_shape/symbol_engine.h"
#include "backend/common/graph_kernel/symbol_engine/kernel_packet_engine.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "utils/log_adapter.h"

namespace mindspore::graphkernel {
bool IsBeginOp(const AnfNodePtr &node, const SymbolEnginePtr &main_engine) {
  if (main_engine->IsDependShape(node) && common::AnfAlgo::IsDynamicShape(node) &&
      !common::AnfAlgo::IsDynamicRankNode(node) && AnfUtils::IsRealCNodeKernel(node)) {
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
  std::queue<AnfNodePtr> todo({base_node});
  std::unordered_set<AnfNodePtr> black_set;
  std::unordered_set<AnfNodePtr> nodes_set;
  while (!todo.empty()) {
    auto node = todo.front();
    todo.pop();
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    // Preserve the control flow introduced by UpdateState operation
    if (IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      MS_LOG(DEBUG) << "Skipping input of node: " << node->fullname_with_scope();
      for (size_t i = 1; i < cnode->size(); ++i) {
        auto input_node = cnode->input(i);
        if (input_node->isa<CNode>()) {
          MS_LOG(DEBUG) << "Add the input of UpdateState to black_set: " << input_node->fullname_with_scope();
          black_set.insert(input_node);
        }
      }
      continue;
    }
    MS_LOG(DEBUG) << "Find nodes for cnode: " << cnode->DebugString();
    for (size_t i = 1; i < cnode->size(); ++i) {
      auto input_node = cnode->input(i);
      if (!input_node->isa<CNode>()) {
        continue;
      }
      MS_LOG(DEBUG) << "The " << (i - 1) << " th input: " << input_node->fullname_with_scope()
                    << " depend value: " << main_engine->IsDependValue(input_node);
      if (IsPrimitiveCNode(input_node, prim::kPrimUpdateState)) {
        todo.push(input_node);
        nodes_set.insert(input_node);
        black_set.insert(input_node);
        continue;
      }
      if (nodes_set.find(input_node) == nodes_set.end() && black_set.find(input_node) == black_set.end()) {
        todo.push(input_node);
        nodes_set.insert(input_node);
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
      MS_LOG(DEBUG) << "Node " << node->fullname_with_scope() << " is a candidate.";
      return FOLLOW;
    }
    return EXCLUDE;
  };
  return TopoSort(base_node, SuccIncoming, include_func);
}

AnfNodePtrList TopoSortFromRoots(const AnfNodePtrList &roots, const IncludeFunc &include) {
  auto seen = NewSeenGeneration();
  std::stack<AnfNodePtr> ops;
  std::for_each(roots.cbegin(), roots.cend(), [&ops](const AnfNodePtr &n) {
    MS_EXCEPTION_IF_NULL(n);
    ops.push(n);
  });

  AnfNodePtrList res;
  while (!ops.empty()) {
    auto node = ops.top();
    if (node->extra_seen_ == seen) {
      ops.pop();
      continue;
    }
    auto inc_res = include(node);
    if (node->seen_ == seen) {
      node->extra_seen_ = seen;
      if (inc_res != EXCLUDE) {
        res.push_back(node);
      }
      ops.pop();
      continue;
    }
    node->seen_ = seen;
    if (inc_res != FOLLOW) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    if (cnode != nullptr) {
      for (size_t i = 1; i < cnode->size(); ++i) {
        auto input_node = cnode->input(i);
        MS_EXCEPTION_IF_NULL(input_node);
        ops.push(input_node);
      }
    }
  }

  return res;
}

void FuseOnlyShapeDependedNodes(const AnfNodePtr &base_node, const SymbolEnginePtr &main_engine,
                                AnfNodePtrList *nodes) {
  MS_EXCEPTION_IF_NULL(nodes);
  // Find input-edge nodes except base_node.
  std::unordered_set<AnfNodePtr> inner;
  (void)std::for_each(nodes->cbegin(), nodes->cend(), [&inner](const AnfNodePtr &n) -> void { inner.insert(n); });

  AnfNodePtrList input_cnodes;
  for (const auto &node : *nodes) {
    if (node == base_node || !node->isa<CNode>()) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    for (size_t i = 1; i < cnode->size(); ++i) {
      auto input_node = cnode->input(i);
      if (IsPrimitiveCNode(input_node, prim::kPrimTupleGetItem)) {
        continue;
      }
      if (input_node->isa<CNode>() && inner.find(input_node) == inner.end()) {
        input_cnodes.push_back(input_node);
      }
    }
  }

  if (input_cnodes.empty()) {
    return;
  }

  std::unordered_set<AnfNodePtr> excludes;
  auto base_cnode = base_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(base_cnode);
  for (size_t i = 1; i < base_cnode->size(); ++i) {
    auto base_input = base_cnode->input(i);
    if (!main_engine->IsDependValue(base_input)) {
      excludes.insert(base_input);
    }
  }

  // Find input-edge node which is only shape-depended and used by inner-nodes. Therefore, only the shape-infer will be
  // executed without the kernel-launch, memory copy and so on.
  auto main_graph = base_node->func_graph();
  MS_EXCEPTION_IF_NULL(main_graph);
  auto mng = main_graph->manager();
  if (mng == nullptr) {
    mng = Manage(main_graph, false);
    main_graph->set_manager(mng);
  }

  const NodeUsersMap &users = mng->node_users();
  auto target_include_func = [&main_engine, &users, &inner, &excludes](const AnfNodePtr &node) {
    if (!node->isa<CNode>() || excludes.find(node) != inner.end() || !main_engine->IsDependShape(node)) {
      return EXCLUDE;
    }

    auto user_iter = users.find(node);
    if (user_iter == users.end()) {
      MS_LOG(INTERNAL_EXCEPTION) << node->fullname_with_scope() << " must have one users at least!";
    }

    // Maybe clone one when more than one use is better...
    for (auto &iter : user_iter->second) {
      auto user_node = iter.first;
      MS_EXCEPTION_IF_NULL(user_node);
      if (inner.find(user_node) == inner.end()) {
        return EXCLUDE;
      }
    }

    return FOLLOW;
  };

  auto new_fuses = TopoSortFromRoots(input_cnodes, target_include_func);
  (void)new_fuses.insert(new_fuses.end(), nodes->begin(), nodes->end());
  *nodes = std::move(new_fuses);
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
    FuseOnlyShapeDependedNodes(node, main_engine, &nodes);
    MS_LOG(DEBUG) << "The size of list of nodes to be clustered: " << nodes.size();
    config.base_node = node;
    cnode->AddAttr("real kernel", MakeValue<int32_t>(1));
    // Check if the symbol engine supports inferring for the graph, if not, skip cluster of this graph
    auto [fg, inputs, outputs] = BuildSingleGraphFromNodes(nodes, config);
    auto symbol_engine = symshape::KernelPacketEngine::Build(fg);
    if (!symbol_engine->SupportInfer()) {
      MS_LOG(INFO) << "symbol engine doesn't support infer shape of node: " << node->fullname_with_scope();
      return false;
    }
    auto new_cnode = ReplaceNodesWithGraphKernelFuncGraph(main_fg, fg, inputs, outputs);
    auto fuse_op_name = GkUtils::ExtractGraphKernelName(nodes, "", "extended");
    fg->set_attr(kAttrKernelPacketNode, MakeValue(fuse_op_name));
    new_cnode->AddAttr(kAttrToPrim, MakeValue(prim::kPrimKernelPacket->name()));
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
    if (cnode == nullptr || !cnode->HasAttr(kAttrToPrim)) {
      continue;
    }
    auto sub_fg = GetCNodeFuncGraph(node);
    if (sub_fg != nullptr) {
      AnfNodePtrList new_inputs = node->cast<CNodePtr>()->inputs();
      auto new_prim = std::make_shared<Primitive>(GetValue<std::string>(cnode->GetAttr(kAttrToPrim)), sub_fg->attrs());
      new_inputs[0] = NewValueNode(new_prim);
      new_prim->AddAttr(kAttrFuncGraph, sub_fg);
      auto newnode = func_graph->NewCNode(new_inputs);
      newnode->CloneCNodeInfo(cnode);
      auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
      if (kernel_mod != nullptr) {
        kernel_mod->Init(new_prim, {}, {});
      }
      mng->Replace(node, newnode);
      changed = true;
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel
