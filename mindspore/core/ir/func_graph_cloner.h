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

#ifndef MINDSPORE_CORE_IR_FUNC_GRAPH_CLONER_H_
#define MINDSPORE_CORE_IR_FUNC_GRAPH_CLONER_H_

#include <functional>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "utils/hashing.h"
#include "mindapi/base/macros.h"

namespace mindspore {
class Cloner;
using ClonerPtr = std::shared_ptr<Cloner>;
using NodeToNodeMap = mindspore::HashMap<AnfNodePtr, AnfNodePtr, PointerHash<AnfNodePtr>>;

enum CloneType { kBasic = 0, kInline = 1, kLifting = 2, kDropping = 3 };

struct CloneInfo {
  FuncGraphPtr origin;
  FuncGraphPtr target;
  AnfNodePtrList params;
};

struct UpdateInfo {
  UpdateInfo(const ScopePtr &scope, const NodeDebugInfoPtr &debug_info) : scope_(scope), debug_info_(debug_info) {}
  ~UpdateInfo() = default;

  ScopePtr scope_;
  NodeDebugInfoPtr debug_info_;
};

using UpdateInfoPtr = std::shared_ptr<UpdateInfo>;

class MS_CORE_API Cloner {
 public:
  explicit Cloner(const FuncGraphVector &func_graphs = {}, bool clone_all_valuenodes = false,
                  bool clone_all_child_graphs = true, bool clone_all_used_graphs = false,
                  const TraceInfoPtr &relation = std::make_shared<TraceCopy>(),
                  const TraceInfoPtr &target_relation = nullptr);
  ~Cloner() = default;
  void AddClone(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph = nullptr,
                const AnfNodePtrList &params = {}, CloneType type = kBasic);
  void Run();

  // Interfaces for specializer
  AnfNodePtr CloneDisconnected(const AnfNodePtr &root);
  AnfNodePtr operator[](const AnfNodePtr &node);
  FuncGraphPtr operator[](const FuncGraphPtr &func_graph);

  // Map of replicate nodes and graphs
  const NodeToNodeMap &cloned_nodes() const { return repl_node_; }
  const mindspore::HashMap<FuncGraphPtr, FuncGraphPtr> &cloned_func_graphs() const { return repl_func_graph_; }

  // Scope of cloned graphs
  void set_scope(const ScopePtr &scope) { scope_ = scope; }
  const ScopePtr scope() const { return scope_; }

  // When clone nodes, the same debug info and scope.
  void set_update_info(const UpdateInfoPtr &update_info) { update_info_ = update_info; }
  const UpdateInfoPtr update_info() const { return update_info_; }

  // set call node debug info of InlineClone.
  void set_inline_call_node_debug_info(const NodeDebugInfoPtr &call_debug_info) {
    inline_call_node_debug_info_ = call_debug_info;
  }

 private:
  void CloneNodes();
  void LinkEdges();
  void SetDefaults();
  void CloneNode(const AnfNodePtr &node, const FuncGraphPtr &target);
  void CloneValueNode(const AnfNodePtr &node);
  void CloneFuncGraphValueNode(const AnfNodePtr &node, const FuncGraphPtr &target);
  void CloneCNode(const AnfNodePtr &node, const FuncGraphPtr &target);
  void CloneParameter(const AnfNodePtr &node, const FuncGraphPtr &target, bool is_add = false);
  void CloneValueNodes(const FuncGraphPtr &func_graph);
  void AddChildGraphs(const FuncGraphPtr &func_graph);
  void AddTotalGraphs(const FuncGraphPtr &func_graph);
  bool CheckStatus(const FuncGraphPtr &func_graph, bool is_inline);
  void CloneAllNodes(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph);
  void CloneOrderList(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph);
  void CloneFuncGraphValueNodes(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph);
  void CloneFuncGraphDefaultValues(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph);
  void InlineCloneParameters(const FuncGraphPtr &func_graph, const AnfNodePtrList &params);
  void SetFuncGraphInfo(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph) const;
  void CloneParameters(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph);
  void GenParameters(const FuncGraphPtr &func_graph);
  void CloneParameter(const ParameterPtr &param, const AnfNodePtr &node) const;
  ParameterPtr AddParameter(const FuncGraphPtr &func_graph, const AnfNodePtr &node, bool is_add = true);
  void AddParameters(const FuncGraphPtr &func_graph, const AnfNodePtrList &params, AnfNodePtrList *const lift_params,
                     AnfNodePtrList *const input_params);
  void AddInputs(const FuncGraphPtr &func_graph_user, const FuncGraphPtr &func_graph, const AnfNodePtrList &params);
  void OrderParameters(const FuncGraphPtr &func_graph, const AnfNodePtrList &inputs, size_t arg_start_index);
  void SetEdges(const FuncGraphPtr &func_graph, FuncGraphTransaction *tx);
  void SetEdgesBfs(const FuncGraphPtr &root_fg, FuncGraphTransaction *tx);
  void LiftParameters(const FuncGraphPtr &func_graph_user, const FuncGraphPtr &func_graph,
                      const AnfNodePtrList &params);
  void Lift(const std::vector<FuncGraphPtr> &sorted);
  void LiftParameters(const FuncGraphVector &todo_func_graphs);
  bool IsLiftTopFuncGraph(const FuncGraphPtr &func_graph);

  bool clone_all_valuenodes_;
  bool clone_all_child_graphs_;
  bool clone_all_used_graphs_;
  TraceInfoPtr relation_;
  TraceInfoPtr target_relation_;
  NodeToNodeMap repl_node_;
  mindspore::HashMap<FuncGraphPtr, FuncGraphPtr> repl_func_graph_;
  FuncGraphManagerPtr manager_;
  FuncGraphSet graph_set_;
  ScopePtr scope_;
  UpdateInfoPtr update_info_;
  NodeDebugInfoPtr inline_call_node_debug_info_{nullptr};
  CloneType type_;
  std::vector<CloneInfo> todo_;
  mindspore::HashMap<FuncGraphPtr, bool> status_;
  mindspore::HashMap<FuncGraphPtr, NodeToNodeMap> repl_map_node_;
  mindspore::HashMap<FuncGraphPtr, mindspore::HashMap<FuncGraphPtr, AnfNodePtr>> repl_map_func_graph_;
  mindspore::HashMap<FuncGraphPtr, AnfNodePtrList> repl_func_graph_params_;
};

MS_CORE_API AnfNodePtr InlineClone(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph,
                                   const AnfNodePtrList &func_graph_args, const ScopePtr &scope = nullptr,
                                   const NodeDebugInfoPtr &call_debug_info = nullptr);

MS_CORE_API FuncGraphPtr LiftingClone(const FuncGraphPtr &func_graph);
MS_CORE_API FuncGraphVector LiftingCloneMulti(const FuncGraphVector &func_graphs);

MS_CORE_API ClonerPtr SpecializerClone(const FuncGraphPtr &func_graph, const TraceInfoPtr &relation);

MS_CORE_API FuncGraphPtr TransformableClone(const FuncGraphPtr &func_graph,
                                            const TraceInfoPtr &relation = std::make_shared<TraceTransform>());
MS_CORE_API FuncGraphPtr BasicClone(const FuncGraphPtr &func_graph, bool clone_value_nodes = false,
                                    const UpdateInfoPtr update_info = nullptr);
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_FUNC_GRAPH_CLONER_H_
