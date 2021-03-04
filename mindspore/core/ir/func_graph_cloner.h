/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/manager.h"

namespace mindspore {
class Cloner;
using ClonerPtr = std::shared_ptr<Cloner>;

enum CloneType { kBasic = 0, kInline = 1, kLifting = 2, kDropping = 3 };

struct CloneInfo {
  FuncGraphPtr origin;
  FuncGraphPtr target;
  AnfNodePtrList params;
};

class Cloner {
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
  std::unordered_map<AnfNodePtr, AnfNodePtr> *cloned_node() { return &repl_node_; }
  std::unordered_map<FuncGraphPtr, FuncGraphPtr> &cloned_func_graph() { return repl_func_graph_; }

  // Scope of cloned graphs
  void set_scope(const ScopePtr &scope) { scope_ = scope; }
  const ScopePtr scope() const { return scope_; }

  std::unordered_map<AnfNodePtr, AnfNodePtr> repl_node_;
  std::unordered_map<FuncGraphPtr, FuncGraphPtr> repl_func_graph_;

 private:
  void CloneNodes();
  void LinkEdges();
  void SetDefaults();
  void CloneNode(const AnfNodePtr &node, const FuncGraphPtr &target);
  void CloneValueNode(const AnfNodePtr &node);
  void CloneValueNode(const AnfNodePtr &node, const FuncGraphPtr &target);
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
  void SetFuncGraphInfo(const FuncGraphPtr &func_graph, FuncGraphPtr *const target_func_graph);
  void CloneParameters(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph);
  void GenParameters(const FuncGraphPtr &func_graph);
  void CloneParameter(const ParameterPtr &param, const AnfNodePtr &node);
  ParameterPtr AddParameter(const FuncGraphPtr &func_graph, const AnfNodePtr &node, bool is_add = true);
  void AddParameters(const FuncGraphPtr &func_graph, const AnfNodePtrList &params, AnfNodePtrList *const lift_params,
                     AnfNodePtrList *const input_params);
  void AddInputs(const FuncGraphPtr &func_graph_user, const FuncGraphPtr &func_graph, const AnfNodePtrList &params);
  void OrderParameters(const FuncGraphPtr &func_graph, const AnfNodePtrList &inputs);
  void SetEdges(const FuncGraphPtr &func_graph);
  void LiftParameters(const FuncGraphPtr &func_graph_user, const FuncGraphPtr &func_graph,
                      const AnfNodePtrList &params);
  void Lift();
  void LiftParameters();

  bool clone_all_valuenodes_;
  bool clone_all_child_graphs_;
  bool clone_all_used_graphs_;
  TraceInfoPtr relation_;
  TraceInfoPtr target_relation_;
  FuncGraphManagerPtr manager_;
  FuncGraphTransaction transaction_;
  FuncGraphSet graph_set_;
  ScopePtr scope_;
  CloneType type_;
  std::list<CloneInfo> todo_;
  std::list<std::pair<CNodePtr, CNodePtr>> nodes_;
  std::unordered_map<FuncGraphPtr, bool> status_;
  std::unordered_map<FuncGraphPtr, std::unordered_map<AnfNodePtr, AnfNodePtr>> repl_map_node_;
  std::unordered_map<FuncGraphPtr, std::unordered_map<FuncGraphPtr, AnfNodePtr>> repl_map_func_graph_;
  std::unordered_map<FuncGraphPtr, AnfNodePtrList> repl_func_graph_params_;
};

AnfNodePtr InlineClone(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph,
                       const AnfNodePtrList &func_graph_args, const ScopePtr &scope = nullptr);

FuncGraphPtr LiftingClone(const FuncGraphPtr &func_graph);

ClonerPtr SpecializerClone(const FuncGraphPtr &func_graph, const TraceInfoPtr &relation);

FuncGraphPtr TransformableClone(const FuncGraphPtr &func_graph,
                                const TraceInfoPtr &relation = std::make_shared<TraceTransform>());
FuncGraphPtr BasicClone(const FuncGraphPtr &func_graph);
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_FUNC_GRAPH_CLONER_H_
