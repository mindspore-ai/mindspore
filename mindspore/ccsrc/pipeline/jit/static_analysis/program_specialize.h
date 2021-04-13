/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_SPECIALIZE_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_SPECIALIZE_H_

#include <memory>
#include <string>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ir/anf.h"
#include "ir/func_graph_cloner.h"
#include "pipeline/jit/static_analysis/evaluator.h"

namespace mindspore {
namespace abstract {
enum SpecializeStatusCode {
  kSpecializeSuccess = 0,
  kSpecializeFindUniqueArgvalDead = 1,  // Dead Node
  kSpecializeFindUniqueArgvalPoly = 2,  // Poly Node
  kSpecializeFailure = 0xFF
};

class FuncGraphSpecializer;

// Specialize a func graph using analyzed abstract values.
class ProgramSpecializer {
 public:
  explicit ProgramSpecializer(const std::shared_ptr<AnalysisEngine> &engine) : engine_(engine) {
    mng_ = engine_->func_graph_manager();
  }
  ~ProgramSpecializer() = default;
  // Run the program specializer on the topmost graph in the given context.
  FuncGraphPtr Run(const FuncGraphPtr &fg, const AnalysisContextPtr &context);
  const std::unordered_set<AnfNodePtr> &seen() const { return seen_; }
  void AddSeen(const AnfNodePtr &node) { (void)seen_.insert(node); }

  std::shared_ptr<FuncGraphSpecializer> GetFuncGraphSpecializer(const AnalysisContextPtr &context);
  // Specialze one FuncGraph in a given context.
  FuncGraphPtr SpecializeFuncGraph(const FuncGraphPtr &fg, const AnalysisContextPtr &context);

  std::shared_ptr<AnalysisEngine> engine() { return engine_; }

 private:
  std::shared_ptr<AnalysisEngine> engine_;
  std::unordered_set<AnfNodePtr> seen_;
  FuncGraphManagerPtr mng_;
  std::unordered_map<AnalysisContextPtr, std::shared_ptr<FuncGraphSpecializer>, ContextHasher, ContextEqual>
    specializations_;
};

class FuncGraphSpecializer : public std::enable_shared_from_this<FuncGraphSpecializer> {
 public:
  FuncGraphSpecializer(ProgramSpecializer *const s, const FuncGraphPtr &fg, const AnalysisContextPtr &context);
  virtual ~FuncGraphSpecializer() {
    specializer_ = nullptr;
    repl_node_ = nullptr;
  }
  void Run();
  FuncGraphPtr specialized_func_graph() { return specialized_func_graph_; }

 private:
  ProgramSpecializer *specializer_;
  FuncGraphPtr func_graph_;
  FuncGraphPtr specialized_func_graph_;
  AnalysisContextPtr context_;
  std::shared_ptr<FuncGraphSpecializer> parent_;
  std::shared_ptr<AnalysisEngine> engine_;
  ClonerPtr cloner_;
  // ProcessNode-> [cloner_->CloneDisconnected] will clone AnfNode again.
  // So, repl_node_ should pointer to GraphCloner->repl_node_ other than a copy of that.
  std::unordered_map<AnfNodePtr, AnfNodePtr> *repl_node_;
  std::vector<AnfNodePtr> todo_;
  std::unordered_set<AnfNodePtr> marked_;
  std::unordered_map<EvaluatorPtr, EvaluatorCacheMapPtr> evalcaches_;

  void FirstPass();
  void SecondPass();
  void ProcessNode(const AnfNodePtr &node);
  void ProcessCNode(const CNodePtr &new_node);

  AnfNodeConfigPtr MakeConfig(const AnfNodePtr &node);
  inline void AddTodoItem(const AnfNodePtr &node) { todo_.push_back(node); }
  // Get node replicated by Cloner.
  AnfNodePtr GetReplicatedNode(const AnfNodePtr &node);
  // Replicated node which is not used directly by a func graph, so it's not searchable from it's return node
  // (disconnected).
  AnfNodePtr ReplicateDisconnectedNode(const AnfNodePtr &node);

  // Build a value node from parameter if the function graph has special flag to hint it can be done.
  AnfNodePtr BuildSpecializedParameterNode(const CNodePtr &new_node);

  // Build a value node if ival is constant and not any-value
  AnfNodePtr BuildPossibleValueNode(const AnfNodePtr &origin_node, const AbstractBasePtr &ival,
                                    const AttrValueMapPtr &attrs);
  // Build a replaceable node for iconf->node; it may be a replicated forwarded CNode in static analysis or just a
  // replicated node.
  AnfNodePtr BuildReplacedNode(const AnfNodeConfigPtr &conf);
  // Build a specialized node from given argvals;
  AnfNodePtr BuildSpecializedNode(const AnfNodePtr &node, const AbstractBasePtr &abs,
                                  const AbstractBasePtrList &argvals);
  AnfNodePtr BuildSpecializedNodeInner(const AnfNodePtr &node, const AbstractBasePtr &abs,
                                       const AbstractFunctionPtr &func, const AbstractBasePtrList &args,
                                       SpecializeStatusCode *errcode);

  // Find the unique argument values which can be used to specialize a primitive or graph function.
  SpecializeStatusCode FindUniqueArgvals(const AbstractFunctionPtr &fn, const EvaluatorPtr &eval,
                                         const AbstractBasePtrList &argvals,
                                         std::pair<AbstractBasePtrList, AbstractBasePtr> *result);
  // Get cache, it may be eval's cache or cache built from broaded argument values.
  const EvaluatorCacheMapPtr &GetEvalCache(const EvaluatorPtr &eval);
  // Try to build unique argvals from the broaded arg vals if it is unique.
  std::pair<AbstractBasePtrList, AbstractBasePtr> BuildFromBroadedArgsVal(const EvaluatorPtr &eval);
  void UpdateNewCNodeInputs(const AnfNodePtr &node, const AnfNodePtr &new_node);
};
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_SPECIALIZE_H_
