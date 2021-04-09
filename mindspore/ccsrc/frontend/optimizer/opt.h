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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_OPT_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_OPT_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/operator/ops.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {

// Define the interaction mode between an Optimize pass and Renormalize pass
// FORCE_RENORM: if the pass modified the graph then the next Renormalize will be executed
// CHECK_RENORM: check if the new node is un-typed to decide if the next Renormalize will be executted
enum RenormAction : int64_t { FORCE_RENORM = 0, CHECK_RENORM };

class Substitution {
 public:
  OptimizerCallerPtr transform_;
  std::string name_;
  PredicateFuncType predicate_{nullptr};
  // an enum to mark this Substitution relation to renormalize pass
  RenormAction renorm_action_;
  Substitution(const OptimizerCallerPtr &transform, const std::string &name, const PredicateFuncType &predicate,
               const RenormAction &renorm_action)
      : transform_(transform), name_(name), predicate_(predicate), renorm_action_(renorm_action) {}
  ~Substitution() = default;
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node);
};

using SubstitutionPtr = std::shared_ptr<Substitution>;

SubstitutionPtr MakeSubstitution(const OptimizerCallerPtr &transform, const std::string &name, const PrimitivePtr &prim,
                                 const RenormAction &action_renorm = CHECK_RENORM);
SubstitutionPtr MakeSubstitution(const OptimizerCallerPtr &transform, const std::string &name,
                                 const std::vector<PrimitivePtr> &prims,
                                 const RenormAction &action_renorm = CHECK_RENORM);
SubstitutionPtr MakeSubstitution(const OptimizerCallerPtr &transform, const std::string &name,
                                 const PredicateFuncType &predicate, const RenormAction &action_renorm = CHECK_RENORM);

enum OptTraverseSubstitutionsMode { kOptTraverseFromIRToSubstitutions = 0, kOptTraverseFromSubstitutionsToIR };

class SubstitutionList {
 public:
  explicit SubstitutionList(const std::vector<SubstitutionPtr> &patterns, bool is_once = false,
                            bool global_sensitive = false)
      : list_(patterns), is_once_(is_once), global_sensitive_(global_sensitive) {}
  ~SubstitutionList() = default;

  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) const;

 private:
  bool ApplyIRToSubstitutions(const OptimizerPtr &optimizer, const FuncGraphPtr &func_graph) const;
  bool ApplySubstitutionToIR(const OptimizerPtr &optimizer, const AnfNodePtr &node, const SubstitutionPtr &sub) const;
  bool ApplySubstitutionsToIR(const OptimizerPtr &optimizer, const FuncGraphPtr &func_graph) const;
  void DisplayStatusOfSubstitution(const std::unordered_map<std::string, std::vector<bool>> &status,
                                   const OptimizerPtr &optimizer, size_t space) const;

  std::vector<SubstitutionPtr> list_;
  // a flag to mark this list of Substitution can only be executed only once
  bool is_once_;
  bool global_sensitive_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_OPT_H_
