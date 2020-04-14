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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_OPT_H_
#define MINDSPORE_CCSRC_OPTIMIZER_OPT_H_

#include <vector>
#include <string>
#include <memory>

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "operator/ops.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
class Optimizer;

using OptimizerPtr = std::shared_ptr<Optimizer>;
using OptimizerWeakPtr = std::weak_ptr<Optimizer>;

using PredicateFuncType = std::function<bool(const AnfNodePtr &)>;
using TransformFuncType = std::function<AnfNodePtr(const OptimizerPtr &, const AnfNodePtr &)>;

// Define the interaction mode between an Optimize pass and Renormalize pass
// FORCE_RENORM: if the pass modified the graph then the next Renormalize will be executed
// CHECK_RENORM: check if the new node is un-typed to decide if the next Renormalize will be executted
enum RenormAction : int { FORCE_RENORM = 0, CHECK_RENORM };

class Substitution {
 public:
  TransformFuncType transform_{nullptr};
  std::string name_;
  PredicateFuncType predicate_{nullptr};
  // an enum to mark this Substitution relation to renormalize pass
  RenormAction renorm_action_;
  explicit Substitution(const TransformFuncType &transform, const std::string &name, const PredicateFuncType &predicate,
                        const RenormAction &renorm_action)
      : transform_(transform), name_(name), predicate_(predicate), renorm_action_(renorm_action) {}
  ~Substitution() = default;
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) const;
};

using SubstitutionPtr = std::shared_ptr<Substitution>;

SubstitutionPtr MakeSubstitution(const TransformFuncType &transform, const std::string &name, const PrimitivePtr &prim,
                                 const RenormAction &action_renorm = CHECK_RENORM);
SubstitutionPtr MakeSubstitution(const TransformFuncType &transform, const std::string &name,
                                 const std::vector<PrimitivePtr> &prims,
                                 const RenormAction &action_renorm = CHECK_RENORM);
SubstitutionPtr MakeSubstitution(const TransformFuncType &transform, const std::string &name,
                                 const PredicateFuncType &predicate, const RenormAction &action_renorm = CHECK_RENORM);

class SubstitutionList {
 public:
  explicit SubstitutionList(const std::vector<SubstitutionPtr> &patterns, bool is_once = false)
      : list_(patterns), is_once_(is_once) {}
  ~SubstitutionList() = default;

  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) const;

 private:
  bool ApplyTransform(const OptimizerPtr &optimizer, const AnfNodePtr &node, const SubstitutionPtr &transform) const;
  std::vector<SubstitutionPtr> list_;
  // a flag to mark this list of Substitution can only be executed only once
  bool is_once_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_OPT_H_
