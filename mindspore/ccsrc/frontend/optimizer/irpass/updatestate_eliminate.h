/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_UPDATESTATE_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_UPDATESTATE_ELIMINATE_H_

#include "include/frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"

namespace mindspore::opt::irpass {
// Eliminate useless node that only used by associated update_state.
class UpdatestateUselessNodeEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};

// Eliminate UpdateStates that attaches a no-side-effect node.
class UpdatestatePureNodeEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};

// Eliminate redundant UpdateState/Depend pair nodes caused by inline.
class UpdatestateDependEliminater {
 public:
  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer);
};

// Eliminate UpdateStates between Assign nodes.
// Eliminate UpdateStates between Assign and MakeTuple.
class UpdatestateAssignEliminater {
 public:
  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer);
};

// Eliminate UpdateStates for consecutive Loads.
class UpdatestateLoadsEliminater {
 public:
  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer);
};

// SwitchCallMonadParameterEliminater eliminates Monad parameter in switch call.
class SwitchCallMonadParameterEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};

AnfNodePtr EliminateUpdateStateMakeTupleWithUselessLoadNode(const CNodePtr &update_state_node);
}  // namespace mindspore::opt::irpass

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_UPDATESTATE_ELIMINATE_H_
