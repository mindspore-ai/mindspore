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

#include "c_api/src/pass.h"
#include "frontend/optimizer/irpass/updatestate_eliminate.h"
#include "frontend/optimizer/auto_monad_eliminate.h"

bool AutoMonadElimOptPass(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->manager());
  py::scoped_interpreter py_scope;
  auto resource = std::make_shared<mindspore::pipeline::Resource>();
  resource->set_func_graph(func_graph);
  resource->set_manager(func_graph->manager());

  // opt::irpass::OptimizeIRPassLib is not used here to avoid double free problems in external calls.
  SubstitutionPtr updatestate_useless_node_eliminater =
    mindspore::opt::MakeSubstitution(std::make_shared<mindspore::opt::irpass::UpdatestateUselessNodeEliminater>(),
                                     "updatestate_useless_node_eliminater", mindspore::prim::kPrimUpdateState);
  SubstitutionPtr updatestate_pure_node_eliminater =
    mindspore::opt::MakeSubstitution(std::make_shared<mindspore::opt::irpass::UpdatestatePureNodeEliminater>(),
                                     "updatestate_pure_node_eliminater", mindspore::prim::kPrimUpdateState);
  SubstitutionPtr switch_call_monad_eliminater =
    mindspore::opt::MakeSubstitution(std::make_shared<mindspore::opt::irpass::SwitchCallMonadParameterEliminater>(),
                                     "switch_call_monad_eliminater", mindspore::opt::irpass::IsCNodeDup);

  OptPassConfig updatestate_eliminater = OptPassConfig({
    updatestate_useless_node_eliminater,
    updatestate_pure_node_eliminater,
    switch_call_monad_eliminater,
  });
  OptPassConfig updatestate_depend_eliminate = OptPassConfig(mindspore::opt::irpass::UpdatestateDependEliminater());
  OptPassConfig updatestate_assign_eliminate = OptPassConfig(mindspore::opt::irpass::UpdatestateAssignEliminater());
  OptPassConfig updatestate_loads_eliminate = OptPassConfig(mindspore::opt::irpass::UpdatestateLoadsEliminater());
  OptPassGroupMap elim_map({
    {"updatestate_eliminater", updatestate_eliminater},
    {"updatestate_depend_eliminate", updatestate_depend_eliminate},
    {"updatestate_assign_eliminate", updatestate_assign_eliminate},
    {"updatestate_loads_eliminate", updatestate_loads_eliminate},
    {"auto_monad_eliminator", OptPassConfig(mindspore::opt::AutoMonadEliminator())},
  });

  auto auto_monad_elim_opt = mindspore::opt::Optimizer::MakeOptimizer("auto_monad_elim", resource, elim_map);
  (void)auto_monad_elim_opt->step(func_graph, false);
  resource->Clean();
  return true;
}
