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

#include "frontend/optimizer/ad/bprop_utils.h"

#include <string>
#include <regex>
#include <algorithm>
#include <vector>
#include <memory>
#include "include/common/utils/primitive_utils.h"
#include "include/common/debug/common.h"
#include "utils/file_utils.h"
#include "utils/system/sha256.h"
#include "mindspore/core/load_mindir/load_model.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "frontend/expander/bprop/bprop.h"
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "frontend/expander/utils.h"
#include "include/common/debug/dump_proto.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/expander/bprop/bprop_meta_func_graph.h"

namespace mindspore {
namespace ad {
FuncGraphPtr GetBprop(const PrimitivePtr &prim, const pipeline::ResourceBasePtr &resources, const CNodePtr &cnode) {
  // Set a child scope named "grad'PrimitiveName'" for the bprop function,
  // and add "Gradients" to the front.
  static const std::string gradients_scope = "Gradients/";
  static const std::string grad_op_child_scope_prefix = "/grad";
  MS_EXCEPTION_IF_NULL(prim);
  const auto &prim_name = prim->name();
  auto scope = std::make_shared<Scope>(gradients_scope + ScopeManager::GetInstance().GetCurrentScope()->name() +
                                       grad_op_child_scope_prefix + prim_name);
  ScopeGuard scope_guard(scope);

  // Firstly we get bprop from expander. If failed, try mindir. If still failed, try the python bprop function.
  FuncGraphPtr func_graph = expander::bprop::GetBpropMetaFuncGraph(prim, cnode);
  if (func_graph != nullptr) {
    return func_graph;
  }
  py::function fn;
  if (prim->is_base()) {
    fn = GetBpropFunction(prim_name);
  } else {
    fn = prim->cast_ptr<PrimitivePy>()->GetBpropFunction();
    if (py::isinstance<py::none>(fn)) {
      fn = GetBpropFunction(prim_name);
    }
  }
  if (!fn || py::isinstance<py::none>(fn)) {
    MS_LOG(INFO) << "Fail to find bprop function for " << prim_name << ". fn: " << py::str(fn);
    return nullptr;
  }
  func_graph = parse::ParsePythonCode(fn);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Fail to parse bprop function for " << prim_name << ".";
    return nullptr;
  }
  auto bprop_flag = GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_BACKPROP);
  if (bprop_flag) {
    func_graph->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);
  }
  pipeline::ResourceBasePtr res = (resources != nullptr) ? resources : std::make_shared<pipeline::Resource>();
  (void)parse::ResolveFuncGraph(func_graph, res, false);
  return func_graph;
}
}  // namespace ad
}  // namespace mindspore
