/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include <memory>
#include <string>
#include <utility>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/meta_func_graph.h"
#include "ir/func_graph_cloner.h"
#include "ir/manager.h"
#include "pipeline/resource.h"
#include "pipeline/parse/parse.h"
#include "optimizer/ad/dfunctor.h"
#include "optimizer/opt.h"
#include "operator/ops.h"
#include "operator/composite/composite.h"
#include "utils/symbolic.h"
#include "utils/primitive_utils.h"
#include "debug/info.h"
#include "debug/trace.h"

#include "./common.h"

namespace mindspore {
namespace ad {
using PatternListType = std::initializer_list<BaseRef>;
KPrim g_k_prims;

FuncGraphPtr KPrim::GetBprop(const PrimitivePtr &prim) {
  // Set a child scope named "grad'PrimitiveName'" for the bprop function,
  // and add "Gradients" to the front.
  static const std::string gradients_scope = "Gradients/";
  static const std::string grad_op_child_scope_prefix = "/grad";
  MS_EXCEPTION_IF_NULL(prim);
  auto scope = std::make_shared<Scope>(gradients_scope + ScopeManager::GetInstance().GetCurrentScope()->name() +
                                       grad_op_child_scope_prefix + prim->name());
  ScopeGuard scope_guard(scope);
  py::function fn = prim->is_base() ? GetBpropFunction(prim->name()) : prim->cast<PrimitivePyPtr>()->GetBpropFunction();
  if (fn == nullptr || py::isinstance<py::none>(fn)) {
    MS_LOG(DEBUG) << "Fail to find bprop function for " << prim->name() << ".";
    return nullptr;
  }
  FuncGraphPtr func_graph = parse::ParsePythonCode(fn);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Fail to parse bprop function for " << prim->name() << ".";
    return nullptr;
  }
  return func_graph;
}

FuncGraphPtr KPrim::GetFprop(const PrimitivePtr &prim) {
  static const std::string ad_module = "mindspore.ops._grad.grad_implementations";
  std::string func_name = "_fprop_" + prim->name();
  py::function fn = parse::python_adapter::GetPyFn(ad_module, func_name);
  auto func_graph = parse::ParsePythonCode(fn);
  MS_EXCEPTION_IF_NULL(func_graph);
  return BasicClone(func_graph);
}

MetaFuncGraphPtr KPrim::KMetaFuncGraph(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);

  auto iter = bprop_registry_meta_.find(prim);
  if (iter != bprop_registry_meta_.end()) {
    return iter->second;
  }

  if (prim->name() == "make_tuple") {
    MetaFuncGraphPtr meta = std::make_shared<prim::MakeTupleGradient>("make_tuple_gradient");
    bprop_registry_meta_[prim::kPrimMakeTuple] = meta;
    return meta;
  }

  MS_LOG(EXCEPTION) << "Fail to find bprop function for " << prim->name() << ".";
}

FuncGraphPtr KPrim::KPrimitive(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources) {
  if (!IsValueNode<Primitive>(value_node)) {
    MS_LOG(EXCEPTION) << "Primitive node is not valid.";
  }

  auto prim = value_node->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(prim);

  auto iter = bprop_registry_.find(prim);
  if (iter != bprop_registry_.end()) {
    return iter->second;
  }

  if (prim->Hash() == prim::kPrimSwitchLayer->Hash() && prim->name() == "switch_layer") {
    auto fprop = GetFprop(prim);
    fprop->transforms().emplace("primal", FuncGraphTransform(prim::kPrimSwitchLayer));
    bprop_registry_[prim::kPrimSwitchLayer] = fprop;
    return fprop;
  }

  if (prim->name() == "make_tuple") {
    return nullptr;
  }

  bool is_faked_bprop = false;
  FuncGraphPtr bprop_fg = nullptr;
  if (prim->Hash() == prim::kPrimHookBackward->Hash() && prim->name() == "HookBackward") {
    bprop_fg = BpropCut(value_node, resources);
  } else {
    bprop_fg = GetBprop(prim);
    if (bprop_fg == nullptr) {
      bprop_fg = FakeBprop(value_node, resources);
      is_faked_bprop = true;
    }
  }

  auto expanded_fg = BpropToK(prim, bprop_fg);
  if (expanded_fg == nullptr) {
    MS_LOG(EXCEPTION) << "Failed convert " << prim->name()
                      << " prim bprop function to J expanded func graph. NodeInfo: "
                      << trace::GetDebugInfo(bprop_fg->debug_info());
  }

  // To support primitives with variable params, do not cache faked bprop
  if (!is_faked_bprop) {
    // Set bprop_g graph cache
    bprop_registry_[prim] = expanded_fg;
  }
  return expanded_fg;
}

AnfNodePtr KPrim::BuildOutput(const FuncGraphPtr &bprop_fg) {
  // bprop_fg has been checked in caller
  if (IsPrimitiveCNode(bprop_fg->output(), prim::kPrimMakeTuple)) {
    // Set bprop output as (env, dx, dy, dz, ...)
    auto cbprop = bprop_fg->output()->cast<CNodePtr>();
    auto &inputs = cbprop->inputs();

    std::vector<AnfNodePtr> args;
    args.push_back(NewValueNode(prim::kPrimMakeTuple));
    args.push_back(NewValueNode(newenv));
    (void)args.insert(args.end(), inputs.begin() + 1, inputs.end());
    return NewCNode(args, bprop_fg);
  }

  // Set bprop output as (env, dx)
  std::string model_name("mindspore.ops.composite.multitype_ops.add_impl");
  std::string python_ops("_tuple_add");
  auto tuple = NewCNode({NewValueNode(prim::kPrimMakeTuple), NewValueNode(newenv)}, bprop_fg);
  return NewCNode({NewValueNode(prim::GetPythonOps(python_ops, model_name)), tuple, bprop_fg->output()}, bprop_fg);
}

void KPrim::TransformArgs(const FuncGraphManagerPtr &mng, const FuncGraphPtr &bprop_fg, const FuncGraphPtr &outer,
                          std::vector<AnfNodePtr> *const transf_args) {
  MS_EXCEPTION_IF_NULL(mng);
  // bprop_fg has been checked in caller
  // transform except the last 2 parameters: out, dout.
  for (size_t i = 0; i < bprop_fg->parameters().size() - 2; ++i) {
    auto p = bprop_fg->parameters()[i];
    MS_EXCEPTION_IF_NULL(p);

    TraceManager::DebugTrace(std::make_shared<TraceGradFprop>(p->debug_info()));
    auto transf_p = outer->add_parameter();
    TraceManager::EndTrace();

    (void)mng->Replace(p, transf_p);
    transf_args->push_back(transf_p);
  }
}

void KPrim::CheckBprop(const FuncGraphPtr &bprop_fg, const string &prim_to_check) {
  // bprop_fg has been checked in caller
  auto check_bprop = prim::GetPythonOps("check_bprop", "mindspore.ops.functional")->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(check_bprop);
  check_bprop->set_attr("prim_to_check", std::make_shared<StringImm>(prim_to_check));

  std::vector<AnfNodePtr> inputs;
  inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  inputs.insert(inputs.begin() + 1, bprop_fg->parameters().begin(), bprop_fg->parameters().end() - 2);
  AnfNodePtr params = bprop_fg->NewCNode(inputs);

  inputs.clear();
  inputs.push_back(NewValueNode(check_bprop));
  inputs.push_back(bprop_fg->output());
  inputs.push_back(params);
  AnfNodePtr bprop_out = bprop_fg->NewCNode(inputs);
  bprop_fg->set_output(bprop_out);
}

FuncGraphPtr KPrim::KUserDefinedCellBprop(const FuncGraphPtr bprop_fg) {
  MS_EXCEPTION_IF_NULL(bprop_fg);
  auto fprop_fg = bprop_fg->transforms().find("primal")->second.func_graph();
  auto expanded_fg = BpropToK(fprop_fg, bprop_fg);
  if (expanded_fg == nullptr) {
    MS_LOG(EXCEPTION) << "Failed convert " << fprop_fg->ToString()
                      << " Cell bprop function to K expanded func graph. NodeInfo: "
                      << trace::GetDebugInfo(fprop_fg->debug_info());
  }
  return expanded_fg;
}

FuncGraphPtr KPrim::BpropCut(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources) {
  auto prim = GetValueNode<PrimitivePtr>(value_node);
  MS_EXCEPTION_IF_NULL(prim);
  auto &node_users = resources->manager()->node_users();

  auto &users = node_users[value_node];
  auto cnode = std::find_if(users.begin(), users.end(), [&prim](const std::pair<AnfNodePtr, int> &user) -> bool {
    return IsPrimitiveCNode(user.first, prim);
  });
  if (cnode == users.end()) {
    MS_LOG(EXCEPTION) << "Fail to find cnode.";
  }
  auto inputs_num = cnode->first->cast<CNodePtr>()->size() - 1;

  auto func_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> outputs;

  auto bprop_cut = std::make_shared<Primitive>("bprop_cut");
  bprop_cut->set_hook(prim->hook());
  auto cell_id = GetValue<std::string>(prim->GetAttr("cell_id"));
  if (cell_id != "") {
    (void)bprop_cut->AddAttr("cell_hook", MakeValue(true));
    (void)bprop_cut->AddAttr("cell_id", MakeValue(cell_id));
  }

  outputs.push_back(NewValueNode(bprop_cut));
  for (size_t i = 0; i < inputs_num; ++i) {
    auto param = func_graph->add_parameter();
    outputs.push_back(param);
  }
  auto p1 = func_graph->add_parameter();
  auto p2 = func_graph->add_parameter();
  outputs.push_back(p1);
  outputs.push_back(p2);

  func_graph->set_output(func_graph->NewCNode(outputs));
  return func_graph;
}

FuncGraphPtr KPrim::FakeBprop(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources) {
  auto prim = value_node->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(prim);
  auto &node_users = resources->manager()->node_users();

  auto &users = node_users[value_node];
  auto cnode = std::find_if(users.begin(), users.end(), [&prim](const std::pair<AnfNodePtr, int> &user) -> bool {
    return IsPrimitiveCNode(user.first, prim);
  });
  if (cnode == users.end()) {
    MS_LOG(EXCEPTION) << "Fail to find cnode.";
  }
  auto inputs_num = cnode->first->cast<CNodePtr>()->inputs().size() - 1;

  auto func_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> outputs;
  outputs.push_back(NewValueNode(prim::kPrimMakeTuple));

  auto fake_bprop = std::make_shared<Primitive>("fake_bprop");
  (void)fake_bprop->AddAttr("info", MakeValue("Primitive " + prim->name() + "'s bprop not defined."));

  for (size_t i = 0; i < inputs_num; ++i) {
    // Mock params for inputs
    auto param = func_graph->add_parameter();
    // Mock derivatives for each inputs
    outputs.push_back(func_graph->NewCNode({NewValueNode(fake_bprop), param}));
  }
  // mock params for out and dout
  (void)func_graph->add_parameter();
  (void)func_graph->add_parameter();
  func_graph->set_output(func_graph->NewCNode(outputs));
  return func_graph;
}
}  // namespace ad
}  // namespace mindspore
