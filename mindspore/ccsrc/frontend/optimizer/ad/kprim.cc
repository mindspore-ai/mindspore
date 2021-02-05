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
#include "pybind_api/ir/primitive_py.h"
#include "ir/meta_func_graph.h"
#include "ir/func_graph_cloner.h"
#include "ir/manager.h"
#include "pipeline/jit/resource.h"
#include "pipeline/jit/parse/parse.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/composite.h"
#include "utils/symbolic.h"
#include "utils/primitive_utils.h"
#include "utils/ms_context.h"
#include "utils/info.h"
#include "debug/trace.h"

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
  py::function fn;
  if (prim->is_base()) {
    fn = GetBpropFunction(prim->name());
  } else {
    fn = prim->cast<PrimitivePyPtr>()->GetBpropFunction();
    if (py::isinstance<py::none>(fn)) {
      fn = GetBpropFunction(prim->name());
    }
  }
  if (!fn || py::isinstance<py::none>(fn)) {
    MS_LOG(DEBUG) << "Fail to find bprop function for " << prim->name() << ".";
    return nullptr;
  }
  FuncGraphPtr func_graph = parse::ParsePythonCode(fn);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Fail to parse bprop function for " << prim->name() << ".";
    return nullptr;
  }
  auto bprop_flag = GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_BACKPROP);
  if (bprop_flag) {
    func_graph->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);
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

  if (prim->Hash() == prim::kPrimMakeTuple->Hash() && prim->name() == prim::kPrimMakeTuple->name()) {
    MetaFuncGraphPtr meta = std::make_shared<prim::MakeTupleGradient>("make_tuple_gradient");
    bprop_registry_meta_[prim::kPrimMakeTuple] = meta;
    return meta;
  }

  if (prim->Hash() == prim::kPrimMakeList->Hash() && prim->name() == prim::kPrimMakeList->name()) {
    MetaFuncGraphPtr meta = std::make_shared<prim::MakeListGradient>("make_list_gradient");
    bprop_registry_meta_[prim::kPrimMakeList] = meta;
    return meta;
  }

  MS_LOG(EXCEPTION) << "Fail to find bprop function for " << prim->name() << ".";
}

static void AppendMonadOutput(const FuncGraphPtr &bprop_fg, const AnfNodePtr &monad) {
  const auto &output = bprop_fg->output();
  MS_EXCEPTION_IF_NULL(output);
  auto output_cnode = output->cast<CNodePtr>();
  if (output_cnode != nullptr) {
    // If output_cnode has the form like (make_tuple, x, y).
    output_cnode->add_input(monad);
    return;
  }
  // If output is an empty tuple, create a (make_tuple, monad) as the new output.
  auto make_tuple = NewValueNode(prim::kPrimMakeTuple);
  output_cnode = bprop_fg->NewCNode({make_tuple, monad});
  bprop_fg->set_output(output_cnode);
}

// Append U or/and IO monad to output of Bprop funcgraph.
static void AdjustForAutoMonad(const PrimitivePtr &prim, const FuncGraphPtr &bprop_fg) {
  auto effect_info = GetPrimEffectInfo(prim);
  if (effect_info.memory) {
    MS_LOG(DEBUG) << "Append U monad for Bprop FuncGraph of Primitive " << prim->ToString();
    auto u = NewValueNode(kUMonad);
    u->set_abstract(kUMonad->ToAbstract());
    AppendMonadOutput(bprop_fg, u);
  }
  if (effect_info.io) {
    MS_LOG(DEBUG) << "Append IO monad for Bprop FuncGraph of Primitive " << prim->ToString();
    auto io = NewValueNode(kIOMonad);
    io->set_abstract(kIOMonad->ToAbstract());
    AppendMonadOutput(bprop_fg, io);
  }
}

FuncGraphPtr KPrim::KPrimitive(const CNodePtr &cnode, const ValueNodePtr &value_node,
                               const pipeline::ResourceBasePtr &resources) {
  if (!IsValueNode<Primitive>(value_node)) {
    MS_LOG(EXCEPTION) << "Primitive node is not valid.";
  }

  auto prim = GetValueNode<PrimitivePtr>(value_node);
  if (prim->Hash() == prim::kPrimSwitchLayer->Hash() && prim->name() == prim::kPrimSwitchLayer->name()) {
    auto fprop = GetFprop(prim);
    fprop->transforms().emplace("primal", FuncGraphTransform(prim::kPrimSwitchLayer));
    return fprop;
  } else if (prim->Hash() == prim::kPrimMakeTuple->Hash() && prim->name() == prim::kPrimMakeTuple->name()) {
    return nullptr;
  } else if (prim->Hash() == prim::kPrimMakeList->Hash() && prim->name() == prim::kPrimMakeList->name()) {
    return nullptr;
  }

  FuncGraphPtr bprop_fg = nullptr;
  if (prim->Hash() == prim::kPrimHookBackward->Hash() && prim->name() == prim::kPrimHookBackward->name()) {
    if (MsContext::GetInstance()->get_param<int>(MsCtxParam::MS_CTX_EXECUTION_MODE) == kGraphMode) {
      MS_LOG(EXCEPTION) << "HookBackward is not supported in graph mode.";
    }
    bprop_fg = BpropCut(value_node, resources);
  } else {
    auto iter = bprop_registry_.find(prim);
    if (iter != bprop_registry_.end()) {
      bprop_fg = iter->second;
    }

    if (bprop_fg == nullptr) {
      bprop_fg = GetBprop(prim);
      if (bprop_fg != nullptr) {
        // Set bprop_g graph cache
        bprop_registry_[prim] = bprop_fg;
      } else {
        bprop_fg = FakeBprop(value_node, resources);
      }
    }
  }
  AdjustForAutoMonad(prim, bprop_fg);
  auto expanded_fg = BpropToK(prim, bprop_fg, nullptr, cnode);
  if (expanded_fg == nullptr) {
    MS_LOG(EXCEPTION) << "Failed convert " << prim->name()
                      << " prim bprop function to J expanded func graph. NodeInfo: "
                      << trace::GetDebugInfo(bprop_fg->debug_info());
  }

  return expanded_fg;
}

AnfNodePtr KPrim::BuildOutput(const FuncGraphPtr &bprop_fg, const FuncGraphPtr &current_primal_fg) {
  // current_primal_fg may have extra parameters like u_monad, io_monad
  std::vector<AnfNodePtr> extra_args;
  // caller had checked size() - 2 is greater than 0.
  auto bprop_fg_param_size = bprop_fg->parameters().size() - 2;
  if (current_primal_fg != nullptr && bprop_fg_param_size < current_primal_fg->parameters().size()) {
    auto current_primal_fg_param_size = current_primal_fg->parameters().size();
    MS_LOG(DEBUG) << "Current Primal FuncGraph may have extra parameters(U or IO monad) which bprop don't define, so "
                     "Insert it. Extra parameters size: "
                  << current_primal_fg_param_size - bprop_fg_param_size;
    for (auto i = bprop_fg_param_size; i < current_primal_fg_param_size; ++i) {
      const auto &primal_node = current_primal_fg->parameters()[i];
      auto extra_node = bprop_fg->NewCNode({NewValueNode(prim::GetPythonOps("zeros_like")), primal_node});
      extra_args.push_back(extra_node);
      MS_LOG(DEBUG) << "Insert to bprop_fg for node: " << primal_node->DebugString();
    }
  }
  // bprop_fg has been checked in caller
  if (IsPrimitiveCNode(bprop_fg->output(), prim::kPrimMakeTuple)) {
    // Set bprop output as (env, dx, dy, dz, ...)
    auto cbprop = bprop_fg->output()->cast<CNodePtr>();
    auto &inputs = cbprop->inputs();

    std::vector<AnfNodePtr> args;
    args.push_back(NewValueNode(prim::kPrimMakeTuple));
    args.push_back(NewValueNode(newenv));
    (void)args.insert(args.end(), inputs.begin() + 1, inputs.end());
    if (!extra_args.empty()) {
      args.insert(args.end(), extra_args.cbegin(), extra_args.cend());
    }
    return NewCNode(args, bprop_fg);
  }

  // Set bprop output as (env, dx)
  std::string model_name("mindspore.ops.composite.multitype_ops.add_impl");
  std::string python_ops("_tuple_add");
  auto tuple_env = NewCNode({NewValueNode(prim::kPrimMakeTuple), NewValueNode(newenv)}, bprop_fg);
  auto tuple_add_ops = NewValueNode(prim::GetPythonOps(python_ops, model_name));
  if (!extra_args.empty()) {
    extra_args.insert(extra_args.begin(), NewValueNode(prim::kPrimMakeTuple));
    auto extra_tuple = NewCNode(extra_args, bprop_fg);
    auto old_output_extra = NewCNode({tuple_add_ops, bprop_fg->output(), extra_tuple}, bprop_fg);
    return NewCNode({tuple_add_ops, tuple_env, old_output_extra}, bprop_fg);
  }

  return NewCNode({tuple_add_ops, tuple_env, bprop_fg->output()}, bprop_fg);
}

static void TransformNormalArgs(const FuncGraphManagerPtr &mng, const FuncGraphPtr &bprop_fg, const FuncGraphPtr &outer,
                                std::vector<AnfNodePtr> *const transf_args) {
  // bprop_fg has been checked in caller
  // transform except the last 2 parameters: out, dout.
  auto bprop_fg_param_size = bprop_fg->parameters().size() - 2;
  for (size_t i = 0; i < bprop_fg_param_size; ++i) {
    auto p = bprop_fg->parameters()[i];
    MS_EXCEPTION_IF_NULL(p);

    TraceGuard trace_guard(std::make_shared<TraceGradFprop>(p->debug_info()));
    auto transf_p = outer->add_parameter();

    (void)mng->Replace(p, transf_p);
    transf_args->push_back(transf_p);
  }
}
void KPrim::TransformArgsForPrimitive(const FuncGraphManagerPtr &mng, const FuncGraphPtr &bprop_fg,
                                      const PrimitivePtr &primitive, const FuncGraphPtr &outer,
                                      std::vector<AnfNodePtr> *const transf_args) {
  MS_EXCEPTION_IF_NULL(mng);
  TransformNormalArgs(mng, bprop_fg, outer, transf_args);
  // Fprop_fg for Primitive with side effect should append extra U or IO monad parameter.
  auto effect_info = GetPrimEffectInfo(primitive);
  if (effect_info.memory) {
    MS_LOG(DEBUG) << "Append U monad to Fprop FuncGraph for Primitive " << primitive->ToString();
    auto transf_p = outer->add_parameter();
    transf_args->push_back(transf_p);
  }
  if (effect_info.io) {
    MS_LOG(DEBUG) << "Append IO monad to Fprop FuncGraph for Primitive " << primitive->ToString();
    auto transf_p = outer->add_parameter();
    transf_args->push_back(transf_p);
  }
}

template <typename T>
void KPrim::TransformArgsForFuncGraph(const FuncGraphManagerPtr &mng, const FuncGraphPtr &bprop_fg,
                                      const T &current_primal_fg, const FuncGraphPtr &outer,
                                      std::vector<AnfNodePtr> *const transf_args) {
  MS_EXCEPTION_IF_NULL(mng);
  TransformNormalArgs(mng, bprop_fg, outer, transf_args);
  auto bprop_fg_param_size = bprop_fg->parameters().size() - 2;
  // current_primal_fg may have extra parameters after AutoMonad
  const auto &current_primal_fg_params = current_primal_fg->parameters();
  if (bprop_fg_param_size < current_primal_fg_params.size()) {
    for (auto i = bprop_fg_param_size; i < current_primal_fg_params.size(); ++i) {
      auto p = current_primal_fg_params[i];
      MS_EXCEPTION_IF_NULL(p);
      // extra parameters should be Monad.
      if (!HasAbstractMonad(p)) {
        continue;
      }
      MS_LOG(DEBUG) << "Function " << current_primal_fg->ToString()
                    << ", has extra monad parameter: " << p->DebugString()
                    << ", abstract: " << p->abstract()->ToString();

      TraceGuard trace_guard(std::make_shared<TraceGradFprop>(p->debug_info()));
      auto transf_p = outer->add_parameter();

      (void)mng->Replace(p, transf_p);
      transf_args->push_back(transf_p);
    }
  }
  if (transf_args->size() != current_primal_fg_params.size()) {
    MS_EXCEPTION(TypeError) << "Function " << current_primal_fg->ToString()
                            << ", The number of parameter of this primal function is "
                            << current_primal_fg_params.size() << ", but the number of parameters of bprop is "
                            << bprop_fg_param_size;
  }
}

void KPrim::CheckBprop(const FuncGraphPtr &bprop_fg, const string &prim_to_check) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool check_bprop_flag = context->get_param<bool>(MS_CTX_CHECK_BPROP_FLAG);
  // Skip checking if check_bprop not set
  if (!check_bprop_flag) {
    return;
  }

  // bprop_fg has been checked in caller
  auto check_bprop_class = prim::GetPythonOps("CheckBprop", "mindspore.ops.operations.other_ops");
  MS_EXCEPTION_IF_NULL(check_bprop_class);
  auto check_bprop =
    bprop_fg->NewCNode({NewValueNode(check_bprop_class), NewValueNode(std::make_shared<StringImm>(prim_to_check))});

  std::vector<AnfNodePtr> inputs;
  inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  inputs.insert(inputs.begin() + 1, bprop_fg->parameters().begin(), bprop_fg->parameters().end() - 2);
  AnfNodePtr params = bprop_fg->NewCNode(inputs);

  inputs.clear();
  inputs.push_back(check_bprop);
  inputs.push_back(bprop_fg->output());
  inputs.push_back(params);
  AnfNodePtr bprop_out = bprop_fg->NewCNode(inputs);
  bprop_fg->set_output(bprop_out);
}

FuncGraphPtr KPrim::KUserDefinedCellBprop(const FuncGraphPtr &bprop_fg, const FuncGraphPtr &current_primal_fg) {
  MS_EXCEPTION_IF_NULL(bprop_fg);
  // primal_fg is FuncGraph just after convert. Refer ConvertCellObjToFuncGraph.
  // current_primal_fg is specalized and AutoMoaded primal_fg;
  auto primal_fg = bprop_fg->transforms().find("primal")->second.func_graph();
  auto expanded_fg = BpropToK(primal_fg, bprop_fg, current_primal_fg, nullptr);
  if (expanded_fg == nullptr) {
    MS_LOG(EXCEPTION) << "Failed convert " << primal_fg->ToString()
                      << " Cell bprop function to K expanded func graph. NodeInfo: "
                      << trace::GetDebugInfo(primal_fg->debug_info());
  }
  return expanded_fg;
}

FuncGraphPtr KPrim::BpropCut(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources) {
  auto prim = GetValueNode<PrimitivePtr>(value_node);
  MS_EXCEPTION_IF_NULL(prim);
  auto &node_users = resources->manager()->node_users();

  auto &users = node_users[value_node];
  auto cnode = std::find_if(users.begin(), users.end(), [&prim](const std::pair<AnfNodePtr, int64_t> &user) -> bool {
    return IsPrimitiveCNode(user.first, prim);
  });
  if (cnode == users.end()) {
    MS_LOG(EXCEPTION) << "Fail to find cnode.";
  }
  auto inputs_num = cnode->first->cast<CNodePtr>()->size() - 1;

  auto func_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> outputs;

  auto bprop_cut = std::make_shared<PrimitivePy>("bprop_cut", py::object());
  bprop_cut->CopyHookFunction(prim);

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
  auto cnode = std::find_if(users.begin(), users.end(), [&prim](const std::pair<AnfNodePtr, int64_t> &user) -> bool {
    return IsPrimitiveCNode(user.first, prim);
  });
  if (cnode == users.end()) {
    MS_LOG(EXCEPTION) << "Fail to find user for " << prim->ToString();
  }
  auto inputs_num = cnode->first->cast<CNodePtr>()->inputs().size() - 1;
  auto effect_info = GetPrimEffectInfo(prim);
  // Don't add U or IO monad parameters as it will be added later.
  size_t monad_params_size = 0;
  if (effect_info.memory) {
    monad_params_size++;
  }
  if (effect_info.io) {
    monad_params_size++;
  }
  if (inputs_num < monad_params_size) {
    MS_LOG(EXCEPTION) << "Arguments number should be greater than or equal to " << monad_params_size
                      << ", but the CNode is: " << cnode->first->DebugString();
  }
  inputs_num -= monad_params_size;

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
