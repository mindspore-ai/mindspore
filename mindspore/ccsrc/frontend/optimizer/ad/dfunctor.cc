/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/ad/dfunctor.h"

#include <map>
#include <memory>
#include <string>

#include "ir/anf.h"
#include "utils/info.h"
#include "ir/func_graph_cloner.h"
#include "ir/manager.h"
#include "pipeline/jit/resource.h"
#include "pipeline/pynative/pynative_execute.h"
#include "frontend/optimizer/ad/adjoint.h"
#include "frontend/operator/ops.h"
#include "utils/symbolic.h"
#include "utils/ms_context.h"
#include "pipeline/jit/action.h"
#include "pipeline/jit/parse/resolve.h"

namespace mindspore {
namespace ad {
std::unordered_map<FuncGraphPtr, DFunctorPtr> DFunctor::func_graph_to_functor_;
std::unordered_map<AnfNodePtr, AdjointPtr> DFunctor::anfnode_to_adjoin_definition_;
FuncGraphSet DFunctor::scope_;

DFunctor::DFunctor(const FuncGraphPtr &primal_graph, const pipeline::ResourceBasePtr &resources)
    : primal_graph_(primal_graph), resources_(resources), need_cut_(false), is_top_(false) {
  {
    TraceGuard guard(std::make_shared<TraceGradFprop>(primal_graph->debug_info()));
    k_graph_ = std::make_shared<FuncGraph>();
  }
  if (primal_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
    std::string grad_op_name = GetValue<std::string>(primal_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
    k_graph_->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(grad_op_name));
  }
  // To keep switch_layer's inputs from being inlined
  k_graph_->set_switch_layer_input(primal_graph->switch_layer_input());
  k_graph_->set_stage(primal_graph->stage());

  {
    TraceGuard guard(std::make_shared<TraceGradBprop>(primal_graph->debug_info()));
    tape_ = std::make_shared<FuncGraph>();
  }
  tape_->set_stage(primal_graph->stage());
  // Add "_Grad" postfix
  if (primal_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
    std::string grad_op_name = GetValue<std::string>(primal_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) + "_Grad";
    tape_->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(grad_op_name));
  }

  dout_ = tape_->add_parameter();
}

void DFunctor::Init(bool is_top) {
  func_graph_to_functor_[primal_graph_] = shared_from_this();
  is_top_ = is_top;
  if (is_top) {
    scope_ = primal_graph_->scope();
  }
}

void DFunctor::Finish() {
  CallDoutHoleOnTape();
  EliminatePrimalGraph();
}

void DFunctor::Clear() {
  func_graph_to_functor_.clear();
  anfnode_to_adjoin_definition_.clear();
  scope_.clear();
}

void DFunctor::BackPropagateFv(const AnfNodePtr &fv, const AnfNodePtr &din) {
  auto fv_adjoint = anfnode_to_adjoin_.find(fv);
  if (fv_adjoint == anfnode_to_adjoin_.end()) {
    MS_LOG(DEBUG) << "BackPropagateFv can not find adjoint in anfnode_to_adjoin_ fv " << fv->func_graph()->ToString()
                  << " " << fv->ToString() << ".";
    fv_adjoint = anfnode_to_adjoin_indirect_fv_.find(fv);
    if (fv_adjoint == anfnode_to_adjoin_indirect_fv_.end()) {
      MS_LOG(DEBUG) << "BackPropagateFv can not find adjoint in anfnode_to_adjoin_indirect_fv_ fv "
                    << fv->func_graph()->ToString() << " " << fv->ToString() << ".";
      auto parent_adjoint = FindAdjoint(fv);
      AdjointPtr adjoint = nullptr;
      if (parent_adjoint != nullptr) {
        adjoint = std::make_shared<Adjoint>(fv, parent_adjoint->k(), tape_);
      } else {
        MS_LOG(DEBUG) << "BackPropagateFv failed can not find adjoint definition fv, add a k hole "
                      << fv->func_graph()->ToString() << " " << fv->ToString() << ".";
        adjoint = std::make_shared<Adjoint>(fv, nullptr, tape_);
      }
      anfnode_to_adjoin_indirect_fv_[fv] = adjoint;
      fv_adjoint = anfnode_to_adjoin_indirect_fv_.find(fv);
    }
  }
  auto fv_node = fv_adjoint->second->k();
  auto cached_envitem_iter = anfnode_to_envitem_.find(fv_node);
  CNodePtr embed_node, default_val_node;
  if (cached_envitem_iter != anfnode_to_envitem_.end()) {
    embed_node = cached_envitem_iter->second.first;
    default_val_node = cached_envitem_iter->second.second;
  } else {
    embed_node = tape_->NewCNode({NewValueNode(prim::kPrimEmbed), fv_node});
    default_val_node = tape_->NewCNode({NewValueNode(prim::GetPythonOps("zeros_like")), fv_node});
    fv_adjoint->second->RegisterKUser(embed_node, 1);
    fv_adjoint->second->RegisterKUser(default_val_node, 1);
    anfnode_to_envitem_[fv_node] = std::make_pair(embed_node, default_val_node);
  }
  auto dfv = tape_->NewCNode({NewValueNode(prim::kPrimEnvGetItem), din, embed_node, default_val_node});
  MS_LOG(DEBUG) << "BackPropagateFv find adjoint in anfnode_to_adjoin_ or anfnode_to_adjoin_indirect_fv_ fv "
                << fv->func_graph()->ToString() << " " << fv->ToString() << ".";
  MS_LOG(DEBUG) << "BackPropagateFv get item from " << din->ToString() << " key " << embed_node->ToString() << ".";
  fv_adjoint->second->AccumulateDout(dfv);
}

void DFunctor::BackPropagateSwitchLayer(const CNodePtr &cnode_morph, const CNodePtr &env) {
  // Take switch_layer as a set of candidate functions.
  auto input = cnode_morph->input(2);
  if (!IsPrimitiveCNode(input, prim::kPrimMakeTuple)) {
    MS_LOG(EXCEPTION) << "The 2th input of switch_layer expect a tuple of graphs, but got " << input->ToString() << ".";
  }
  std::unordered_map<AnfNodePtr, FuncGraphPtr> node_to_fg;
  auto tuple_graphs = input->cast<CNodePtr>();
  for (size_t i = 1; i < tuple_graphs->size(); ++i) {
    auto graph = tuple_graphs->input(i);
    if (!IsValueNode<FuncGraph>(graph)) {
      MS_LOG(EXCEPTION) << "The 2th input of switch_layer expect a tuple of graphs, but got " << graph->ToString()
                        << " as the " << i << "th element.";
    }
    auto func_graph = GetValueNode<FuncGraphPtr>(graph);
    auto functor = func_graph_to_functor_.find(func_graph);
    if (functor == func_graph_to_functor_.end()) {
      MS_LOG(EXCEPTION) << "BackPropagateSwitchLayer failed functor for subgraph does not exist input[" << i << "] "
                        << func_graph->ToString() << ".";
    }
    // Consider direct and indirect fvs.
    for (auto fv : func_graph->free_variables_nodes()) {
      if (node_to_fg.find(fv) != node_to_fg.end()) {
        continue;
      }
      node_to_fg[fv] = func_graph;
      BackPropagateFv(fv, env);
    }
    for (auto indirect_fv : functor->second->anfnode_to_adjoin_indirect_fv_) {
      MS_LOG(DEBUG) << "BackPropagateSwitchLayer backprop indirect fv " << func_graph->ToString() << " "
                    << indirect_fv.first->ToString() << ".";
      if (node_to_fg.find(indirect_fv.first) != node_to_fg.end()) {
        continue;
      }
      node_to_fg[indirect_fv.first] = func_graph;
      BackPropagateFv(indirect_fv.first, env);
    }
  }
}

static bool HasSideEffectBackProp(const CNodePtr &cnode) {
  if (IsPrimitiveCNode(cnode)) {
    const auto &prim = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    auto bprop_flag = GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_BACKPROP);
    return bprop_flag;
  }
  return false;
}

void DFunctor::BackPropagate(const CNodePtr &cnode_morph, const CNodePtr &k_app, const AdjointPtr &node_adjoint) {
  auto bprop =
    k_graph_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), k_app, NewValueNode(static_cast<int64_t>(1))});
  // Call with delimited continuation dout.
  CNodePtr bprop_app;
  if (HasSideEffectBackProp(cnode_morph)) {
    // as MapMorphism is called recursively, so the order of bprop_app should reversed as visited order.
    bprop_app = tape_->NewCNodeInFront({bprop, node_adjoint->dout()});
    tape_->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);
  } else {
    bprop_app = tape_->NewCNode({bprop, node_adjoint->dout()});
  }
  node_adjoint->RegisterDoutUser(bprop_app, 1);
  // Special case for switch_layer
  if (IsPrimitiveCNode(cnode_morph, prim::kPrimSwitchLayer)) {
    auto din =
      tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), bprop_app, NewValueNode(static_cast<int64_t>(0))});
    BackPropagateSwitchLayer(cnode_morph, din);
    return;
  }
  for (size_t i = 0; i < cnode_morph->size(); i++) {
    auto din = tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), bprop_app, NewValueNode(SizeToLong(i))});
    auto input = cnode_morph->input(i);
    // Backprop sens wrt fvs.
    if (IsValueNode<FuncGraph>(input)) {
      auto func_graph = GetValueNode<FuncGraphPtr>(input);
      auto functor = func_graph_to_functor_.find(func_graph);
      if (functor == func_graph_to_functor_.end()) {
        MS_LOG(EXCEPTION) << "BackPropagate failed functor for subgraph does not exist input[" << i << "] "
                          << func_graph->ToString() << ".";
      }
      // Consider direct and indirect fvs.
      for (auto fv : func_graph->free_variables_nodes()) {
        BackPropagateFv(fv, din);
      }
      for (auto indirect_fv : functor->second->anfnode_to_adjoin_indirect_fv_) {
        MS_LOG(DEBUG) << "BackPropagate backprop indirect fv " << func_graph->ToString() << " "
                      << indirect_fv.first->ToString() << ".";
        BackPropagateFv(indirect_fv.first, din);
      }
      continue;
    }
    // Backprop sens wrt inputs.
    auto input_adjoint = anfnode_to_adjoin_.find(input);
    if (input_adjoint == anfnode_to_adjoin_.end()) {
      MS_LOG(EXCEPTION) << "BackPropagate adjoint does not exist input[" << i << "] " << input->ToString() << ".";
    }
    input_adjoint->second->AccumulateDout(din);
  }
}

// Map a morphism.
AdjointPtr DFunctor::MapMorphism(const AnfNodePtr &morph) {
  MS_LOG(DEBUG) << "start MapMorphism:" << morph->DebugString(4);
  // MapMorphism All type except CNode should already be mapped by MapObject.
  if (!morph->isa<CNode>()) {
    return nullptr;
  }
  // for free variable, which may be handled in MapValueObject, just return it
  auto node_adjoint_found = anfnode_to_adjoin_.find(morph);
  if (node_adjoint_found != anfnode_to_adjoin_.end()) {
    return node_adjoint_found->second;
  }
  ScopeGuard scope_guard(morph->scope());
  auto cnode_morph = morph->cast<CNodePtr>();

  std::vector<AnfNodePtr> inputs;
  std::vector<AdjointPtr> param_adjoints;
  for (size_t i = 0; i < cnode_morph->size(); i++) {
    auto node = cnode_morph->input(i);
    AdjointPtr node_adjoint = nullptr;
    auto node_adjoint_iter = anfnode_to_adjoin_.find(node);
    if (node_adjoint_iter != anfnode_to_adjoin_.end()) {
      node_adjoint = node_adjoint_iter->second;
    } else {
      // Input might be a CNode that needs to be handled previously.
      node_adjoint = MapMorphism(node);
    }
    MS_EXCEPTION_IF_NULL(node_adjoint);
    AnfNodePtr k = node_adjoint->k();
    if (k == nullptr) {
      MS_LOG(EXCEPTION) << "MapMorphism adjoint node does not exist, input[" << i << "] " << node->ToString() << ".";
    }
    inputs.push_back(k);
    param_adjoints.push_back(node_adjoint);
  }
  CNodePtr k_app = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceGradFpropApp>(cnode_morph->debug_info()));
    k_app = k_graph_->NewCNode(inputs);
  }
  ReplaceEquivdout(k_app, cnode_morph);
  cnode_morph->clear_inputs_value();
  cnode_morph->set_forward(nullptr, "");
  for (size_t i = 0; i < param_adjoints.size(); ++i) {
    param_adjoints[i]->RegisterKUser(k_app, i);
  }

  // Do forward computation
  auto foward_app =
    k_graph_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), k_app, NewValueNode(static_cast<int64_t>(0))});
  // K:: cnode -> forward_app
  auto node_adjoint = std::make_shared<Adjoint>(morph, foward_app, tape_);
  UpdateAdjoint(node_adjoint);
  anfnode_to_adjoin_[morph] = node_adjoint;
  if (cnode_morph->stop_gradient()) {
    MS_LOG(DEBUG) << "MapMorphism node " << morph->ToString() << " is stopped.";
    return node_adjoint;
  }

  // Do sens backpropagation
  BackPropagate(cnode_morph, k_app, node_adjoint);
  MS_LOG(DEBUG) << "MapMorphism node " << morph->DebugString(4) << ".";
  return node_adjoint;
}

ValuePtr DFunctor::GenNewTensorInner(const ValuePtr &value) {
  std::vector<ValuePtr> value_list;
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    auto new_tensor = std::make_shared<tensor::Tensor>(*tensor);
    new_tensor->set_device_address(nullptr);
    return new_tensor;
  }
  if (value->isa<ValueTuple>()) {
    auto tuple = value->cast<ValueTuplePtr>();
    for (size_t i = 0; i < tuple->size(); i++) {
      value_list.push_back(GenNewTensorInner((*tuple)[i]));
    }
    return std::make_shared<ValueTuple>(value_list);
  }
  return value;
}

ValuePtr DFunctor::GenNewTensor(const FuncGraphManagerPtr &mng, const AnfNodePtr &node, const ValuePtr &value,
                                bool need_replace_forward) {
  ValuePtr out = value;
  auto ref_size = mng->node_users()[node].size();
  if (ref_size < 2) {
    if (need_replace_forward) {
      out = GenNewTensorInner(value);
    } else {
      auto tensor = value->cast<tensor::TensorPtr>();
      tensor->set_device_address(nullptr);
      return tensor;
    }
  }
  return out;
}

void DFunctor::ReplaceEquivdout(const CNodePtr &cnode, const CNodePtr &cnode_morph) {
  auto forward = cnode_morph->forward().first;
  if (forward == nullptr) {
    return;
  }
  auto &input = cnode->input(0);
  if (!IsValueNode<FuncGraph>(input)) {
    return;
  }
  auto fg = GetValueNode<FuncGraphPtr>(input);
  // {prim::maketuple, forward_output, bprop_graph}
  auto output = fg->output();
  if (!output->isa<CNode>()) {
    return;
  }
  auto cnode_output = output->cast<CNodePtr>();
  auto &cnode_input = cnode_output->input(1);
  if (!cnode_input->isa<CNode>()) {
    return;
  }
  auto &input_fg = cnode_output->input(2);
  if (!IsValueNode<FuncGraph>(input_fg)) {
    return;
  }
  // replace forward output with value node
  auto equivdout = cnode_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(equivdout);
  auto func_graph = GetValueNode<FuncGraphPtr>(input_fg);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = Manage({fg, func_graph}, false);
  auto need_replace_forward = pynative::PynativeExecutor::GetInstance()->need_replace_forward();
  auto forward_value = GenNewTensor(manager, equivdout, forward, need_replace_forward);
  if (!need_replace_forward) {
    cnode_morph->clear_inputs_value();
    MS_LOG(DEBUG) << "No need replace forward result";
    return;
  }
  MS_LOG(DEBUG) << "Replace: " << equivdout->ToString() << " with " << forward;
  auto value_node = NewValueNode(forward_value);
  value_node->set_has_new_value(true);
  manager->Replace(equivdout, value_node);
  // replace input object with value node
  auto paras = fg->parameters();
  auto inputs_value = cnode_morph->inputs_value();
  if (inputs_value.empty()) {
    return;
  }
  if (inputs_value.size() > paras.size()) {
    MS_LOG(EXCEPTION) << "Parameter size:" << paras.size() << " but inputs size:" << inputs_value.size();
  }
  for (size_t i = 0; i < inputs_value.size(); i++) {
    auto para_ref_size = manager->node_users()[paras[i]].size();
    auto input_value = inputs_value[i];
    if (para_ref_size > 0 && input_value.first != nullptr) {
      MS_LOG(DEBUG) << "Replace: " << paras[i]->ToString() << " with " << input_value.first;
      auto input_value_node = NewValueNode(input_value.first);
      input_value_node->set_has_new_value(true);
      input_value_node->set_used_graph_count(para_ref_size);
      manager->Replace(paras[i], input_value_node);
    }
  }
  MS_LOG(DEBUG) << "Start opt node" << fg->output()->DebugString(4);
  auto res = std::make_shared<pipeline::Resource>();
  res->set_manager(manager);
  res->set_func_graph(fg);
  PynativeElimOpt(res);
  auto out = fg->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(out);
  auto c_input = out->input(1);
  MS_EXCEPTION_IF_NULL(c_input);
  if (!c_input->isa<ValueNode>()) {
    return;
  }
  auto out_node = c_input->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(out_node);
  out_node->set_value(GenNewTensor(manager, out_node, out_node->value(), need_replace_forward));
}

bool DFunctor::IsFreeMorphism(const AnfNodePtr &node) {
  // Do not care about non-CNode
  if (!node->isa<CNode>()) {
    return false;
  }
  // Do not care about kPrimReturn
  if (IsPrimitiveCNode(node, prim::kPrimReturn)) {
    return false;
  }
  auto &users = primal_graph_->manager()->node_users()[node];
  // Do not care about isolated morphisms
  if (users.empty()) {
    return false;
  }
  // Not free if it's used by some node in primal_graph
  bool nonfree = std::any_of(std::begin(users), std::end(users), [&](const auto &kv) {
    auto &user = kv.first;
    return user->func_graph() == primal_graph_;
  });
  return !nonfree;
}

void DFunctor::MapFreeMorphism() {
  // Handle cnode not attached to output, that might be referred in other functions.
  for (auto &node : primal_graph_->nodes()) {
    if (!IsFreeMorphism(node)) {
      continue;
    }
    MS_LOG(DEBUG) << "MapFreeMorphism map nonoutput cnode after MapMorphism " << node->ToString() << ".";
    (void)MapMorphism(node);
  }
}

AnfNodePtr DFunctor::AttachFvDoutToTape(const AnfNodePtr &grad_fv) {
  AnfNodePtr new_grad_fv = grad_fv;
  // Add grads wrt fv.
  const auto &free_variables_nodes = primal_graph_->free_variables_nodes();
  for (auto &fv : free_variables_nodes) {
    auto fv_adjoint = anfnode_to_adjoin_.find(fv);
    if (fv_adjoint == anfnode_to_adjoin_.end()) {
      MS_LOG(EXCEPTION) << "AttachFvDoutToTape fv adjoint does not exist " << fv->ToString() << ".";
    }
    auto node = tape_->NewCNode({NewValueNode(prim::kPrimEmbed), fv_adjoint->second->k()});
    fv_adjoint->second->RegisterKUser(node, 1);
    auto sens = fv_adjoint->second->dout();
    new_grad_fv = tape_->NewCNode({
      NewValueNode(prim::kPrimEnvSetItem),
      new_grad_fv,
      node,
      sens,
    });
    fv_adjoint->second->RegisterDoutUser(new_grad_fv->cast<CNodePtr>(), 3);
    MS_LOG(DEBUG) << "AttachFvDoutToTape add fv sens " << sens->ToString() << " to " << new_grad_fv->ToString() << " "
                  << fv->ToString() << " " << primal_graph_->ToString() << ".";
  }
  return new_grad_fv;
}

AnfNodePtr DFunctor::AttachIndirectFvDoutToTape(const AnfNodePtr &grad_fv) {
  AnfNodePtr new_grad_fv = grad_fv;
  // Add indirect fv bprop.
  for (auto &fv_adjoint : anfnode_to_adjoin_indirect_fv_) {
    MS_LOG(DEBUG) << "AttachIndirectFvDoutToTape backprop indirect fv " << fv_adjoint.first->ToString() << " "
                  << primal_graph_->ToString() << ".";
    auto node = tape_->NewCNode({NewValueNode(prim::kPrimEmbed), fv_adjoint.second->k()});
    fv_adjoint.second->RegisterKUser(node, 1);
    auto sens = fv_adjoint.second->dout();
    new_grad_fv = tape_->NewCNode({
      NewValueNode(prim::kPrimEnvSetItem),
      new_grad_fv,
      node,
      sens,
    });
    fv_adjoint.second->RegisterDoutUser(new_grad_fv->cast<CNodePtr>(), 3);
    MS_LOG(DEBUG) << "AttachIndirectFvDoutToTape add indirect fv sens " << sens->ToString() << " to "
                  << new_grad_fv->ToString() << ".";
  }
  return new_grad_fv;
}

void DFunctor::MapMorphism() {
  // Set stop_gradient before MapMorphism.
  BroadCastStopFlag();

  // Handle free morphism before output, because in some case, free morphism might depend on output's fv tangent
  MapFreeMorphism();
  // Handle morphism from output.
  (void)MapMorphism(primal_graph_->output());

  // Construct K for primal_graph_
  auto output_adjoint = anfnode_to_adjoin_.find(primal_graph_->output());
  // Attach dout_ parameter to output_adjoint.
  output_adjoint->second->AccumulateDout(dout_);

  // Set output for tape closure.
  auto grad_fv = AttachIndirectFvDoutToTape(AttachFvDoutToTape(NewValueNode(newenv)));

  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimMakeTuple), grad_fv};
  // Add grads wrt inputs.
  std::vector<AdjointPtr> param_adjoints;
  for (auto &param : primal_graph_->parameters()) {
    auto param_adjoint = anfnode_to_adjoin_.find(param);
    inputs.push_back(param_adjoint->second->dout());
    param_adjoints.push_back(param_adjoint->second);
  }
  auto tape_output = tape_->NewCNode(inputs);
  for (size_t i = 0; i < param_adjoints.size(); ++i) {
    param_adjoints[i]->RegisterDoutUser(tape_output, i + 2);
  }
  tape_->set_output(tape_output);
  // Set output for k_graph_, K:: cnode->forward_app.
  auto forward_app = output_adjoint->second->k();
  auto output = k_graph_->NewCNode({NewValueNode(prim::kPrimMakeTuple), forward_app, NewValueNode(tape_)});
  output_adjoint->second->RegisterKUser(output, 1);
  k_graph_->set_output(output);
  (void)primal_graph_->transforms().insert(std::make_pair("grad", FuncGraphTransform(k_graph_)));
  (void)k_graph_->transforms().insert(std::make_pair("primal", FuncGraphTransform(primal_graph_)));
}

FuncGraphPtr DFunctor::KUserDefined(const FuncGraphPtr &primal) {
  // K user defined cell bprop.
  auto bprop = primal->transforms().find("bprop");
  if (bprop != primal->transforms().end()) {
    FuncGraphPtr bprop_graph = bprop->second.func_graph();
    resources_->manager()->AddFuncGraph(bprop_graph);

    if (!bprop_graph->free_variables_nodes().empty() || !primal->free_variables_nodes().empty()) {
      MS_LOG(EXCEPTION) << "User defined Cell bprop " << primal->ToString() << " in scope "
                        << primal->output()->scope()->name() << " does not support Parameter data type.";
    }
    bprop_graph->set_flag(mindspore::kFuncGraphFlagBackPropEntry, true);
    bprop_graph->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);

    auto fg = g_k_prims.KUserDefinedCellBprop(bprop_graph, primal);
    if (fg == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to expand user defined Cell bprop " << primal->ToString() << " in scope "
                        << primal->output()->scope()->name() << ".";
    }

    // Cache the grad func
    (void)primal->transforms().insert(std::make_pair("grad", FuncGraphTransform(fg)));
    (void)fg->transforms().insert(std::make_pair("primal", FuncGraphTransform(primal)));
    // Reset defer_inline to enable successive inlining
    primal->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, false);

    auto functor = std::make_shared<DFunctor>(primal, resources_);
    functor->Init();
    functor->k_graph_ = fg;

    return fg;
  }
  return nullptr;
}

// Construct representation graph for {CNode, Index} of Primitive.
AnfNodePtr DFunctor::MapPrimitiveToK(const CNodePtr &primitive_user, size_t index) {
  auto primal = primitive_user->input(index);
  if (!IsValueNode<Primitive>(primal)) {
    MS_LOG(EXCEPTION) << "Primal graph \"" << primal->ToString() << "\" is not a ValueNode of Primitive.";
  }
  ScopeGuard scope_guard(primal->scope());
  // Map Primitive to K
  auto value_node = primal->cast<ValueNodePtr>();
  auto prim = GetValueNode<PrimitivePtr>(value_node);
  if ((prim->Hash() == prim::kPrimStopGradient->Hash() && prim->name() == prim::kPrimStopGradient->name()) ||
      (prim->Hash() == prim::kPrimUpdateState->Hash() && prim->name() == prim::kPrimUpdateState->name())) {
    MS_LOG(DEBUG) << "Should stop gradient for " << prim->ToString();
    need_cut_ = true;
  }
  auto k_prim = g_k_prims.KPrimitive(primitive_user, value_node, resources_);
  if (k_prim != nullptr) {
    return NewValueNode(k_prim);
  }
  // When failed to find k_prim, try k_meta.
  auto k_meta = g_k_prims.KMetaFuncGraph(prim);
  if (k_meta != nullptr) {
    return NewValueNode(k_meta);
  }
  MS_LOG(EXCEPTION) << "Fail to map Primitive of \"" << primal->ToString() << "\" to K.";
}

// Construct representation graph for ValueNode of FuncGraph.
AnfNodePtr DFunctor::MapFuncGraphToK(const AnfNodePtr &primal) {
  if (!IsValueNode<FuncGraph>(primal)) {
    MS_LOG(EXCEPTION) << "Primal graph \"" << primal->ToString() << "\" is not a ValueNode of FuncGraph.";
  }
  ScopeGuard scope_guard(primal->scope());
  // Map func graph to K
  auto func_graph = GetValueNode<FuncGraphPtr>(primal);
  auto f = func_graph_to_functor_.find(func_graph);
  if (f != func_graph_to_functor_.end()) {
    MS_LOG(DEBUG) << "K graph functor already exist " << func_graph->ToString() << ".";
    return NewValueNode(f->second->k_graph_);
  }
  auto k_user_defined = KUserDefined(func_graph);
  if (k_user_defined != nullptr) {
    MS_LOG(DEBUG) << "K graph functor user defined bprop " << func_graph->ToString() << ".";
    return NewValueNode(k_user_defined);
  }
  auto functor = std::make_shared<DFunctor>(func_graph, resources_);
  functor->Init();
  functor->MapObject();
  functor->MapMorphism();

  MS_LOG(DEBUG) << "Map \"" << func_graph->ToString() << "\" to \"" << functor->k_graph_->ToString() << "\"";
  return NewValueNode(functor->k_graph_);
}

// Construct for ValueNode of Parameter.
AnfNodePtr DFunctor::MapParameterToK(const AnfNodePtr &primal) {
  if (!primal->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "Primal graph \"" << primal->ToString() << "\" is not a ValueNode of Parameter.";
  }
  ScopeGuard scope_guard(primal->scope());
  // Map Parameter to K
  TraceGuard trace_guard(std::make_shared<TraceGradFprop>(primal->debug_info()));
  auto ret = k_graph_->add_parameter();
  return ret;
}

bool DFunctor::IsInScope(const AnfNodePtr &node) {
  return std::any_of(scope_.begin(), scope_.end(),
                     [&](const FuncGraphPtr &graph) { return node->func_graph() == graph; });
}

void DFunctor::MapFvObject() {
  // Map free variable.
  const auto &free_variables_nodes = primal_graph_->free_variables_nodes();
  for (auto &node : free_variables_nodes) {
    ScopeGuard scope_guard(node->scope());
    MS_LOG(DEBUG) << "MapFvObject free variable " << node->ToString() << ".";
    // Find fv's K from parent.
    AdjointPtr adjoint = nullptr;
    auto parent_adjoint = FindAdjoint(node);
    if (parent_adjoint != nullptr) {
      adjoint = std::make_shared<Adjoint>(node, parent_adjoint->k(), tape_);
    } else {
      if (is_top_ || node->isa<Parameter>()) {
        // Out of ad scope, add adjoint for free variables.
        adjoint = std::make_shared<Adjoint>(node, node, tape_);
        UpdateAdjoint(adjoint);
      } else {
        MS_LOG(DEBUG) << "MapFvObject fail to find parent adjoint for nontop fv " << node->ToString() << ".";
        adjoint = std::make_shared<Adjoint>(node, nullptr, tape_);
      }
    }
    if (adjoint == nullptr) {
      MS_LOG(EXCEPTION) << "MapFvObject failed for free variable " << node->ToString() << ".";
    }
    anfnode_to_adjoin_[node] = adjoint;
  }
}

void DFunctor::MapParamObject() {
  // Map parameter.
  for (auto &p : primal_graph_->parameters()) {
    ScopeGuard scope_guard(p->scope());
    MS_LOG(DEBUG) << "MapParamObject parameter " << p->ToString() << ".";
    auto adjoint = std::make_shared<Adjoint>(p, MapParameterToK(p), tape_);
    UpdateAdjoint(adjoint);
    anfnode_to_adjoin_[p] = adjoint;
  }
}

void DFunctor::MapValueObject() {
  // Map ValueNode.
  auto manager = resources_->manager();
  auto &value_nodes = primal_graph_->value_nodes();
  for (const auto &value_pair : value_nodes) {
    auto node = value_pair.first;
    auto parent_adjoint = FindAdjoint(node);
    if (parent_adjoint != nullptr) {
      auto adjoint = std::make_shared<Adjoint>(node, parent_adjoint->k(), tape_);
      anfnode_to_adjoin_[node] = adjoint;
      continue;
    }

    AdjointPtr adjoint = nullptr;
    if (IsValueNode<Primitive>(node)) {  // Primitive.
      if (GetValueNode<PrimitivePtr>(node) == prim::kPrimReturn) {
        continue;
      }
      MS_LOG(DEBUG) << "Map Primitive node " << node->DebugString() << ".";
      auto &users = manager->node_users()[node];
      if (users.size() == 0) {
        MS_LOG(ERROR) << "\"" << node->DebugString() << "\" has no user.";
        continue;
      } else if (users.size() > 1) {
        MS_LOG(DEBUG) << "\"" << node->DebugString() << "\" supposed to be used once, but users size: " << users.size();
      }
      auto cnode = users.begin()->first->cast<CNodePtr>();  // We just use the first user.
      auto index = users.begin()->second;
      adjoint = std::make_shared<Adjoint>(node, MapPrimitiveToK(cnode, index), tape_);
    } else if (IsValueNode<FuncGraph>(node)) {  // FuncGraph
      MS_LOG(DEBUG) << "Map FuncGraph node " << node->DebugString() << ".";
      adjoint = std::make_shared<Adjoint>(node, MapFuncGraphToK(node), tape_);
    } else if (node->isa<Parameter>()) {  // Parameter, hardly reach here.
      MS_LOG(DEBUG) << "Map Parameter node " << node->DebugString() << ".";
      adjoint = std::make_shared<Adjoint>(node, MapParameterToK(node), tape_);
    } else {
      adjoint = std::make_shared<Adjoint>(node, node, tape_);
    }
    UpdateAdjoint(adjoint);
    anfnode_to_adjoin_[node] = adjoint;
  }
}

// Skip morphism.
void DFunctor::MapObject() {
  // The order does not matter
  MapFvObject();
  MapParamObject();
  MapValueObject();
}

void DFunctor::UpdateAdjoint(const AdjointPtr &adjoint_definition) {
  auto primal = adjoint_definition->primal();
  if (anfnode_to_adjoin_definition_.find(primal) != anfnode_to_adjoin_definition_.end()) {
    MS_LOG(EXCEPTION) << "UpdateAdjoint adjoint definition already exists " << primal_graph_->ToString() << " "
                      << primal->ToString() << ".";
  }
  anfnode_to_adjoin_definition_[primal] = adjoint_definition;
  // Update k hole for primal.
  for (auto &f : func_graph_to_functor_) {
    auto adjoint = f.second->anfnode_to_adjoin_.find(primal);
    if (adjoint != f.second->anfnode_to_adjoin_.end()) {
      adjoint->second->UpdateK(adjoint_definition->k());
    }
    adjoint = f.second->anfnode_to_adjoin_indirect_fv_.find(primal);
    if (adjoint != f.second->anfnode_to_adjoin_indirect_fv_.end()) {
      adjoint->second->UpdateK(adjoint_definition->k());
    }
  }
}

AdjointPtr DFunctor::FindAdjoint(const AnfNodePtr &primal) {
  auto adjoint = anfnode_to_adjoin_definition_.find(primal);
  if (adjoint != anfnode_to_adjoin_definition_.end()) {
    MS_LOG(DEBUG) << "FindAdjoint found adjoint definition for free variable " << primal->ToString() << ".";
    return adjoint->second;
  }
  MS_LOG(DEBUG) << "FindAdjoint adjoint definition for free variable not defined yet " << primal->ToString() << ".";
  return nullptr;
}

void DFunctor::CallDoutHoleOnTape() {
  if (!is_top_) {
    return;
  }

  // Call dout hole of all adjoint.
  for (auto &f : func_graph_to_functor_) {
    for (auto &adjoint : f.second->anfnode_to_adjoin_) {
      adjoint.second->CallDoutHole();
    }
    for (auto &adjoint : f.second->anfnode_to_adjoin_indirect_fv_) {
      adjoint.second->CallDoutHole();
    }
  }
}

FuncGraphPtr DFunctor::k_graph() { return k_graph_; }

FuncGraphPtr DFunctor::tape() { return tape_; }

void DFunctor::BroadCastStopFlag() {
  // As stop set expanding, all directly or indirectly stopped CNode will be cut off
  while (need_cut_) {
    need_cut_ = false;
    for (auto &node : primal_graph_->nodes()) {
      if (node->isa<CNode>()) {
        auto cnode = node->cast<CNodePtr>();
        if (!cnode->stop_gradient()) {
          // Cut off the cnode only when it's not referred any more
          if (IsPrimitiveCNode(cnode, prim::kPrimStopGradient) || IsPrimitiveCNode(cnode, prim::kPrimUpdateState) ||
              AllReferencesStopped(cnode)) {
            MS_LOG(DEBUG) << "Set stop gradient flag for " << cnode->ToString() << ".";
            cnode->set_stop_gradient(true);
            // The stop set changed, more cut required
            need_cut_ = true;
          }
        }
      }
    }
  }
}

bool DFunctor::AllReferencesStopped(const CNodePtr &node) {
  auto &users = primal_graph_->manager()->node_users()[node];
  // Only care about stop_gradient caused cutting
  if (users.empty()) {
    return false;
  }
  for (auto &kv : users) {
    auto &user = kv.first;
    if (!user->isa<CNode>() || !user->cast<CNodePtr>()->stop_gradient()) {
      return false;
    }
  }
  return true;
}

CNodePtr GetJUser(const NodeUsersMap &node_user_map, const CNodePtr &cnode, int index) {
  auto it = node_user_map.find(cnode);
  if (it == node_user_map.end()) {
    MS_LOG(EXCEPTION) << "J CNode not used {" << cnode->DebugString(2) << "/" << index << "}";
  }
  auto &j_users = it->second;
  auto size = j_users.size();
  if (size != 1) {
    MS_LOG(EXCEPTION) << "Wrong J CNode use size " << size << " {" << cnode->DebugString(2) << "/" << index << "}";
  }
  return j_users.begin()->first->cast<CNodePtr>();
}

static std::vector<std::pair<CNodePtr, CNodePtr>> FindPrimalJPair(const FuncGraphManagerPtr &manager,
                                                                  const FuncGraphPtr &primal_graph) {
  std::vector<std::pair<CNodePtr, CNodePtr>> primal_j_pair;
  std::map<FuncGraphPtr, std::pair<CNodePtr, int>> primal_users_map;
  const auto &node_user_map = manager->node_users();
  // Search primal graph user cnodes.
  for (auto &entry : primal_graph->func_graph_cnodes_index()) {
    auto cnode = entry.first->first->cast<CNodePtr>();
    auto index = entry.first->second;
    if (index == 0) {
      // To find real calling.
      auto fg = cnode->func_graph();
      MS_EXCEPTION_IF_NULL(fg);
      auto iter = primal_users_map.find(fg);
      if (iter != primal_users_map.end()) {
        iter->second.second++;
        continue;
      }
      primal_users_map[fg] = std::make_pair(cnode, 1);
    } else if (IsPrimitive(cnode->inputs().at(0), prim::kPrimJ)) {
      // To find J user.
      auto j_user = GetJUser(node_user_map, cnode, index);
      primal_j_pair.push_back({nullptr, j_user});
    }
  }

  for (auto &[primal_user, j_user] : primal_j_pair) {
    // Check if J operation has relevant primal call in the same graph
    auto graph = j_user->func_graph();
    auto iter = primal_users_map.find(graph);
    if (iter == primal_users_map.end()) {
      MS_LOG(WARNING) << "J operation has no relevant primal call in the same graph. Func graph: " << graph->ToString()
                      << ", J user: " << j_user->DebugString();
      continue;
    }

    auto primal_count_pair = iter->second;
    // Check input size.
    auto primal = primal_count_pair.first;
    if (primal->size() != j_user->size()) {
      MS_LOG(WARNING) << "Input size incorrect, the input size of primal " << primal->DebugString() << " is "
                      << primal->size() << ", and J user " << j_user->DebugString() << " is " << j_user->size();
      continue;
    }
    if (primal_count_pair.second != 1) {
      MS_LOG(WARNING) << "It is recommended to call the forward network only once.";
      MS_LOG(INFO) << "There is more than one primal call for J operation in the same graph. Func graph: "
                   << graph->ToString() << ", primal call: " << primal->DebugString()
                   << ", J user: " << j_user->DebugString() << ", trace: " << trace::DumpSourceLines(primal);
      continue;
    }

    primal_user = primal;
    MS_LOG(DEBUG) << "Primal_J pair is found, where primal is: " << primal->DebugString()
                  << " and J user is: " << j_user->DebugString();
  }
  return primal_j_pair;
}

static void RemovePrimalUpdateStates(const FuncGraphManagerPtr &manager, const CNodePtr &primal_call) {
  auto &node_users = manager->node_users();
  auto iter = node_users.find(primal_call);
  if (iter == node_users.end()) {
    // Skip if user of primal_call not found.
    return;
  }
  // Find UpdateState nodes after the primal call.
  std::vector<CNodePtr> update_states;
  for (auto &user : iter->second) {
    auto &user_node = user.first;
    if (IsPrimitiveCNode(user_node, prim::kPrimUpdateState)) {
      update_states.emplace_back(user_node->cast<CNodePtr>());
    }
  }
  // Remove UpdateStates by replace them with their monad input.
  for (auto &update_state : update_states) {
    auto &input_monad = update_state->inputs().at(1);
    manager->Replace(update_state, input_monad);
  }
}

static bool CopyMonadArguments(const CNodePtr &primal_user, const CNodePtr &j_user) {
  auto &primal_inputs = primal_user->inputs();
  auto &j_user_inputs = j_user->inputs();
  bool has_monad = false;
  for (size_t i = 1; i < primal_inputs.size(); ++i) {
    auto &input = primal_inputs.at(i);
    if (HasAbstractMonad(input)) {
      // Copy monad input from primal to j_user.
      j_user->set_input(i, input);
      has_monad = true;
    } else if (input != j_user_inputs.at(i)) {
      // Skip if there are different non-monad inputs.
      return false;
    }
  }
  return has_monad;
}

//
// To replace the primal graph with k graph.
// Convert:
//   x = primal(args, u0)
//   u1 = update_state(u0, x)
//   ...
//   tuple = K(args, u1)
//   u2 = update_state(u1, tuple)
//   ...
// To:
//   tuple = K(args, u0)
//   x = get_item(tuple, 0)
//   ...
//   tuple = K(args, u0)
//   u2 = update_state(u0, tuple)
//   ...
//
void DFunctor::EliminatePrimalGraph() {
  // Find primal user and paired J user cnodes.
  auto manager = primal_graph_->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto prim_j_pair = FindPrimalJPair(manager, primal_graph_);
  for (auto &[primal_user, j_user] : prim_j_pair) {
    if (primal_user == nullptr || j_user == nullptr) {
      // Skip if one of them not found.
      return;
    }

    // Replace primal graph with k graph.
    auto k_vnode = NewValueNode(k_graph_);
    auto primal_abs = primal_user->abstract();
    primal_user->set_input(0, k_vnode);
    primal_user->set_abstract(j_user->abstract());

    // If both inputs are same except monads, we copy primal monad args to k graph
    // so that they can be combined in CSE (common subexpression elimination) pass.
    const bool has_monad = CopyMonadArguments(primal_user, j_user);
    // Remove the UpdateState nodes after primal_user if need.
    if (has_monad) {
      RemovePrimalUpdateStates(manager, primal_user);
    }

    // Insert tuple_getitem after primal user cnode.
    auto construct_wrapper = primal_user->func_graph();
    auto tuple_getitem = NewValueNode(prim::kPrimTupleGetItem);
    auto imm0 = std::make_shared<Int64Imm>(0);
    auto idx0 = NewValueNode(SizeToLong(0));
    idx0->set_abstract(std::make_shared<abstract::AbstractScalar>(imm0));
    auto getitem0 = construct_wrapper->NewCNode({tuple_getitem, primal_user, idx0});
    getitem0->set_abstract(primal_abs);
    manager->Replace(primal_user, getitem0);
  }
}
}  // namespace ad
}  // namespace mindspore
