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

#include "optimizer/ad/dfunctor.h"

#include <memory>
#include <string>
#include <utility>

#include "ir/anf.h"
#include "ir/meta_func_graph.h"
#include "debug/info.h"
#include "ir/func_graph_cloner.h"
#include "ir/manager.h"
#include "pipeline/resource.h"
#include "pipeline/parse/parse.h"
#include "optimizer/ad/adjoint.h"
#include "optimizer/opt.h"
#include "operator/ops.h"
#include "operator/composite/composite.h"
#include "utils/symbolic.h"
#include "utils/context/ms_context.h"
#include "./common.h"

namespace mindspore {
namespace ad {
std::unordered_map<FuncGraphPtr, DFunctorPtr> DFunctor::func_graph_to_functor_;
std::unordered_map<AnfNodePtr, AdjointPtr> DFunctor::anfnode_to_adjoin_definition_;
FuncGraphSet DFunctor::scope_;

DFunctor::DFunctor(const FuncGraphPtr &primal_graph, const pipeline::ResourceBasePtr &resources)
    : primal_graph_(primal_graph), resources_(resources), need_cut_(false), is_top_(false) {
  TraceManager::DebugTrace(std::make_shared<TraceGradFprop>(primal_graph->debug_info()));
  k_graph_ = std::make_shared<FuncGraph>();
  TraceManager::EndTrace();

  TraceManager::DebugTrace(std::make_shared<TraceGradBprop>(primal_graph->debug_info()));
  tape_ = std::make_shared<FuncGraph>();
  TraceManager::EndTrace();

  dout_ = tape_->add_parameter();
}

void DFunctor::Init(const DFunctorPtr &functor, bool is_top) {
  func_graph_to_functor_[primal_graph_] = functor;
  is_top_ = is_top;
  if (is_top) {
    scope_ = primal_graph_->scope();
  }
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
  auto key = tape_->NewCNode({NewValueNode(prim::kPrimEmbed), fv_adjoint->second->k()});
  fv_adjoint->second->RegisterKUser(key, 1);
  auto default_val = tape_->NewCNode({NewValueNode(prim::GetPythonOps("zeros_like")), fv_adjoint->second->k()});
  fv_adjoint->second->RegisterKUser(default_val, 1);
  auto dfv = tape_->NewCNode({NewValueNode(prim::kPrimEnvGetItem), din, key, default_val});
  MS_LOG(DEBUG) << "BackPropagateFv find adjoint in anfnode_to_adjoin_ or anfnode_to_adjoin_indirect_fv_ fv "
                << fv->func_graph()->ToString() << " " << fv->ToString() << ".";
  MS_LOG(DEBUG) << "BackPropagateFv get item from " << din->ToString() << " key " << key->ToString() << ".";
  fv_adjoint->second->AccumulateDout(dfv);
}

void DFunctor::BackPropagateSwitchLayer(const CNodePtr &cnode_morph, const CNodePtr &env) {
  // Take switch_layer as a set of candidate functions.
  auto input = cnode_morph->input(2);
  if (!IsPrimitiveCNode(input, prim::kPrimMakeTuple)) {
    MS_LOG(EXCEPTION) << "The 2th input of switch_layer expect a tuple of graphs, but got " << input->ToString() << ".";
  }
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
      BackPropagateFv(fv, env);
    }
    for (auto indirect_fv : functor->second->anfnode_to_adjoin_indirect_fv_) {
      MS_LOG(DEBUG) << "BackPropagateSwitchLayer backprop indirect fv " << func_graph->ToString() << " "
                    << indirect_fv.first->ToString() << ".";
      BackPropagateFv(indirect_fv.first, env);
    }
  }
}

void DFunctor::BackPropagate(const CNodePtr &cnode_morph, const CNodePtr &k_app, const AdjointPtr &node_adjoint) {
  auto bprop = k_graph_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), k_app, NewValueNode(1)});
  // Call with delimited continuation dout.
  auto bprop_app = tape_->NewCNode({bprop, node_adjoint->dout()});
  node_adjoint->RegisterDoutUser(bprop_app, 1);
  // Special case for switch_layer
  if (IsPrimitiveCNode(cnode_morph, prim::kPrimSwitchLayer)) {
    auto din = tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), bprop_app, NewValueNode(0)});
    BackPropagateSwitchLayer(cnode_morph, din);
    return;
  }
  for (size_t i = 0; i < cnode_morph->size(); i++) {
    auto din = tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), bprop_app, NewValueNode(SizeToInt(i))});
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
  // MapMorphism All type except CNode should already be mapped by MapObject.
  if (!morph->isa<CNode>()) {
    return nullptr;
  }
  ScopeGuard scope_guard(morph->scope());
  auto cnode_morph = morph->cast<CNodePtr>();

  std::vector<AnfNodePtr> inputs;
  std::vector<AdjointPtr> param_adjoints;
  for (size_t i = 0; i < cnode_morph->size(); i++) {
    auto node = cnode_morph->input(i);
    auto node_adjoint_iter = anfnode_to_adjoin_.find(node);
    AdjointPtr node_adjoint = nullptr;
    AnfNodePtr k = nullptr;
    if (node_adjoint_iter != anfnode_to_adjoin_.end()) {
      node_adjoint = node_adjoint_iter->second;
    } else {
      // Input might be a CNode that needs to be handled before hand.
      node_adjoint = MapMorphism(node);
    }
    MS_EXCEPTION_IF_NULL(node_adjoint);
    k = node_adjoint->k();
    if (k == nullptr) {
      MS_LOG(EXCEPTION) << "MapMorphism adjoint node does not exist, input[" << i << "] " << node->ToString() << ".";
    }
    inputs.push_back(k);
    param_adjoints.push_back(node_adjoint);
  }
  TraceManager::DebugTrace(std::make_shared<TraceGradFpropApp>(cnode_morph->debug_info()));
  auto k_app = k_graph_->NewCNode(inputs);
  TraceManager::EndTrace();
  for (size_t i = 0; i < param_adjoints.size(); ++i) {
    param_adjoints[i]->RegisterKUser(k_app, i);
  }

  // Do forward computation
  auto foward_app = k_graph_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), k_app, NewValueNode(0)});
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
  MS_LOG(DEBUG) << "MapMorphism node " << morph->ToString() << ".";
  return node_adjoint;
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
  // Handle cnode not attached to output, that might be refered in other functions.
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
    auto key = tape_->NewCNode({NewValueNode(prim::kPrimEmbed), fv_adjoint->second->k()});
    fv_adjoint->second->RegisterKUser(key, 1);
    auto sens = fv_adjoint->second->dout();
    new_grad_fv = tape_->NewCNode({
      NewValueNode(prim::kPrimEnvSetItem),
      new_grad_fv,
      key,
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
    auto key = tape_->NewCNode({NewValueNode(prim::kPrimEmbed), fv_adjoint.second->k()});
    fv_adjoint.second->RegisterKUser(key, 1);
    auto sens = fv_adjoint.second->dout();
    new_grad_fv = tape_->NewCNode({
      NewValueNode(prim::kPrimEnvSetItem),
      new_grad_fv,
      key,
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

    if (bprop_graph->free_variables_nodes().size() != 0 || primal->free_variables_nodes().size() != 0) {
      MS_LOG(EXCEPTION) << "User defined Cell bprop " << primal->ToString() << " in scope "
                        << primal->output()->scope()->name() << " does not support Parameter data type.";
    }
    auto fg = g_k_prims.KUserDefinedCellBprop(bprop_graph);
    if (fg == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to expand user defined Cell bprop " << primal->ToString() << " in scope "
                        << primal->output()->scope()->name() << ".";
    }

    // Cache the grad func
    (void)primal->transforms().insert(std::make_pair("grad", FuncGraphTransform(fg)));
    (void)fg->transforms().insert(std::make_pair("primal", FuncGraphTransform(primal)));
    // Reset defer_inline to enable successive inlining
    primal->set_flags(FUNC_GRAPH_FLAG_DEFER_INLINE, false);

    auto functor = std::make_shared<DFunctor>(primal, resources_);
    functor->Init(functor);
    functor->k_graph_ = fg;

    return fg;
  }
  return nullptr;
}

// MapToK(func)
AnfNodePtr DFunctor::MapToK(const FuncGraphPtr &primal) {
  auto f = func_graph_to_functor_.find(primal);
  if (f != func_graph_to_functor_.end()) {
    MS_LOG(DEBUG) << "K graph functor already exist " << primal->ToString() << ".";
    return NewValueNode(f->second->k_graph_);
  }

  auto k_user_defined = KUserDefined(primal);
  if (k_user_defined != nullptr) {
    MS_LOG(DEBUG) << "K graph functor user defined bprop " << primal->ToString() << ".";
    return NewValueNode(k_user_defined);
  }

  auto functor = std::make_shared<DFunctor>(primal, resources_);
  functor->Init(functor);
  functor->MapObject();
  functor->MapMorphism();

  MS_LOG(DEBUG) << "K graph K function graph " << primal->ToString() << " " << functor->k_graph_->ToString() << ".";
  return NewValueNode(functor->k_graph_);
}

// Construct representation graph for given node.
AnfNodePtr DFunctor::MapToK(const AnfNodePtr &primal) {
  ScopeGuard scope_guard(primal->scope());
  // MapToK(prim)
  if (IsValueNode<Primitive>(primal)) {
    auto value_node = primal->cast<ValueNodePtr>();
    auto prim = GetValueNode<PrimitivePtr>(value_node);
    if (prim->Hash() == prim::kPrimStopGradient->Hash() && prim->name() == prim::kPrimStopGradient->name()) {
      MS_LOG(DEBUG) << "Meet a kPrimStopGradient " << prim->ToString() << ".";
      need_cut_ = true;
    }
    auto k_prim = g_k_prims.KPrimitive(value_node, resources_);
    if (k_prim != nullptr) {
      k_prim = BasicClone(k_prim);
      return NewValueNode(k_prim);
    }
    // When failed to find k_prim, try k_meta.
    auto k_meta = g_k_prims.KMetaFuncGraph(prim);
    if (k_meta != nullptr) {
      return NewValueNode(k_meta);
    }
  }

  // MapToK(func)
  if (IsValueNode<FuncGraph>(primal)) {
    auto func_graph = GetValueNode<FuncGraphPtr>(primal);
    auto k_func = MapToK(func_graph);
    return k_func;
  }

  if (primal->isa<Parameter>()) {
    TraceManager::DebugTrace(std::make_shared<TraceGradFprop>(primal->debug_info()));
    auto ret = k_graph_->add_parameter();
    TraceManager::EndTrace();
    return ret;
  }

  if (!primal->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "K node keeped node from primal_graph_ " << primal->ToString() << " that is not a ValueNode.";
  }
  return primal;
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
      if (is_top_ || node->isa<Parameter>() || !IsInScope(node)) {
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
    auto adjoint = std::make_shared<Adjoint>(p, MapToK(p), tape_);
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
    // Skip Return.
    if (IsValueNode<Primitive>(node) && GetValueNode<PrimitivePtr>(node) == prim::kPrimReturn) {
      continue;
    }
    MS_LOG(DEBUG) << "MapValueObject node " << node->ToString() << ".";
    auto adjoint = std::make_shared<Adjoint>(node, MapToK(node), tape_);
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
FuncGraphPtr DFunctor::k_graph() {
  CallDoutHoleOnTape();
  return k_graph_;
}

void DFunctor::BroadCastStopFlag() {
  // As stop set expanding, all directly or indirectly stopped CNode will be cut off
  while (need_cut_) {
    need_cut_ = false;
    for (auto &node : primal_graph_->nodes()) {
      if (node->isa<CNode>()) {
        auto cnode = node->cast<CNodePtr>();
        if (!cnode->stop_gradient()) {
          // Cut off the cnode only when it's not referred any more
          if (IsPrimitiveCNode(cnode, prim::kPrimStopGradient) || AllReferencesStopped(cnode)) {
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
}  // namespace ad
}  // namespace mindspore
