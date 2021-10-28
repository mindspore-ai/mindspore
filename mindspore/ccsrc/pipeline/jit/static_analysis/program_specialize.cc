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

#include "pipeline/jit/static_analysis/program_specialize.h"

#include <algorithm>
#include <exception>
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/do_signature.h"
#include "abstract/abstract_function.h"
#include "abstract/utils.h"
#include "utils/utils.h"
#include "ir/graph_utils.h"
#include "utils/log_adapter.h"
#include "debug/trace.h"

namespace mindspore {
namespace abstract {
namespace {
inline AbstractBasePtr GetEvaluatedValue(const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  if (conf->node()->intermediate_abstract()) {
    return conf->node()->intermediate_abstract();
  }
  MS_EXCEPTION_IF_NULL(conf->ObtainEvalResult());
  return conf->ObtainEvalResult()->abstract();
}

AnfNodePtr BuildValueNode(const ValuePtr &v, const AbstractBasePtr &abs_base) {
  MS_EXCEPTION_IF_NULL(abs_base);
  AnfNodePtr value_node = NewValueNode(v);
  value_node->set_abstract(abs_base);
  MS_LOG(DEBUG) << "Create ValueNode: " << value_node->ToString() << ", with abstract: " << abs_base->ToString();
  return value_node;
}

bool IsVisible(FuncGraphPtr fg, const FuncGraphPtr &parent) {
  while (fg != nullptr && fg != parent) {
    fg = fg->parent();
  }
  return fg == parent;
}

bool CheckAbstractTensor(const AbstractBasePtr &abs_base) {
  MS_EXCEPTION_IF_NULL(abs_base);
  if (abs_base->isa<AbstractTensor>()) {
    return true;
  } else if (abs_base->isa<AbstractSequeue>()) {
    const auto &abs_seq = abs_base->cast<AbstractSequeuePtr>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    const auto &elements = abs_seq->elements();
    return std::all_of(elements.cbegin(), elements.cend(), [](const auto &v) { return CheckAbstractTensor(v); });
  } else {
    return false;
  }
}
}  // namespace

FuncGraphPtr ProgramSpecializer::Run(const FuncGraphPtr &fg, const AnalysisContextPtr &context) {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(context);
  MS_LOG(DEBUG) << "Specialize topmost function graph: "
                << (context->func_graph() ? context->func_graph()->ToString() : "FG(Null)");
  if (top_context_ == nullptr) {
    top_context_ = context;
    MS_LOG(INFO) << "Specialize set top func graph context: " << context->ToString();
  }
  return SpecializeFuncGraph(fg, context);
}

FuncGraphPtr ProgramSpecializer::SpecializeFuncGraph(const FuncGraphPtr &fg, const AnalysisContextPtr &context) {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(context);
  auto iter = specializations_.find(context->SpecializeKey());
  if (iter != specializations_.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    return iter->second->specialized_func_graph();
  }

  std::shared_ptr<FuncGraphSpecializer> fg_spec = std::make_shared<FuncGraphSpecializer>(this, fg, context);
  FuncGraphPtr fg2 = fg_spec->specialized_func_graph();
  specializations_[context->SpecializeKey()] = fg_spec;
  fg_spec->Run();
  return fg2;
}

std::shared_ptr<FuncGraphSpecializer> ProgramSpecializer::GetFuncGraphSpecializer(const AnalysisContextPtr &context) {
  MS_EXCEPTION_IF_NULL(context);
  auto iter = specializations_.find(context->SpecializeKey());
  if (iter != specializations_.end()) {
    return iter->second;
  }
  if (context->func_graph() != nullptr) {
    MS_LOG(EXCEPTION) << "Specialize inner error";
  }
  return nullptr;
}

std::string GetNextCounter() {
  static int64_t g_CloneCounter = 1;
  std::string str_count = std::to_string(g_CloneCounter);
  g_CloneCounter++;
  return str_count;
}

FuncGraphSpecializer::FuncGraphSpecializer(ProgramSpecializer *const s, const FuncGraphPtr &fg,
                                           const AnalysisContextPtr &context)
    : specializer_(s), func_graph_(fg), context_(context) {
  parent_ = s->GetFuncGraphSpecializer(context->parent());
  engine_ = s->engine();
  cloner_ = SpecializerClone(fg, std::make_shared<TraceSpecialize>(GetNextCounter()));
  repl_node_ = cloner_->cloned_node();
  specialized_func_graph_ = cloner_->cloned_func_graph()[fg];
  todo_.push_back(fg->get_return());
  auto ps = fg->parameters();
  (void)todo_.insert(todo_.end(), ps.begin(), ps.end());
}

AnfNodePtr FuncGraphSpecializer::ReplicateDisconnectedNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>()) {
    return node;
  }
  std::shared_ptr<FuncGraphSpecializer> specializer = GetTopSpecializer(node);

  // If had replicated, just return that.
  MS_EXCEPTION_IF_NULL(specializer->repl_node_);
  auto iter = specializer->repl_node_->find(node);
  if (iter != specializer->repl_node_->end()) {
    return iter->second;
  }
  auto new_node = specializer->cloner_->CloneDisconnected(node);
  if (node->isa<CNode>()) {
    if (!new_node->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "new_node must be a CNode, but is " << new_node->DebugString() << ".";
    }
    UpdateNewCNodeInputs(node, new_node);
  }

  iter = specializer->repl_node_->find(node);
  if (iter != specializer->repl_node_->end()) {
    if (iter->second == node) {
      MS_LOG(EXCEPTION) << "Replicated is same as original node, node: " << node->ToString();
    }
  } else {
    MS_LOG(EXCEPTION) << "Replicate node failed, node: " << node->ToString();
  }
  return new_node;
}

void FuncGraphSpecializer::UpdateNewCNodeInputs(const AnfNodePtr &node, const AnfNodePtr &new_node) {
  MS_EXCEPTION_IF_NULL(node);
  auto c_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(c_node);
  auto inputs = c_node->inputs();
  std::vector<AnfNodePtr> new_inputs;
  (void)std::transform(
    inputs.begin(), inputs.end(), std::back_inserter(new_inputs), [this](const AnfNodePtr &inp) -> AnfNodePtr {
      auto new_inp = ReplicateDisconnectedNode(inp);
      // Refer the comments in BuildReplacedNode.
      if (inp->isa<CNode>()) {
        auto c_inp = inp->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(c_inp);
        auto c_new_inp = new_inp->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(c_new_inp);
        MS_EXCEPTION_IF_NULL(c_new_inp->func_graph());
        MS_LOG(DEBUG) << "Replace in order, inp node: " << inp->DebugString() << " -> " << new_inp->DebugString();
        c_new_inp->func_graph()->ReplaceInOrder(c_inp, c_new_inp);
      }
      return new_inp;
    });

  auto c_new_node = new_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(c_new_node);
  c_new_node->set_inputs(new_inputs);
}

AnfNodePtr FuncGraphSpecializer::GetReplicatedNode(const AnfNodePtr &node) {
  std::shared_ptr<FuncGraphSpecializer> specializer = GetTopSpecializer(node);
  MS_EXCEPTION_IF_NULL(specializer->repl_node_);
  auto iter = specializer->repl_node_->find(node);
  if (iter != specializer->repl_node_->end()) {
    return iter->second;
  }
  return node;
}

// Return itself if node's ValueNode as top,
// return the top func graph specializer as top if node's forward Parameter,
// or, return the top parent specializer as top.
std::shared_ptr<FuncGraphSpecializer> FuncGraphSpecializer::GetTopSpecializer(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  FuncGraphPtr fg = node->func_graph();
  if (fg == nullptr) {  // If ValueNode, return current specializer.
    MS_LOG(DEBUG) << "Node's a ValueNode, node: " << node->DebugString();
    return shared_from_this();
  }
  std::shared_ptr<FuncGraphSpecializer> specializer = shared_from_this();
  while (fg != specializer->func_graph_) {
    if (specializer->parent_ == nullptr && node->isa<Parameter>()) {
      // If `parent_` is null and forwarded `node` is a Parameter, we'll try to use top func graph as parent.
      MS_EXCEPTION_IF_NULL(specializer_->top_context());
      if (specializer_->top_context()->func_graph() == fg) {  // `fg` is top func graph.
        specializer = specializer_->GetFuncGraphSpecializer(specializer_->top_context());
        MS_LOG(INFO) << "Used top func graph specializer as parent for "
                     << (func_graph_ ? func_graph_->ToString() : "FG(Null)") << ", node: " << node->DebugString()
                     << ", NodeInfo: " << trace::GetDebugInfo(node->debug_info());
        MS_EXCEPTION_IF_NULL(specializer);
        break;
      }
    } else {
      specializer = specializer->parent_;
    }
    if (specializer == nullptr) {
      MS_LOG(EXCEPTION) << "`specializer` should not be null, node: " << node->DebugString()
                        << ", NodeInfo: " << trace::GetDebugInfo(node->debug_info()) << ".\n"
                        << (func_graph_ ? func_graph_->ToString() : "FG(Null)")
                        << " has no parent context? At least not " << fg->ToString();
    }
  }
  return specializer;
}

void FuncGraphSpecializer::Run() {
  MS_LOG(DEBUG) << "Before run, origin func graph name: " << (func_graph_ ? func_graph_->ToString() : "FG(Null)")
                << ", cloned func graph name: "
                << (specialized_func_graph_ ? specialized_func_graph_->ToString() : "FG(Null)") << ", func graph: "
                << (func_graph_ ? func_graph_->get_return() ? func_graph_->get_return()->DebugString() : "return null"
                                : "FG(null)");
  FirstPass();
  SecondPass();
  MS_LOG(DEBUG) << "After run, origin func graph name: " << (func_graph_ ? func_graph_->ToString() : "FG(Null)")
                << ", cloned func graph name: "
                << (specialized_func_graph_ ? specialized_func_graph_->ToString() : "FG(Null)") << ", new func graph: "
                << (specialized_func_graph_ ? specialized_func_graph_->get_return()
                                                ? specialized_func_graph_->get_return()->DebugString()
                                                : "return null"
                                            : "FG(null)");
}

void FuncGraphSpecializer::FirstPass() {
  while (todo_.size()) {
    AnfNodePtr node = todo_.back();
    todo_.pop_back();
    if (node->func_graph() == nullptr) {
      // do nothing for ValueNode
      continue;
    }
    if (node->func_graph() != func_graph_) {
      std::shared_ptr<FuncGraphSpecializer> parent = nullptr;
      if (parent_ != nullptr) {
        parent = parent_;
      } else if (specializer_->top_context()->func_graph() == node->func_graph() && node->isa<Parameter>()) {
        // If `parent_` is null and forwarded `node` is a Parameter, we'll try to use top func graph as parent.
        parent = specializer_->GetFuncGraphSpecializer(specializer_->top_context());
        MS_LOG(INFO) << "Used top func graph specializer as parent for " << func_graph_->ToString()
                     << ", node: " << node->DebugString() << ", NodeInfo: " << trace::GetDebugInfo(node->debug_info());
      }
      if (parent == nullptr) {
        MS_LOG(EXCEPTION) << "Parent must not null, node: " << node->DebugString()
                          << ", NodeInfo: " << trace::GetDebugInfo(node->debug_info());
      }
      parent->AddTodoItem(node);
      parent->FirstPass();
      AnfNodePtr new_node = parent->GetReplicatedNode(node);
      if (node->isa<CNode>()) {
        parent->ProcessCNode(new_node->cast<CNodePtr>());
      }
      continue;
    }
    if (marked_.count(node) > 0) {
      continue;
    }
    (void)marked_.insert(node);
    ProcessNode(node);
  }
}

// Specialize CNode in func graphs
void FuncGraphSpecializer::SecondPass() {
  for (auto &node : BroadFirstSearchGraphCNodes({specialized_func_graph_->get_return()})) {
    if (node->isa<CNode>()) {
      ProcessCNode(node->cast<CNodePtr>());
    }
  }
}

void FuncGraphSpecializer::ProcessNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  ScopeGuard scope_guard(node->scope());
  AnfNodeConfigPtr conf = MakeConfig(node);
  AnfNodePtr new_node = GetReplicatedNode(node);
  MS_EXCEPTION_IF_NULL(new_node);
  if (new_node->func_graph() != specialized_func_graph_) {
    MS_LOG(EXCEPTION) << "Error in specializer [A] node: " << node->DebugString()
                      << ", new_node: " << new_node->DebugString() << ", new_node->func_graph(): "
                      << (new_node->func_graph() ? new_node->func_graph()->ToString() : "FG(Null)")
                      << ", specialized_func_graph_: " << specialized_func_graph_->ToString();
    return;
  }
  new_node->set_abstract(GetEvaluatedValue(conf));
  if (new_node->isa<CNode>() && new_node->abstract()->isa<PartialAbstractClosure>()) {
    auto partial_abstract = dyn_cast<PartialAbstractClosure>(new_node->abstract());
    if (partial_abstract->node() == node) {
      partial_abstract->set_node(new_node);
    }
  }

  MS_LOG(DEBUG) << "Set new_node: " << new_node->ToString() << ", abstract as: " << new_node->abstract()->ToString();

  if (node->isa<CNode>()) {
    auto attrs = conf->ObtainEvalResult()->attribute();
    auto c_old = node->cast<CNodePtr>();
    auto c_new = new_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(c_new);
    auto new_inputs = c_new->inputs();
    auto old_inputs = c_old->inputs();
    for (size_t i = 0; i < old_inputs.size(); ++i) {
      auto node_input = old_inputs[i];
      AnfNodeConfigPtr iconf = MakeConfig(node_input);
      AbstractBasePtr ival = GetEvaluatedValue(iconf);
      // First try to check if node_input can be replaced by a ValueNode. If cannot, then try to check if
      // can be replaced by another CNode from anfnode_config_map, otherwise use the replicated node.
      AnfNodePtr replace_node = BuildPossibleValueNode(iconf->node(), ival, attrs);
      if (replace_node == nullptr) {
        replace_node = BuildReplacedNode(iconf);
        replace_node->set_abstract(ival);
        MS_LOG(DEBUG) << "Set replaced: " << replace_node->ToString() << ", to abstract: " << ival->ToString();
      } else {
        MS_LOG(DEBUG) << "Build possible value node for node: " << node_input->DebugString()
                      << ", ival: " << ival->ToString() << ", replace_node: " << replace_node->ToString();
      }
      if (new_inputs[i] != replace_node) {
        new_inputs[i] = replace_node;
        MS_LOG(DEBUG) << "Set new_input[" << i << "] = " << replace_node->DebugString();
      }
    }
    c_new->set_inputs(new_inputs);
  }
}

AnfNodePtr FuncGraphSpecializer::BuildReplacedNode(const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);

  auto conf_iter = engine_->anfnode_config_map().find(conf);
  AnfNodeConfigPtr new_conf = conf;
  while (conf_iter != engine_->anfnode_config_map().end()) {
    MS_LOG(DEBUG) << "Origin conf: node(" << (new_conf->node() ? new_conf->node()->DebugString() : "Node(Null)") << ")";
    new_conf = conf_iter->second;
    MS_EXCEPTION_IF_NULL(new_conf);
    const auto &forward_node = new_conf->node();
    MS_LOG(DEBUG) << "Replaced conf: node(" << forward_node->DebugString() << ")";
    const auto &replicated_forward_node = ReplicateDisconnectedNode(forward_node);
    if (replicated_forward_node && replicated_forward_node->isa<CNode>()) {
      // The AnfNode in order_list can be:
      // case 1: also in FuncGraphManager, so it can be got from nodes API of func_graph. it will
      //         be replaced in CloneOrderList in Cloner.
      // case 2: AnfNode is not in FuncGraphManager which generated in Analyze phase, so it will not
      //         be cloned by normal clone API.
      //    2.1: A forward node , the original node is in FuncGraphManager. The original node will
      //         be cloned in CloneOrderList in Cloner, and the replicated forward node will replace
      //         the replicated original node here.
      //    2.2: an input of a forward node, such as Cast CNode generated in DoCast. It is also another
      //         original node to fowrad.
      //    2.3: an input of an input of a forward node, but it's not an original node. Like the Cast CNode
      //         in MixedPrecisionCastHelper.
      // For 2.2 and 2.3, we will put a placeholder in order list of replicated func_graph, refer to
      // CloneOrderlist, and it will be replaced inside ReplicateDisconnectedNode.
      // For 2.1 the following code will do the job, replace replicated origin cnode with the replicated
      // forward one in the replicated func_graph.
      MS_EXCEPTION_IF_NULL(conf_iter->first);
      const auto &origin_node = conf_iter->first->node();
      const auto &replicated_origin_node = GetReplicatedNode(origin_node);
      if (replicated_origin_node != origin_node) {
        MS_LOG(DEBUG) << "Replace replicated origin node in order list: " << replicated_origin_node->DebugString()
                      << ", with replicated forwarded node: " << replicated_forward_node->DebugString();
        MS_EXCEPTION_IF_NULL(replicated_forward_node->func_graph());
        replicated_forward_node->func_graph()->ReplaceInOrder(replicated_origin_node, replicated_forward_node);
      } else {
        MS_LOG(EXCEPTION) << "Origin node is not replicated in specialized func_graph, origin node: "
                          << (origin_node ? origin_node->DebugString() : "Node(Null)");
      }
    }
    conf_iter = engine_->anfnode_config_map().find(new_conf);
  }
  todo_.push_back(new_conf->node());
  auto repl = GetReplicatedNode(new_conf->node());
  if (repl->func_graph()) {
    MS_LOG(DEBUG) << "Set repl: graph(" << repl->func_graph()->ToString() << "), node:" << repl->DebugString()
                  << ") to replace origin:" << new_conf->node()->DebugString();
  } else {
    MS_LOG(DEBUG) << "Set repl: graph(nullptr), node(" << repl->DebugString()
                  << ") to replace origin: " << new_conf->node()->DebugString();
  }
  return repl;
}

namespace {
const StringImmPtr kDeadNode = std::make_shared<StringImm>(kDeadNodeName);
const StringImmPtr kPolyNode = std::make_shared<StringImm>(kPolyNodeName);

inline bool CanSpecializeNode(const AnfNodePtr &node) {
  if (IsValueNode<FuncGraph>(node) || IsValueNode<MetaFuncGraph>(node) || IsValueNode<Primitive>(node)) {
    return true;
  }
  return false;
}
}  // namespace

AnfNodePtr FuncGraphSpecializer::BuildSpecializedNode(const AnfNodePtr &node, const AbstractBasePtr &abs,
                                                      const AbstractBasePtrList &argvals) {
  MS_EXCEPTION_IF_NULL(abs);
  MS_EXCEPTION_IF_NULL(node);
  AbstractFunctionPtr real_a = dyn_cast<AbstractFunction>(abs);
  MS_EXCEPTION_IF_NULL(real_a);

  AbstractFunctionPtr func = real_a->GetUnique();
  SpecializeStatusCode errcode;
  ScopeGuard scope_guard(node->scope());
  AnfNodePtr repl = BuildSpecializedNodeInner(node, abs, func, argvals, &errcode);
  if (repl == nullptr) {
    if (errcode == kSpecializeFindUniqueArgvalDead) {
      const auto error_dead_node = std::make_shared<AbstractError>(kDeadNode, node);
      repl = BuildValueNode(kDeadNode, error_dead_node);
      MS_LOG(DEBUG) << "DEAD for node: " << node->DebugString() << ", abstract: " << abs->ToString();
    } else if (errcode == kSpecializeFindUniqueArgvalPoly) {
      const auto error_poly_node = std::make_shared<AbstractError>(kPolyNode, node);
      repl = BuildValueNode(kPolyNode, error_poly_node);
      MS_LOG(DEBUG) << "POLY for node: " << node->DebugString() << ", abstract: " << abs->ToString();
    } else {
      MS_LOG(EXCEPTION) << "Failed to build specialized node, node: " << node->DebugString()
                        << ", abstract: " << abs->ToString();
    }
  }

  // Set the flag, so this MetaFuncGraph will be Re-AutoMonaded.
  MS_EXCEPTION_IF_NULL(func);
  if (func->isa<MetaFuncGraphAbstractClosure>()) {
    auto specialized_fg = GetValueNode<FuncGraphPtr>(repl);
    if (specialized_fg != nullptr && (argvals.size() > 1) && argvals.back() != nullptr &&
        argvals.back()->isa<AbstractUMonad>()) {
      specialized_fg->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);
    }
  }
  return repl;
}

AnfNodePtr FuncGraphSpecializer::BuildSpecializedNodeInner(const AnfNodePtr &node, const AbstractBasePtr &abs,
                                                           const AbstractFunctionPtr &func,
                                                           const AbstractBasePtrList &args,
                                                           SpecializeStatusCode *errcode) {
  MS_EXCEPTION_IF_NULL(abs);
  MS_EXCEPTION_IF_NULL(func);
  MS_EXCEPTION_IF_NULL(errcode);
  *errcode = kSpecializeSuccess;

  auto real_func = dyn_cast<TypedPrimitiveAbstractClosure>(func);
  if (real_func != nullptr) {
    return BuildValueNode(real_func->prim(), abs);
  }

  EvaluatorPtr eval = engine_->GetEvaluatorFor(func);
  MS_EXCEPTION_IF_NULL(eval);
  AbstractBasePtrList argvals = eval->NormalizeArgs(args);

  std::pair<AbstractBasePtrList, AbstractBasePtr> result;
  SpecializeStatusCode status = FindUniqueArgvals(func, eval, argvals, &result);
  if (status != kSpecializeSuccess) {
    *errcode = status;
    return nullptr;
  }
  argvals = result.first;
  AbstractBasePtr unique_output = result.second;

  auto prim_func = dyn_cast<PrimitiveAbstractClosure>(func);
  if (prim_func != nullptr) {
    auto type_func = std::make_shared<TypedPrimitiveAbstractClosure>(prim_func->prim(), argvals, unique_output);
    return BuildValueNode(prim_func->prim(), type_func);
  }

  if (!eval->isa<BaseFuncGraphEvaluator>()) {
    MS_LOG(EXCEPTION) << "Eval is not BaseGraphEvaluator, but " << eval->ToString();
  }
  auto real_eval = dyn_cast<BaseFuncGraphEvaluator>(eval);

  if (func->context() == nullptr) {
    MS_LOG(EXCEPTION) << "Func context is nullptr NodeInfo: " << trace::GetDebugInfo(func_graph_->debug_info());
  }
  AnalysisContextPtr context = MakeContext(engine_, real_eval, argvals);
  MS_LOG(DEBUG) << "Specialize function graph: " << context->func_graph()->ToString() << ", args: " << argvals.size()
                << ", graph: " << context->func_graph()->get_return()->DebugString();
  MS_EXCEPTION_IF_NULL(context->func_graph());
  if (context->func_graph()->stub()) {
    MS_LOG(DEBUG) << "Specialize stub function graph, return the original node: " << context->func_graph()->ToString()
                  << ", args: " << argvals.size() << ", graph: " << context->func_graph()->get_return()->DebugString()
                  << ", " << node->ToString();
    return node;
  }
  FuncGraphPtr v = specializer_->SpecializeFuncGraph(context->func_graph(), context);
  MS_EXCEPTION_IF_NULL(v);
  v->set_flag(kFuncGraphFlagUndetermined, false);
  return BuildValueNode(v, abs);
}

inline AnalysisContextPtr FuncGraphSpecializer::MakeContext(const AnalysisEnginePtr &engine,
                                                            const BaseFuncGraphEvaluatorPtr &evaluator,
                                                            const AbstractBasePtrList &args_spec_list) {
  AbstractBasePtrList normalized_args_spec_list = evaluator->NormalizeArgs(args_spec_list);
  FuncGraphPtr fg = evaluator->GetFuncGraph(engine, normalized_args_spec_list);
  MS_EXCEPTION_IF_NULL(evaluator->parent_context());
  AnalysisContextPtr new_context = evaluator->parent_context()->NewContext(fg, normalized_args_spec_list);
  return new_context;
}

AnfNodePtr FuncGraphSpecializer::BuildSpecializedParameterNode(const CNodePtr &new_node) {
  MS_EXCEPTION_IF_NULL(new_node);
  auto new_inputs = new_node->inputs();
  if (new_inputs.empty()) {
    MS_LOG(EXCEPTION) << "inputs can't be empty.";
  }
  AnfNodePtr func = new_inputs[0];
  MS_EXCEPTION_IF_NULL(new_inputs[0]);
  AbstractBasePtr fnval = new_inputs[0]->abstract();

  AbstractBasePtrList args;
  auto backed_fnval = fnval;
  if (fnval->isa<PartialAbstractClosure>()) {
    auto partial_closure = dyn_cast<PartialAbstractClosure>(fnval);
    backed_fnval = partial_closure->fn();
    args = partial_closure->args();
  }
  std::transform(new_inputs.cbegin() + 1, new_inputs.cend(), std::back_inserter(args),
                 [](const AnfNodePtr &inp) { return inp->abstract(); });

  ScopeGuard scope_guard(new_node->scope());

  auto specialized_node = BuildSpecializedNode(func, backed_fnval, args);
  auto wrapped_node = specialized_node;
  if (fnval->isa<PartialAbstractClosure>()) {
    auto partial_closure = dyn_cast<PartialAbstractClosure>(fnval);
    AnfNodePtrList partial_node_list = {BuildValueNode(prim::kPrimPartial, FromValueInside(prim::kPrimPartial)),
                                        specialized_node};
    auto anf_node = partial_closure->node();
    if (!anf_node->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "Must be cnode, but " << anf_node->DebugString();
    }
    auto cnode = anf_node->cast<CNodePtr>();
    if (cnode->size() != partial_closure->args().size() + 2) {
      MS_LOG(EXCEPTION) << "Size of cnode: " << cnode->DebugString()
                        << " is not equal to 2 added to size of args: " << mindspore::ToString(partial_closure->args());
    }
    auto attrs = std::make_shared<AttrValueMap>();
    for (size_t i = 0; i < partial_closure->args().size(); i++) {
      auto old_node = cnode->input(i + 2);
      auto possibile_value_node = BuildPossibleValueNode(old_node, partial_closure->args()[i], attrs);
      if (possibile_value_node != nullptr) {
        partial_node_list.push_back(possibile_value_node);
      } else {
        if (!(old_node->isa<CNode>() || old_node->isa<Parameter>())) {
          MS_LOG(EXCEPTION) << "Old node should be CNode or Parameter, but " << old_node->ToString();
        }
        partial_node_list.push_back(old_node);
      }
    }
    MS_EXCEPTION_IF_NULL(new_node->func_graph());
    wrapped_node = new_node->func_graph()->NewCNode(partial_node_list);
    wrapped_node->set_abstract(partial_closure);
  }
  return wrapped_node;
}

const EvaluatorCacheMgrPtr FuncGraphSpecializer::GetEvalCache(const EvaluatorPtr &eval) {
  MS_EXCEPTION_IF_NULL(eval);
  auto cache_iter = evalcaches_.find(eval);
  if (cache_iter == evalcaches_.end()) {
    evalcaches_[eval] = eval->evaluator_cache_mgr();
    return eval->evaluator_cache_mgr();
  }
  return cache_iter->second;
}

std::pair<AbstractBasePtrList, AbstractBasePtr> FuncGraphSpecializer::BuildFromBroadedArgsVal(
  const EvaluatorPtr &eval) {
  MS_EXCEPTION_IF_NULL(eval);
  std::unordered_set<AbstractBasePtrList, AbstractBasePtrListHasher, AbstractBasePtrListEqual> choices;
  EvalResultPtr ret = nullptr;
  AbstractBasePtrList broaded_argvals;
  std::vector<AbstractBasePtrList> args_vector;
  auto eval_cache_iter = evalcaches_.find(eval);
  if (eval_cache_iter == evalcaches_.end()) {
    MS_LOG(EXCEPTION) << "Evaluator:" << eval->ToString() << " not exist in cache.";
  }
  auto &origin_eval_cache = eval_cache_iter->second->GetCache();
  for (auto &argvals_map : origin_eval_cache) {
    auto argvals = argvals_map.first;
    args_vector.push_back(argvals);
    broaded_argvals.clear();
    BroadenArgs(argvals, &broaded_argvals);
    (void)choices.insert(broaded_argvals);
    MS_LOG(DEBUG) << "Broaded_argvals: " << broaded_argvals.size() << ", " << ::mindspore::ToString(broaded_argvals);
  }
  if (choices.size() == 1) {
    constexpr auto args_size = 2;
    if (args_vector.size() < args_size) {
      MS_LOG(EXCEPTION) << "Should have " << args_size << " or more choices, but: " << args_vector.size();
    }
    AbstractBasePtrList joined_argvals = args_vector[0];
    for (size_t i = 1; i < args_vector.size(); ++i) {
      joined_argvals = abstract::AbstractJoin(joined_argvals, args_vector[i]);
    }
    MS_LOG(DEBUG) << "Joined argvals: " << joined_argvals.size() << ", " << ::mindspore::ToString(joined_argvals);
    EvaluatorCacheMgrPtr real = std::make_shared<EvaluatorCacheMgr>();
    const auto joined_eval_result = origin_eval_cache.get(joined_argvals);
    if (joined_eval_result != nullptr) {
      MS_LOG(DEBUG) << "Find unique Choices in original eval cache, so use it: " << joined_eval_result->ToString();

      real->SetValue(joined_argvals, joined_eval_result);
      evalcaches_[eval] = real;
      return std::make_pair(joined_argvals, joined_eval_result->abstract());
    } else {
      bool all_args_tensor = std::all_of(broaded_argvals.cbegin(), broaded_argvals.cend(),
                                         [](const AbstractBasePtr &v) { return CheckAbstractTensor(v); });
      if (all_args_tensor) {
        ConfigPtrList args_conf_list;
        (void)std::transform(broaded_argvals.cbegin(), broaded_argvals.cend(), std ::back_inserter(args_conf_list),
                             [](const AbstractBasePtr &v) -> ConfigPtr { return std::make_shared<VirtualConfig>(v); });
        MS_LOG(WARNING) << "Cannot find joined argvals in cache, run with broaded argsvals: " << broaded_argvals.size()
                        << ", " << ::mindspore::ToString(broaded_argvals);
        ret = eval->SingleRun(engine_, args_conf_list, nullptr);
        MS_EXCEPTION_IF_NULL(ret);
        real->SetValue(broaded_argvals, ret);
        evalcaches_[eval] = real;
        return std::make_pair(broaded_argvals, ret->abstract());
      }
    }
  }
  MS_LOG(DEBUG) << "Choices.size: " << choices.size();
  return std::make_pair(AbstractBasePtrList(), nullptr);
}

void FuncGraphSpecializer::ProcessCNode(const CNodePtr &new_node) {
  MS_EXCEPTION_IF_NULL(new_node);
  if (specializer_->seen().count(new_node) > 0) {
    return;
  }
  specializer_->AddSeen(new_node);
  auto new_inputs = new_node->inputs();
  if (new_inputs.empty()) {
    MS_LOG(EXCEPTION) << "Inputs of CNode is empty";
  }
  AnfNodePtr func = new_inputs[0];
  MS_EXCEPTION_IF_NULL(func);

  // First element is func so arg start from 1
  std::vector<AnfNodePtr> args(new_inputs.begin() + 1, new_inputs.end());
  // CNode(CNode(Partial, f, arg1), arg2, ...) --> CNode(f, arg1, arg2, ...)
  const size_t arg_start_index = 2;
  while (IsPrimitiveCNode(func, prim::kPrimPartial)) {
    std::vector<AnfNodePtr> inputs = func->cast<CNodePtr>()->inputs();
    // First element is partial, second is func so arg is start from 2
    (void)args.insert(args.begin(), inputs.begin() + SizeToInt(arg_start_index), inputs.end());
    func = inputs[1];
  }
  new_inputs = args;
  (void)new_inputs.insert(new_inputs.begin(), func);

  AbstractBasePtrList argvals;
  MS_EXCEPTION_IF_NULL(new_inputs[0]);
  AbstractBasePtr fnval = new_inputs[0]->abstract();
  MS_LOG(DEBUG) << "The new_inputs[0] node: pointer: " << new_inputs[0]->ToString() << ", "
                << new_inputs[0]->DebugString() << ", abstract: " << new_inputs[0]->abstract()->ToString();

  // First element is func so function arguments start from 1
  for (size_t i = 1; i < new_inputs.size(); ++i) {
    argvals.push_back(new_inputs[i]->abstract());
    MS_LOG(DEBUG) << "The new_inputs[" << i << "] node: pointer: " << new_inputs[i]->ToString() << ", "
                  << new_inputs[i]->DebugString() << ", abstract: " << new_inputs[i]->abstract()->ToString();
  }

  if (!func->isa<ValueNode>()) {
    MS_LOG(DEBUG) << func->abstract()->type_name() << " | " << func->abstract()->ToString();
    if (func->abstract()->isa<AbstractFunction>() && !func->abstract()->isa<AbstractFuncUnion>()) {
      auto func_abs = func->abstract()->cast<AbstractFunctionPtr>();
      EvaluatorPtr eval = engine_->GetEvaluatorFor(func_abs);
      std::pair<AbstractBasePtrList, AbstractBasePtr> result;
      AbstractBasePtrList empty_args;
      auto status = FindUniqueArgvals(func_abs, eval, empty_args, &result);
      MS_LOG(DEBUG) << "FindUniqueArgvals return status: " << status;
      // if a node is a poly node, or an input parameter is a PartialAbstractClosure, expand it early
      MS_EXCEPTION_IF_NULL(func->func_graph());
      if (status == kSpecializeFindUniqueArgvalPoly ||
          (func->isa<Parameter>() && func->func_graph()->has_flag(FUNC_GRAPH_FLAG_SPECIALIZE_PARAMETER))) {
        auto wrapped_node = BuildSpecializedParameterNode(new_node);
        new_inputs[0] = wrapped_node;
      }
    }
  }

  if (CanSpecializeNode(func)) {
    // for primitive node, we build the primitive node with inferred attributes in the first pass
    // so we do not build replaced node again here in second pass
    if (IsValueNode<Primitive>(func)) {
      new_inputs[0] = func;
    } else {
      new_inputs[0] = BuildSpecializedNode(func, fnval, argvals);
    }
  }

  for (size_t i = 0; i < argvals.size();) {
    size_t next = i + 1;
    if (CanSpecializeNode(args[i])) {
      new_inputs[next] = BuildSpecializedNode(args[i], argvals[i], std::vector<AbstractBasePtr>{});
    }
    i = next;
  }
  new_node->set_inputs(new_inputs);
}

namespace {
void DumpEvaluatorCache(const EvaluatorCacheMgrPtr &evaluator_cache_mgr, const AbstractBasePtrList &argvals) {
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr);
  MS_LOG(DEBUG) << "Find unique argvals failed: " << argvals.size() << ", " << argvals << ". Check cache all items.";
  int64_t i = 0;
  const EvalResultCache &map = evaluator_cache_mgr->GetCache();
  for (const auto &item : map) {
    MS_LOG(DEBUG) << "evaluator_cache[" << i++ << "]: " << item.first;
  }
}

bool IsPolyFunc(const AbstractFunctionPtr &func, const AbstractBasePtrList &argvals) {
  MS_EXCEPTION_IF_NULL(func);
  if (func->isa<PrimitiveAbstractClosure>() && argvals.empty()) {
    MS_LOG(DEBUG) << "High order primitive return POLY.";
    return true;
  }
  if (func->isa<MetaFuncGraphAbstractClosure>() && argvals.empty()) {
    auto meta_func_graph_wrapper = dyn_cast<MetaFuncGraphAbstractClosure>(func);
    auto meta_func_graph = meta_func_graph_wrapper->meta_func_graph();
    if (meta_func_graph != nullptr && meta_func_graph->isa<prim::DoSignatureMetaFuncGraph>()) {
      auto do_signature = dyn_cast<prim::DoSignatureMetaFuncGraph>(meta_func_graph);
      if (do_signature != nullptr && do_signature->function()->isa<Primitive>()) {
        MS_LOG(DEBUG) << "High order primitive " << do_signature->function()->ToString() << " return POLY.";
        return true;
      }
    }
  }
  return false;
}
}  // end anonymous namespace

SpecializeStatusCode FuncGraphSpecializer::FindUniqueArgvals(const AbstractFunctionPtr &func, const EvaluatorPtr &eval,
                                                             const AbstractBasePtrList &argvals,
                                                             std::pair<AbstractBasePtrList, AbstractBasePtr> *result) {
  MS_EXCEPTION_IF_NULL(func);
  MS_EXCEPTION_IF_NULL(eval);
  MS_EXCEPTION_IF_NULL(result);

  EvaluatorCacheMgrPtr evaluator_cache_mgr = eval->evaluator_cache_mgr();
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr);
  auto data = evaluator_cache_mgr->GetValue(argvals);
  if (data != nullptr) {
    *result = std::make_pair(argvals, data->abstract());
    return kSpecializeSuccess;
  }
  DumpEvaluatorCache(evaluator_cache_mgr, argvals);

  auto cache = GetEvalCache(eval);
  MS_EXCEPTION_IF_NULL(cache);
  const EvalResultCache &choices = cache->GetCache();
  if (choices.get(argvals) != nullptr) {
    MS_EXCEPTION_IF_NULL(cache->GetValue(argvals));
    *result = std::make_pair(argvals, cache->GetValue(argvals)->abstract());
    return kSpecializeSuccess;
  } else if (choices.size() == 1) {
    MS_LOG(DEBUG) << "Evaluator cache has a single item, just use it.";
    MS_EXCEPTION_IF_NULL(choices.begin()->second);
    *result = std::make_pair(choices.begin()->first, choices.begin()->second->abstract());
    return kSpecializeSuccess;
  } else if (choices.empty()) {
    MS_LOG(DEBUG) << "Find DEAD code, it may be optimized in later phase " << func->ToString() << " | "
                  << func->type_name();
    return kSpecializeFindUniqueArgvalDead;
  } else {
    if (IsPolyFunc(func, argvals)) {
      return kSpecializeFindUniqueArgvalPoly;
    }

    MS_LOG(DEBUG) << "Try to find generalized argvals.";
    *result = BuildFromBroadedArgsVal(eval);
    if (!result->first.empty()) {
      return kSpecializeSuccess;
    }
    MS_LOG(DEBUG) << "Find POLY code, it may be unused code or unresolved polymorphism.";
    return kSpecializeFindUniqueArgvalPoly;
  }
}
static PrimitivePtr BuildPrimtiveValueWithAttributes(const PrimitivePtr &prim, const AttrValueMapPtr &attrs) {
  MS_EXCEPTION_IF_NULL(prim);
  auto &prim_attrs = prim->attrs();
  bool is_attr_same = true;
  for (auto &item : *attrs) {
    auto itr = prim_attrs.find(item.first);
    if (itr != prim_attrs.end()) {
      MS_EXCEPTION_IF_NULL(itr->second);
      MS_EXCEPTION_IF_NULL(item.second);
      if (!(*(itr->second) == *(item.second))) {
        is_attr_same = false;
        break;
      }
    } else {
      is_attr_same = false;
      break;
    }
  }
  if (!is_attr_same) {
    auto cloned_prim = prim->Clone();
    for (auto &item : *attrs) {
      cloned_prim->AddAttr(item.first, item.second);
    }
    return cloned_prim;
  }
  return prim;
}

AnfNodePtr FuncGraphSpecializer::BuildPossibleValueNode(const AnfNodePtr &origin_node, const AbstractBasePtr &ival,
                                                        const AttrValueMapPtr &attrs) {
  MS_EXCEPTION_IF_NULL(origin_node);
  MS_EXCEPTION_IF_NULL(ival);

  AbstractFunctionPtr abs = dyn_cast<AbstractFunction>(ival);
  if (abs != nullptr) {
    // Cannot build a deterministic ValueNode if there are multiple possible AbstractFunction.
    if (abs->isa<AbstractFuncUnion>()) {
      return nullptr;
    }
    ValuePtr value = nullptr;
    if (abs->isa<PrimitiveAbstractClosure>()) {
      auto real_fn = dyn_cast<PrimitiveAbstractClosure>(abs);
      // for primitive, check if the attribute is the same with cnode inferred attribute, if not, clone a new one
      if (attrs != nullptr) {
        value = BuildPrimtiveValueWithAttributes(real_fn->prim(), attrs);
      } else {
        value = real_fn->prim();
      }
    } else if (abs->isa<MetaFuncGraphAbstractClosure>()) {
      auto real_fn = dyn_cast<MetaFuncGraphAbstractClosure>(abs);
      value = real_fn->meta_func_graph();
    } else if (abs->isa<FuncGraphAbstractClosure>()) {
      auto real_fn = dyn_cast<FuncGraphAbstractClosure>(abs);
      value = real_fn->func_graph();
    } else {
      return nullptr;
    }
    MS_EXCEPTION_IF_NULL(value);
    if (!value->isa<FuncGraph>() || value->cast<FuncGraphPtr>()->parent() == nullptr ||
        (IsValueNode<FuncGraph>(origin_node) && IsVisible(func_graph_, value->cast<FuncGraphPtr>()->parent()))) {
      return BuildValueNode(value, ival);
    } else {
      return nullptr;
    }
  } else {
    ValuePtr val = ival->BuildValue();
    if (val->isa<AnyValue>()) {
      return nullptr;
    }
    // keep primitive 'depend' not to be optimized
    if (IsPrimitiveCNode(origin_node, prim::kPrimDepend)) {
      return nullptr;
    }
    return BuildValueNode(val, ival);
  }
}

inline AnfNodeConfigPtr FuncGraphSpecializer::MakeConfig(const AnfNodePtr &node) {
  return engine_->MakeConfig(node, context_, func_graph_);  // `func_graph_` is dummy here.
}
}  // namespace abstract
}  // namespace mindspore
