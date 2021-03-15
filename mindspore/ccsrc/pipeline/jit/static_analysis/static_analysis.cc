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

#include "pipeline/jit/static_analysis/static_analysis.h"

#include <algorithm>
#include <set>

#include "abstract/utils.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "frontend/operator/ops.h"
#include "utils/symbolic.h"
#include "ir/tensor.h"
#include "ir/func_graph_cloner.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/static_analysis/evaluator.h"
#include "debug/trace.h"
#include "debug/anf_ir_dump.h"

namespace mindspore {
namespace abstract {
bool IsIntermediateAbstract(const AbstractBasePtr &arg_spec) {
  if (dyn_cast<AbstractScalar>(arg_spec)) {
    auto v = arg_spec->GetValueTrack();
    if (v->isa<SymbolicKeyInstance>()) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

AbstractBasePtr IntermediateJoin(const AbstractBasePtr &arg1, const AbstractBasePtr &arg2) {
  if (dyn_cast<AbstractScalar>(arg1) && dyn_cast<AbstractScalar>(arg2)) {
    return arg1->Join(arg2);
  }
  return nullptr;
}

void AnalysisCache::set_value(const AnfNodeConfigPtr &conf, const EvalResultPtr &result) {
  MS_LOG(DEBUG) << "AnalysisCache set for NodeConfig: " << conf->node()->DebugString()
                << ", Context: " << conf->context()->ToString() << ", Value: " << result->abstract()->ToString()
                << ", Pointer: " << result->abstract().get();
  analysis_cache_map_[conf] = result;

  // Set intermediate abstract value.
  if (IsIntermediateAbstract(result->abstract())) {
    if (conf->node()->intermediate_abstract() == nullptr) {
      conf->node()->set_intermediate_abstract(result->abstract());
      MS_LOG(DEBUG) << "Set intermediate abstract: " << result->abstract()->ToString();
    } else {
      auto old_spec = conf->node()->intermediate_abstract();
      auto joined_spec = IntermediateJoin(result->abstract(), old_spec);
      conf->node()->set_intermediate_abstract(joined_spec);
      MS_LOG(DEBUG) << "Set joined intermediate abstract:\nold_spec:\t\t" << old_spec->ToString() << "\nnew_spec:\t\t"
                    << result->abstract()->ToString() << "\njoined_spec:\t"
                    << (joined_spec != nullptr ? joined_spec->ToString() : "nullptr");
    }
  }
}

EvalResultPtr AnalysisCache::GetValue(const AnfNodeConfigPtr &conf) {
  auto value = analysis_cache_map_.find(conf);
  if (value == analysis_cache_map_.end()) {
    return nullptr;
  }
  return value->second;
}

std::size_t AnfNodeConfigHasher::operator()(const AnfNodeConfigPtr conf) const {
  MS_EXCEPTION_IF_NULL(conf);
  MS_EXCEPTION_IF_NULL(conf->node());
  std::size_t hash_value = conf->node()->hash();
  if (!conf->context()->IsDummyContext()) {
    hash_value = hash_combine(hash_value, std::hash<AnalysisContext *>{}(conf->context().get()));
  }
  return hash_value;
}

bool AnfNodeConfigEqual::operator()(const AnfNodeConfigPtr lhs, const AnfNodeConfigPtr rhs) const {
  if (lhs == nullptr || rhs == nullptr) {
    return false;
  }
  if (lhs == rhs) {
    return true;
  }
  return (*lhs == *rhs);
}

AnalysisResult AnalysisEngine::Run(const FuncGraphPtr &func_graph, const AbstractBasePtrList &args_spec_list) {
  ConfigPtrList args_conf_list;
  (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(args_conf_list),
                       [](const AbstractBasePtr &arg) -> ConfigPtr { return std::make_shared<VirtualConfig>(arg); });
  MS_EXCEPTION_IF_NULL(func_graph_manager_);
  func_graph_manager_->AddFuncGraph(func_graph);

  AnalysisContextPtr empty_context = AnalysisContext::DummyContext();

  // Running the analyzer.
  ResetFunctionCallDepth();
  AnalysisContextPtr root_context = Run(func_graph, empty_context, args_conf_list);
  MS_EXCEPTION_IF_NULL(root_context);
  MS_EXCEPTION_IF_NULL(root_context->func_graph());
  AnfNodeConfigPtr output_conf = MakeConfig(root_context->func_graph()->get_return(), root_context);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << func_graph->ToString() << ": Run finished.";

  AnalysisResult result;
  MS_EXCEPTION_IF_NULL(output_conf);
  result.inferred = output_conf->ObtainEvalResult();
  result.context = root_context;
  return result;
}

AnalysisContextPtr AnalysisEngine::Run(const FuncGraphPtr &func_graph, const AnalysisContextPtr &context,
                                       const ConfigPtrList &args_conf_list) {
  std::shared_ptr<FuncGraphEvaluator> eval = std::make_shared<FuncGraphEvaluator>(func_graph, context);
  (void)eval->Run(shared_from_this(), args_conf_list, nullptr);
  return eval->graph_context();
}

EvalResultPtr AnalysisEngine::ObtainEvalResultWithCache(const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  EvalResultPtr result = analysis_cache_.GetValue(conf);
  if (result != nullptr) {
    MS_LOG(DEBUG) << "Evaluate cache hit for NodeConfig: " << conf->ToString()
                  << ", Value: " << result->abstract().get() << ", " << result->abstract()->ToString();
    return result;
  }

  MS_LOG(DEBUG) << "Evaluate cache miss for NodeConfig: " << conf->ToString();
  result = Eval(conf);
  if (result == nullptr) {
    MS_LOG(EXCEPTION) << "Evaluate for NodeConfig " << conf->ToString() << " get nullptr";
  }
  MS_LOG(DEBUG) << "Evaluate node on demond for NodeConfig: " << conf->ToString()
                << ", result: " << result->abstract().get() << ", " << result->abstract()->ToString();
  analysis_cache_.set_value(conf, result);
  return result;
}

EvalResultPtr AnalysisEngine::Eval(const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  AnfNodePtr node = conf->node();
  EvalResultPtr eval_result = nullptr;
#ifdef DEBUG
  compute_conf_stack_.push_back(node);
  std::ostringstream buffer;
  buffer << "Compute Config Begin:";
  for (auto iter : compute_conf_stack_) {
    buffer << " -> " << iter->DebugString();
  }
  MS_LOG(DEBUG) << buffer.str();
#endif
  MS_LOG(DEBUG) << "Begin Eval NodeConfig " << conf->ToString();
  MS_EXCEPTION_IF_NULL(node);
  if (node->abstract() != nullptr) {
    MS_LOG(DEBUG) << "Return old abstract: " << node->DebugString();
    eval_result = std::make_shared<EvalResult>(node->abstract(), std::make_shared<AttrValueMap>());
  } else if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    auto abstract = EvalValueNode(value_node, conf);
    eval_result = std::make_shared<EvalResult>(abstract, std::make_shared<AttrValueMap>());
  } else if (node->isa<CNode>()) {
    CheckNoStackInSameFuncGraph(conf);
    auto cnode = node->cast<CNodePtr>();
    trace::TraceEvalCNodeEnter(conf);
    eval_result = EvalCNode(cnode, conf);
    trace::TraceEvalCNodeLeave();
  } else {
    MS_LOG(EXCEPTION) << "Illegal AnfNode for evaluating, node: " << node->DebugString()
                      << ", fg: " << (node->func_graph() != nullptr ? node->func_graph()->ToString() : "nullgraph")
                      << ". NodeInfo: " << trace::GetDebugInfo(node->debug_info());
  }

#ifdef DEBUG
  compute_conf_stack_.pop_back();
  if (eval_result == nullptr) {
    MS_LOG(EXCEPTION) << "Compute Config failed, node: " << node->DebugString()
                      << " NodeInfo: " << trace::GetDebugInfo(node->debug_info());
  }
#endif
  MS_LOG(DEBUG) << "End Eval NodeConfig " << conf->ToString() << ", res: " << eval_result->abstract()->ToString();
  return eval_result;
}

void AnalysisEngine::CheckNoStackInSameFuncGraph(const AnfNodeConfigPtr &conf) {
  auto &list = trace::GetCNodeDebugStack();
  if (list.empty()) {
    return;
  }
  auto &previous_stack = list.back();
  auto previous_cnode_fg = previous_stack->node()->func_graph();
  auto current_cnode_fg = conf->node()->func_graph();
  if (previous_cnode_fg != current_cnode_fg) {  // Normal.
    return;
  }
  if (forward_count_ != 0) {  // Ignore Forward Config.
    return;
  }
  auto &infer_stack = trace::GetCurrenGraphEvalStack();
  if (infer_stack.empty()) {
    return;
  }
  auto top_evaluator = infer_stack.top().first;
  if (!top_evaluator->isa<BaseFuncGraphEvaluator>()) {
    MS_LOG(EXCEPTION) << "Top evaluator is " << top_evaluator->ToString();
  }
  auto top_fg_evaluator = dyn_cast<BaseFuncGraphEvaluator>(top_evaluator);
  auto top_context_fg = top_fg_evaluator->graph_context()->func_graph();
  if (current_cnode_fg != top_context_fg) {  // Ignore FV call.
    return;
  }
  MS_LOG(ERROR) << "Should not use call stack in the same function: " << top_context_fg->ToString() << ", for "
                << conf->node()->DebugString(2);
  for (size_t i = 0; i < list.size(); ++i) {
    auto old_conf = list[i];
    MS_LOG(ERROR) << "  #" << i << ": " << old_conf->node()->DebugString(2) << ", in "
                  << old_conf->context()->func_graph()->ToString();
  }
  DumpIR("use_stack_error.ir", conf->node()->func_graph());
  MS_LOG(EXCEPTION) << "To check above CNode stack and dumped use_stack_error.ir";
}

AbstractBasePtr AnalysisEngine::EvalValueNode(const ValueNodePtr &value_node, const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  MS_EXCEPTION_IF_NULL(value_node);
  auto out = ToAbstract(value_node->value(), conf->context(), conf);
  if (value_node->has_new_value() && out->isa<AbstractTensor>()) {
    out = out->Broaden();
  }
  return out;
}

EvalResultPtr AnalysisEngine::EvalCNode(const CNodePtr &cnode, const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  MS_EXCEPTION_IF_NULL(cnode);
  auto &inputs = cnode->inputs();
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "CNode->inputs() is empty, CNode: " << cnode->DebugString();
  }

  AnfNodePtr func_node = inputs[0];
  MS_EXCEPTION_IF_NULL(func_node);
  MS_LOG(DEBUG) << "Current CNode function: " << func_node->DebugString();
  AnalysisContextPtr context = conf->context();
  AnfNodeConfigPtr func_conf = MakeConfig(func_node, context);
  MS_EXCEPTION_IF_NULL(func_conf);
  // Keep it in a local variable, otherwise smart pointer will free it.
  auto maybe_func_eval_result = func_conf->ObtainEvalResult();
  AbstractBasePtr maybe_func = maybe_func_eval_result->abstract();
  if (maybe_func == nullptr) {
    MS_LOG(EXCEPTION) << "No abstract, func_conf: " << func_conf->ToString()
                      << " NodeInfo: " << trace::GetDebugInfo(cnode->debug_info());
  }
  if (maybe_func->BuildType()->type_id() == kObjectTypeUndeterminedType) {
    MS_LOG(DEBUG) << "EvalCNode eval Undetermined";
    return std::make_shared<EvalResult>(maybe_func->Clone(), std::make_shared<AttrValueMap>());
  }
  AbstractFunctionPtr func = dyn_cast<AbstractFunction>(maybe_func);
  if (func == nullptr) {
    MS_LOG(EXCEPTION) << "Not AbstractFunction: " << maybe_func->ToString() << ", func_conf: " << func_conf->ToString()
                      << " NodeInfo: " << trace::GetDebugInfo(cnode->debug_info());
  }

  ConfigPtrList args_conf_list;
  // ignore the first node which is function name
  for (std::size_t i = 1; i < inputs.size(); i++) {
    const AnfNodePtr &node = inputs[i];
    args_conf_list.push_back(MakeConfig(node, context));
  }
  std::vector<EvaluatorPtr> infs;

  auto build_evaluator = [this, &infs, &cnode](const AbstractFuncAtomPtr &poss) {
    auto evaluator = this->GetEvaluatorFor(poss);
    evaluator->set_bound_node(cnode);
    infs.push_back(evaluator);
  };
  func->Visit(build_evaluator);

  auto eval_result = ExecuteEvaluators(infs, conf, args_conf_list);
  return eval_result;
}

EvalResultPtr AnalysisEngine::Execute(const AbstractFunctionPtr &func, const AbstractBasePtrList &args_spec_list) {
  ConfigPtrList args_conf_list;
  (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(args_conf_list),
                       [](const AbstractBasePtr &arg) -> ConfigPtr { return std::make_shared<VirtualConfig>(arg); });
  std::vector<EvaluatorPtr> infs;
  MS_EXCEPTION_IF_NULL(func);
  auto build_evaluator = [this, &infs](const AbstractFuncAtomPtr &poss) {
    auto evaluator = this->GetEvaluatorFor(poss);
    infs.push_back(evaluator);
  };
  func->Visit(build_evaluator);
  return ExecuteEvaluators(infs, nullptr, args_conf_list);
}

void AnalysisEngine::ClearEvaluatorCache() {
  for (std::pair<AbstractFunctionPtr, EvaluatorPtr> element : constructors_) {
    EvaluatorPtr evaluator = element.second;
    MS_EXCEPTION_IF_NULL(evaluator);
    MS_EXCEPTION_IF_NULL(evaluator->evaluator_cache_map());
    evaluator->evaluator_cache_map()->clear();
  }
  for (auto &element : prim_constructors_) {
    EvaluatorPtr evaluator = element.second;
    MS_EXCEPTION_IF_NULL(evaluator);
    MS_EXCEPTION_IF_NULL(evaluator->evaluator_cache_map());
    evaluator->evaluator_cache_map()->clear();
  }
  for (auto &element : prim_py_evaluators_) {
    EvaluatorPtr evaluator = element.second;
    MS_EXCEPTION_IF_NULL(evaluator);
    MS_EXCEPTION_IF_NULL(evaluator->evaluator_cache_map());
    evaluator->evaluator_cache_map()->clear();
  }
}

void AnalysisEngine::Clear() {
  analysis_cache_.Clear();
  anfnode_config_map_.clear();
  eval_trace_.clear();
  constructors_.clear();
  constructors_app_.clear();
  continued_evals_.clear();
}

namespace {
EvaluatorPtr GetPrimEvaluator(const PrimitivePtr &prim, const AnalysisEnginePtr &engine) {
  // Custom Primitive with python infer_shape, infer_type
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->isa<prim::DoSignaturePrimitive>()) {
    return std::make_shared<DoSignatureEvaluator>(prim);
  }
  if (prim->isa<prim::UnpackGraphPrimitive>()) {
    return std::make_shared<UnpackGraphEvaluator>(prim);
  }
  if (prim->Hash() == prim::kPrimMixedPrecisionCast->Hash() && prim->name() == prim::kPrimMixedPrecisionCast->name()) {
    return std::make_shared<MixedPrecisionCastEvaluator>(prim);
  }

  // find prim infer function in the prim function map return a standard evaluator
  StandardPrimitiveEvalImpl eval_impl = GetPrimitiveInferImpl(prim);
  if (eval_impl != nullptr) {
    return std::make_shared<StandardPrimEvaluator>(prim, eval_impl);
  }

  // use python infer function if the infer function not founded in the map return a python evaluator
  EvaluatorPtr evaluator = nullptr;
  if (prim->HasPyEvaluator()) {
    auto prim_py = dyn_cast<PrimitivePy>(prim);
    if (prim_py != nullptr) {
      if (engine == nullptr) {
        return std::make_shared<PythonPrimEvaluator>(prim_py);
      }

      const auto &iter = engine->prim_py_evaluators_.find(prim_py);
      if (iter != engine->prim_py_evaluators_.end()) {
        return iter->second;
      }
      evaluator = std::make_shared<PythonPrimEvaluator>(prim_py);
      engine->prim_py_evaluators_[prim_py] = evaluator;
      return evaluator;
    }
    MS_LOG(EXCEPTION) << "The primitive with python evaluator should be a python primitive.";
  }

  // return a default evaluator
  if (engine == nullptr) {
    // If engine is nullptr, get constructor from default.
    const PrimEvaluatorMap &prim_evaluator_map = GetPrimEvaluatorConstructors();
    auto iter = prim_evaluator_map.find(prim);
    if (iter != prim_evaluator_map.end()) {
      evaluator = iter->second;
    }
  } else {
    // If engine is given, get constructor from engine resource.
    const PrimEvaluatorMap &prim_evaluator_map = engine->PrimConstructors();
    auto iter = prim_evaluator_map.find(prim);
    if (iter != prim_evaluator_map.end()) {
      evaluator = iter->second;
    }
  }
  if (evaluator == nullptr) {
    MS_LOG(EXCEPTION) << "The evaluator of the primitive is not defined (" << prim->name() << ").";
  }
  return evaluator;
}
}  // namespace

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<PrimitiveAbstractClosure> &func) {
  auto inf_pair = constructors_.find(func);
  if (inf_pair != constructors_.end()) {
    return inf_pair->second;
  }
  MS_EXCEPTION_IF_NULL(func);
  auto primitive = func->prim();
  auto evaluator = GetPrimEvaluator(primitive, shared_from_this());
  constructors_[func] = evaluator;
  return evaluator;
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<FuncGraphAbstractClosure> &func) {
  auto inf_pair = constructors_.find(func);
  if (inf_pair != constructors_.end()) {
    return inf_pair->second;
  }
  MS_EXCEPTION_IF_NULL(func);
  std::shared_ptr<FuncGraphEvaluator> func_graph_evaluator =
    std::make_shared<FuncGraphEvaluator>(func->func_graph(), func->context());
  constructors_[func] = func_graph_evaluator;
  return func_graph_evaluator;
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<MetaFuncGraphAbstractClosure> &func) {
  auto inf_pair = constructors_.find(func);
  if (inf_pair != constructors_.end()) {
    return inf_pair->second;
  }
  MS_EXCEPTION_IF_NULL(func);
  std::shared_ptr<MetaFuncGraphEvaluator> evaluator =
    std::make_shared<MetaFuncGraphEvaluator>(func->meta_func_graph(), func->context(), func->GetScope());
  constructors_[func] = evaluator;
  return evaluator;
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<JTransformedAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  AbstractFunctionPtr func_orig = func->fn();
  EvaluatorPtr evaluator_orig = GetEvaluatorFor(func_orig);
  auto jevaluator = std::make_shared<JEvaluator>(evaluator_orig, func_orig);
  return jevaluator;
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<VirtualAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  std::shared_ptr<VirtualEvaluator> virtual_evaluator =
    std::make_shared<VirtualEvaluator>(func->args_spec_list(), func->output());
  return virtual_evaluator;
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<PartialAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  AbstractFunctionPtr func_orig = func->fn();
  EvaluatorPtr evaluator_orig = GetEvaluatorFor(func_orig);
  auto part_pair = std::make_pair(func_orig, func->args());
  auto itr = constructors_app_.find(part_pair);
  if (itr != constructors_app_.end()) {
    return itr->second;
  }
  std::shared_ptr<PartialAppEvaluator> partial_evaluator =
    std::make_shared<PartialAppEvaluator>(evaluator_orig, func->args());
  constructors_app_[part_pair] = partial_evaluator;
  return partial_evaluator;
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<TypedPrimitiveAbstractClosure> &) {
  MS_LOG(EXCEPTION) << "Should not be called ";
}

// Forward to specific subclass of FunctionWrapper.
EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const AbstractFunctionPtr &func) {
  MS_EXCEPTION_IF_NULL(func);
  if (func->isa<PrimitiveAbstractClosure>()) {
    return _GetEvaluatorFor(func->cast<std::shared_ptr<PrimitiveAbstractClosure>>());
  } else if (func->isa<FuncGraphAbstractClosure>()) {
    return _GetEvaluatorFor(func->cast<std::shared_ptr<FuncGraphAbstractClosure>>());
  } else if (func->isa<MetaFuncGraphAbstractClosure>()) {
    return _GetEvaluatorFor(func->cast<std::shared_ptr<MetaFuncGraphAbstractClosure>>());
  } else if (func->isa<JTransformedAbstractClosure>()) {
    return _GetEvaluatorFor(func->cast<std::shared_ptr<JTransformedAbstractClosure>>());
  } else if (func->isa<VirtualAbstractClosure>()) {
    return _GetEvaluatorFor(func->cast<std::shared_ptr<VirtualAbstractClosure>>());
  } else if (func->isa<PartialAbstractClosure>()) {
    return _GetEvaluatorFor(func->cast<std::shared_ptr<PartialAbstractClosure>>());
  } else if (func->isa<TypedPrimitiveAbstractClosure>()) {
    return _GetEvaluatorFor(func->cast<std::shared_ptr<TypedPrimitiveAbstractClosure>>());
  } else if (func->isa<AbstractFuncAtom>()) {
    MS_LOG(EXCEPTION) << "Cannot GetEvaluator from AbstractFuncAtom";
  } else if (func->isa<AbstractFuncUnion>()) {
    MS_LOG(EXCEPTION) << "Cannot GetEvaluator from AbstractFuncUnion";
  } else if (func->isa<DummyAbstractClosure>()) {
    MS_LOG(EXCEPTION) << "A dummy function cannot eval";
  } else {
    MS_LOG(EXCEPTION) << "Cannot GetEvaluator from AbstractFunction";
  }
}

EvaluatorPtr AnalysisEngine::GetEvaluatorFor(const AbstractFunctionPtr &func) {
  MS_LOG(DEBUG) << "The func value: " << func->ToString();
  if (func->tracking_id() != nullptr) {
    MS_LOG(DEBUG) << "The tracking_id: " << func->tracking_id()->DebugString();
  }
  MS_EXCEPTION_IF_NULL(func);
  if (func->tracking_id() == nullptr || func->isa<abstract::MetaFuncGraphAbstractClosure>() ||
      func->isa<abstract::FuncGraphAbstractClosure>()) {
    EvaluatorPtr evaluator = _GetEvaluatorFor(func);
    return evaluator;
  }
  auto inf_pair = constructors_.find(func);
  if (inf_pair != constructors_.end()) {
    return inf_pair->second;
  }

  AbstractFunctionPtr func_generic = func->Copy();
  func_generic->set_tracking_id(nullptr);
  EvaluatorPtr eval = _GetEvaluatorFor(func_generic);
  auto tracked_eval = std::make_shared<TrackedEvaluator>(eval);
  constructors_[func] = tracked_eval;

  return tracked_eval;
}

EvalResultPtr AnalysisEngine::ForwardConfig(const AnfNodeConfigPtr &orig_conf, const AnfNodeConfigPtr new_conf) {
  // Use anfnode_config_map_[orig_conf] = new_conf will require AnfNodeConfig provide copy constructor.
  (void)anfnode_config_map_.emplace(orig_conf, new_conf);
  MS_LOG(DEBUG) << "Forward orig_conf: " << orig_conf->node()->DebugString()
                << ", to new_conf: " << new_conf->node()->DebugString();
  if (orig_conf->node()->isa<CNode>()) {
    auto old_cnode = orig_conf->node()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(old_cnode);
    if (new_conf->node()->isa<CNode>()) {
      auto new_cnode = new_conf->node()->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(new_cnode);
      if (old_cnode->func_graph() == new_cnode->func_graph()) {
        MS_LOG(DEBUG) << "Try to remove forward node from order list, forward node: " << new_cnode->ToString()
                      << ", as origin node should be in order list, origin_node: " << old_cnode->ToString();
        old_cnode->func_graph()->EraseUnusedNodeInOrder(new_cnode);
      } else {
        MS_LOG(EXCEPTION) << "Forward orig_node to different func_graph, old_node: " << old_cnode->DebugString()
                          << ", new_node: " << new_cnode->DebugString();
      }
    }
  }
  forward_count_++;
  auto res = ObtainEvalResultWithCache(new_conf);
  forward_count_--;
  return res;
}

EvalResultPtr AnalysisEngine::ExecuteEvaluators(const std::vector<EvaluatorPtr> &evaluators,
                                                const AnfNodeConfigPtr &out_conf, const ConfigPtrList &args_conf_list) {
  if (evaluators.size() == 1) {
    EvaluatorPtr eval = evaluators[0];
    MS_EXCEPTION_IF_NULL(eval);
    return eval->Run(shared_from_this(), args_conf_list, out_conf);
  }
  return ExecuteMultipleEvaluators(evaluators, out_conf, args_conf_list);
}

void AnalysisEngine::SetUndeterminedFlag(const EvaluatorPtr &evaluator) {
  auto fg_eval = evaluator->cast<FuncGraphEvaluatorPtr>();
  if (fg_eval == nullptr) {
    return;
  }

  auto fg = fg_eval->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto undetermined_fgs = fg->recursive();
  if (undetermined_fgs) {
    auto fg_parent = fg->parent();
    MS_EXCEPTION_IF_NULL(fg_parent);
    fg_parent->set_flag(kFuncGraphFlagUndetermined, true);
    MS_LOG(DEBUG) << "Set graph undetermined: " << fg_parent->ToString();
  }
}

EvaluatorPtr AnalysisEngine::HandleNestedRecursion(const std::vector<EvaluatorPtr> &evaluators,
                                                   const EvaluatorPtr &eval, const AbstractBasePtrList &args_spec_list,
                                                   const EvalTraceRevIter &it, bool *continue_flag) {
  *continue_flag = false;
  // Find latest entry function to handle nested recursion.
  EvaluatorPtr latest_entry = eval;
  auto latest_entry_iter = eval_trace_.rbegin();
  for (auto r_it = eval_trace_.rbegin(); *r_it != *it;) {
    auto it_temp = std::find(evaluators.begin(), evaluators.end(), r_it->evaluator_);
    if (it_temp != evaluators.end()) {
      latest_entry = *it_temp;
      latest_entry_iter = r_it;
      break;
    }
    latest_entry_iter = ++r_it;
  }
  if (latest_entry != eval) {
    MS_LOG(DEBUG) << "Continue Evaluator " << eval->ToString();
    *continue_flag = true;
    return latest_entry;
  }

  bool has_undetermined = false;
  // Check whether sub loop has untraced undetermined evaluator.
  std::unordered_set<EvaluatorArgs, EvaluatorArgsHasher, EvaluatorArgsEqual> undetermined_evals;
  for (auto r_it = eval_trace_.rbegin(); r_it != latest_entry_iter; r_it++) {
    undetermined_evals.insert(*r_it);
  }
  MS_LOG(DEBUG) << "undetermined_evals size(): " << undetermined_evals.size();

  for (auto u_eval : undetermined_evals) {
    MS_LOG(DEBUG) << u_eval.evaluator_->ToString() << "check undetermined.";
    auto &alternate_evaluator = multi_poss_[u_eval.evaluator_];
    auto &eval_cache = alternate_evaluator->evaluator_cache_map();
    const auto &alt_eval_args = EvaluatorArgs(alternate_evaluator, args_spec_list);
    if ((!undetermined_evals.count(alt_eval_args)) &&
        (((!continued_evals_.count(u_eval)) && (eval_cache->find(args_spec_list) != eval_cache->end())) ||
         (eval_cache->find(args_spec_list) == eval_cache->end()))) {
      MS_LOG(DEBUG) << u_eval.evaluator_->ToString() << "has undetermined.";
      has_undetermined = true;
      break;
    }
  }
  if (has_undetermined == false) {
    MS_LOG(DEBUG) << eval->ToString() << "has no undetermined.";
    *continue_flag = true;
    return latest_entry;
  }

  return latest_entry;
}

EvalResultPtr AnalysisEngine::ProcessEvalResults(const AbstractBasePtrList &out_specs) {
  if (out_specs.size() == 0) {
    MS_LOG(EXCEPTION) << "There is an endless loop for evaluator.";
  }

  if (out_specs.size() == 1) {
    MS_EXCEPTION_IF_NULL(out_specs[0]);
    // If only one result derived, then broaden it to avoid wrong constant propagation.
    return std::make_shared<EvalResult>(out_specs[0]->Broaden(), std::make_shared<AttrValueMap>());
  }
  auto joined_spec = AbstractJoin(out_specs);
  MS_EXCEPTION_IF_NULL(joined_spec);
  MS_LOG(DEBUG) << "Multiple evaluators joined: " << joined_spec->ToString();
  return std::make_shared<EvalResult>(joined_spec, std::make_shared<AttrValueMap>());
}

EvalResultPtr AnalysisEngine::ExecuteMultipleEvaluators(const std::vector<EvaluatorPtr> &evaluators,
                                                        const AnfNodeConfigPtr &out_conf,
                                                        const ConfigPtrList &args_conf_list) {
  AbstractBasePtrList out_specs;
  multi_poss_[evaluators[0]] = evaluators[1];
  multi_poss_[evaluators[1]] = evaluators[0];
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->ObtainEvalResult()->abstract();
                       });
  for (auto eval : evaluators) {
    SetUndeterminedFlag(eval);

    const auto current_inf = EvaluatorArgs(eval, args_spec_list);
    MS_LOG(DEBUG) << "Check Evaluator " << eval->ToString();
    // If current evaluator is under tracing, then skip current evaluator to avoid recursively evaluating.
    auto it = std::find(eval_trace_.rbegin(), eval_trace_.rend(), current_inf);
    if (it == eval_trace_.rend()) {
      eval_trace_.push_back(current_inf);
      MS_EXCEPTION_IF_NULL(eval);
      auto eval_result = eval->Run(shared_from_this(), args_conf_list, out_conf);
      MS_EXCEPTION_IF_NULL(eval_result->abstract());
      out_specs.push_back(eval_result->abstract());
      eval_trace_.pop_back();
      if (eval_trace_.empty()) {
        multi_poss_.clear();
      }
    } else {
      bool continue_flag = false;
      auto latest_entry = HandleNestedRecursion(evaluators, eval, args_spec_list, it, &continue_flag);
      if (continue_flag) {
        MS_LOG(DEBUG) << "continued_evals_ add " << current_inf.evaluator_.get() << current_inf.evaluator_->ToString();
        continued_evals_.insert(current_inf);
        continue;
      }

      // Try to travel the latest undetermined.
      if (latest_entry != eval_trace_.rbegin()->evaluator_) {
        MS_LOG(DEBUG) << "Direct Run Evaluator " << eval.get() << "----" << eval->ToString();
        auto eval_result = latest_entry->Run(shared_from_this(), args_conf_list, out_conf);
        MS_EXCEPTION_IF_NULL(eval_result->abstract());
        MS_LOG(DEBUG) << "end Direct Evaluator " << latest_entry->ToString()
                      << " return out_spec: " << eval_result->abstract()->ToString();
        return eval_result;
      }
    }
  }

  return ProcessEvalResults(out_specs);
}

EvalResultPtr AnfNodeConfig::ObtainEvalResult() {
  AnfNodeConfigPtr self = shared_from_base<AnfNodeConfig>();
  return engine_.lock()->ObtainEvalResultWithCache(self);
}

abstract::AbstractBasePtr MakeAbstractClosure(const FuncGraphPtr &func_graph,
                                              const abstract::AnalysisContextPtr &context, const AnfNodePtr &anf_node) {
  AnalysisContextPtr temp_context = context;
  if (temp_context == nullptr) {
    temp_context = abstract::AnalysisContext::DummyContext();
  }
  return std::make_shared<abstract::FuncGraphAbstractClosure>(func_graph, temp_context, anf_node);
}

abstract::AbstractBasePtr MakeAbstractClosure(const MetaFuncGraphPtr &meta_func_graph, const AnfNodePtr &anf_node) {
  abstract::MetaFuncGraphAbstractClosurePtr meta_func_graph_fn;
  if (anf_node == nullptr) {
    meta_func_graph_fn = std::make_shared<abstract::MetaFuncGraphAbstractClosure>(meta_func_graph);
  } else {
    meta_func_graph_fn =
      std::make_shared<abstract::MetaFuncGraphAbstractClosure>(meta_func_graph, anf_node, anf_node->scope());
  }
  return meta_func_graph_fn;
}

abstract::AbstractBasePtr MakeAbstractClosure(const PrimitivePtr &primitive, const AnfNodePtr &anf_node) {
  auto prim_func = std::make_shared<abstract::PrimitiveAbstractClosure>(primitive, anf_node);
  return prim_func;
}

AbstractBasePtr ToAbstract(const ValuePtr &value, const AnalysisContextPtr &context, const AnfNodeConfigPtr &conf) {
  AnfNodePtr anf_node = nullptr;
  if (conf != nullptr) {
    anf_node = conf->node();
  }
  if (value->isa<FuncGraph>()) {
    auto func_graph = value->cast<FuncGraphPtr>();
    return MakeAbstractClosure(func_graph, context, anf_node);
  }
  if (value->isa<MetaFuncGraph>()) {
    auto meta_func_graph = value->cast<MetaFuncGraphPtr>();
    return MakeAbstractClosure(meta_func_graph, anf_node);
  }
  if (value->isa<Primitive>()) {
    auto prim = value->cast<PrimitivePtr>();
    return MakeAbstractClosure(prim, anf_node);
  }
  return value->ToAbstract();
}

AbstractBasePtr FromValueInside(const ValuePtr &value, bool broaden) {
  AbstractBasePtr a = ToAbstract(value, nullptr, nullptr);
  if (broaden) {
    a = a->Broaden();
  }
  return a;
}

EvalResultPtr EvalOnePrim(const PrimitivePtr &primitive, const AbstractBasePtrList &arg_specs) {
  auto evaluator = GetPrimEvaluator(primitive, nullptr);
  MS_EXCEPTION_IF_NULL(evaluator);
  if (!evaluator->isa<TrivialPrimEvaluator>()) {
    MS_LOG(EXCEPTION) << "Prim " << primitive->ToString() << " should build a TrivialPrimEvaluator, but "
                      << evaluator->ToString();
  }
  auto trivial_evaluator = dyn_cast<TrivialPrimEvaluator>(evaluator);
  auto eval_result = trivial_evaluator->EvalPrim(nullptr, arg_specs);
  return eval_result;
}
}  // namespace abstract
}  // namespace mindspore
