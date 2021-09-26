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
#include "abstract/abstract_value.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "frontend/operator/ops.h"
#include "utils/symbolic.h"
#include "utils/ms_exception.h"
#include "ir/tensor.h"
#include "ir/func_graph_cloner.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/static_analysis/evaluator.h"
#include "debug/trace.h"
#include "debug/anf_ir_dump.h"
#include "pipeline/jit/static_analysis/async_eval_result.h"

namespace mindspore {
namespace abstract {
// Record current depth of function call stack, including `stack_frame_depth`.
thread_local size_t function_call_depth;
thread_local size_t function_call_max_depth;
// Record current depth of stack frames call.
thread_local size_t stack_frame_depth;
thread_local size_t stack_frame_max_depth;

void ResetFunctionCallDepth() {
  function_call_depth = 0;
  function_call_max_depth = 0;
}
void IncreaseFunctionCallDepth() {
  function_call_depth++;
  if (function_call_max_depth < function_call_depth) {
    function_call_max_depth = function_call_depth;
  }
}
void DecreaseFunctionCallDepth() {
  if (function_call_depth == 0) {
    MS_LOG(EXCEPTION) << "Current function call depth is already 0, can not decrease it.";
  }
  function_call_depth--;
}
size_t FunctionCallDepth() { return function_call_depth; }
size_t FunctionCallMaxDepth() { return function_call_max_depth; }

void ResetStackFrameDepth() {
  stack_frame_depth = 0;
  stack_frame_max_depth = 0;
}
void IncreaseStackFrameDepth() {
  stack_frame_depth++;
  if (stack_frame_max_depth < stack_frame_depth) {
    stack_frame_max_depth = stack_frame_depth;
  }
}
void DecreaseStackFrameDepth() {
  if (stack_frame_depth == 0) {
    MS_LOG(EXCEPTION) << "Current stack frame depth is already 0, can not decrease it.";
  }
  stack_frame_depth--;
}
size_t StackFrameDepth() { return stack_frame_depth; }
size_t StackFrameMaxDepth() { return stack_frame_max_depth; }

bool IsIntermediateAbstract(const AbstractBasePtr &arg_spec) {
  MS_EXCEPTION_IF_NULL(arg_spec);
  if (dyn_cast<AbstractScalar>(arg_spec)) {
    auto v = arg_spec->GetValueTrack();
    if (v->isa<SymbolicKeyInstance>()) {
      return true;
    }
  }
  return false;
}

AbstractBasePtr IntermediateJoin(const AbstractBasePtr &arg1, const AbstractBasePtr &arg2) {
  if (dyn_cast<AbstractScalar>(arg1) && dyn_cast<AbstractScalar>(arg2)) {
    MS_EXCEPTION_IF_NULL(arg1);
    return arg1->Join(arg2);
  }
  return nullptr;
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
  StaticAnalysisException::Instance().ClearException();
  AnalysisResult result;
  try {
    MS_EXCEPTION_IF_NULL(func_graph);
    ConfigPtrList args_conf_list;
    (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(args_conf_list),
                         [](const AbstractBasePtr &arg) -> ConfigPtr { return std::make_shared<VirtualConfig>(arg); });
    MS_EXCEPTION_IF_NULL(func_graph_manager_);
    func_graph_manager_->AddFuncGraph(func_graph);
    root_func_graph_ = func_graph;

    // Running the analyzer.
    ResetFunctionCallDepth();
    ResetStackFrameDepth();
    AnalysisContextPtr dummy_context = AnalysisContext::DummyContext();
    AnalysisContextPtr root_context = Run(func_graph, dummy_context, args_conf_list);
    MS_EXCEPTION_IF_NULL(root_context);
    auto root_context_fg = root_context->func_graph();
    MS_EXCEPTION_IF_NULL(root_context_fg);
    AnfNodeConfigPtr output_conf = MakeConfig(root_context_fg->get_return(), root_context, root_context_fg);
    MS_EXCEPTION_IF_NULL(func_graph);
    MS_LOG(INFO) << func_graph->ToString() << ": Run finished.";

    MS_EXCEPTION_IF_NULL(output_conf);
    result.inferred = output_conf->ObtainEvalResult();
    result.context = root_context;
  } catch (const std::exception &ex) {
    MS_LOG(INFO) << "Eval " << func_graph->ToString() << " threw exception.";
    AnalysisSchedule::GetInstance().HandleException(ex);
  }
  AnalysisSchedule::GetInstance().Wait();
  return result;
}

AnalysisContextPtr AnalysisEngine::Run(const FuncGraphPtr &func_graph, const AnalysisContextPtr &context,
                                       const ConfigPtrList &args_conf_list) {
  std::shared_ptr<FuncGraphEvaluator> eval = std::make_shared<FuncGraphEvaluator>(func_graph, context);
  (void)eval->Run(shared_from_this(), args_conf_list, nullptr);
  return root_context_;
}

void AnalysisEngine::SaveEvalResultInCache(const AnfNodeConfigPtr &conf, const EvalResultPtr &result) {
  MS_EXCEPTION_IF_NULL(conf);
  MS_EXCEPTION_IF_NULL(result);
  static AnalysisResultCacheMgr &cache_mgr = AnalysisResultCacheMgr::GetInstance();
  cache_mgr.SetValue(conf, result);

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

EvalResultPtr AnalysisEngine::ObtainEvalResultWithCache(const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  static AnalysisResultCacheMgr &cache_mgr = AnalysisResultCacheMgr::GetInstance();
  auto result = cache_mgr.GetValue(conf);
  if (result != nullptr) {
    return result;
  }
  MS_LOG(DEBUG) << "Evaluate cache miss for NodeConfig: " << conf->ToString();
  result = Eval(conf);
  if (result == nullptr) {
    MS_LOG(EXCEPTION) << "Evaluate for NodeConfig " << conf->ToString() << " get nullptr";
  }
  MS_LOG(DEBUG) << "Evaluate node on demond for NodeConfig: " << conf->ToString()
                << ", result: " << result->abstract().get() << ", " << result->abstract()->ToString();
  SaveEvalResultInCache(conf, result);
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
    auto cnode = node->cast<CNodePtr>();
    trace::TraceEvalCNodeEnter(conf);
    eval_result = EvalCNode(cnode, conf);
    trace::TraceEvalCNodeLeave();
  } else {
    MS_LOG(EXCEPTION) << "Illegal AnfNode for evaluating, node: " << node->DebugString() << "(" << node->type_name()
                      << "), fg: " << (node->func_graph() != nullptr ? node->func_graph()->ToString() : "nullgraph");
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

AbstractBasePtr AnalysisEngine::EvalValueNode(const ValueNodePtr &value_node, const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  MS_EXCEPTION_IF_NULL(value_node);
  auto out = ToAbstract(value_node->value(), conf->context(), conf);
  if (value_node->has_new_value() && out->isa<AbstractTensor>()) {
    out = out->Broaden();
  }
  return out;
}

AbstractBasePtr AnalysisEngine::GetCNodeOperatorAbstract(const CNodePtr &cnode, const AnalysisContextPtr &context,
                                                         const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto &inputs = cnode->inputs();
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "CNode->inputs() is empty, CNode: " << cnode->DebugString();
  }
  AnfNodePtr func_node = inputs[0];
  MS_EXCEPTION_IF_NULL(func_node);
  MS_LOG(DEBUG) << "Current CNode function: " << func_node->DebugString();
  AnfNodeConfigPtr func_conf = MakeConfig(func_node, context, func_graph);
  MS_EXCEPTION_IF_NULL(func_conf);
  // Keep it in a local variable, otherwise smart pointer will free it.
  auto possible_func_eval_result = func_conf->ObtainEvalResult();
  AbstractBasePtr possible_func = possible_func_eval_result->abstract();
  if (possible_func == nullptr) {
    MS_LOG(EXCEPTION) << "No abstract, func_conf: " << func_conf->ToString();
  }
  return possible_func;
}

EvalResultPtr AnalysisEngine::EvalCNode(const CNodePtr &cnode, const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  MS_EXCEPTION_IF_NULL(cnode);
  AbstractBasePtr possible_func = GetCNodeOperatorAbstract(cnode, conf->context(), conf->func_graph());
  if (possible_func->BuildType()->type_id() == kObjectTypeUndeterminedType) {
    MS_LOG(DEBUG) << "EvalCNode eval Undetermined";
    return std::make_shared<EvalResult>(possible_func->Clone(), std::make_shared<AttrValueMap>());
  }

  AbstractFunctionPtr func = dyn_cast<AbstractFunction>(possible_func);
  if (func == nullptr) {
    MS_LOG(ERROR) << "Can not cast to a AbstractFunction from " << possible_func->ToString() << ".";
    MS_LOG(ERROR) << "It's called at: " << cnode->DebugString();
    MS_EXCEPTION(ValueError) << "This may be not defined, or it can't be a operator. Please check code.";
  }

  ConfigPtrList args_conf_list;
  // Ignore the first node which is function name
  auto &inputs = cnode->inputs();
  for (std::size_t i = 1; i < inputs.size(); i++) {
    const AnfNodePtr &node = inputs[i];
    args_conf_list.push_back(MakeConfig(node, conf->context(), conf->func_graph()));
  }

  std::vector<EvaluatorPtr> evaluators;
  auto build_evaluator = [this, &evaluators, &cnode](const AbstractFuncAtomPtr &poss) {
    auto evaluator = this->GetEvaluatorFor(poss);
    evaluator->set_bound_node(cnode);
    evaluators.push_back(evaluator);
  };
  func->Visit(build_evaluator);

  auto eval_result = ExecuteEvaluators(evaluators, conf, args_conf_list);
  return eval_result;
}

EvalResultPtr AnalysisEngine::Execute(const AbstractFunctionPtr &func, const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(func);
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
  for (auto &element : evaluators_) {
    EvaluatorPtr evaluator = element.second;
    MS_EXCEPTION_IF_NULL(evaluator);
    MS_EXCEPTION_IF_NULL(evaluator->evaluator_cache_mgr());
    evaluator->evaluator_cache_mgr()->Clear();
  }
  for (auto &element : prim_constructors_) {
    EvaluatorPtr evaluator = element.second;
    MS_EXCEPTION_IF_NULL(evaluator);
    MS_EXCEPTION_IF_NULL(evaluator->evaluator_cache_mgr());
    evaluator->evaluator_cache_mgr()->Clear();
  }
  for (auto &element : prim_py_evaluators_) {
    EvaluatorPtr evaluator = element.second;
    MS_EXCEPTION_IF_NULL(evaluator);
    MS_EXCEPTION_IF_NULL(evaluator->evaluator_cache_mgr());
    evaluator->evaluator_cache_mgr()->Clear();
  }
  // Release Exception to avoid hup at exit.
  StaticAnalysisException::Instance().ClearException();
}

void AnalysisEngine::Clear() {
  AnalysisResultCacheMgr::GetInstance().Clear();
  anfnode_config_map_.clear();
  eval_trace_.clear();
  evaluators_.clear();
  constructors_app_.clear();
  continued_evals_.clear();
  root_func_graph_ = nullptr;
  root_context_ = nullptr;
}

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
  auto eval_impl = GetPrimitiveInferImpl(prim);
  if (eval_impl.infer_shape_impl_ != nullptr) {
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
    MS_LOG(ERROR) << "The primitive with python evaluator should be a python primitive.";
    return nullptr;
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
    MS_LOG(DEBUG) << "The evaluator of the primitive is not defined (" << prim->name() << ").";
  }
  return evaluator;
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<PrimitiveAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  auto inf_pair = evaluators_.find(func);
  if (inf_pair != evaluators_.end()) {
    return inf_pair->second;
  }
  auto primitive = func->prim();
  auto evaluator = GetPrimEvaluator(primitive, shared_from_this());
  if (evaluator == nullptr) {
    MS_LOG(EXCEPTION) << "The evaluator of the primitive is not defined (" << primitive->name() << ").";
  }
  evaluators_[func] = evaluator;
  return evaluator;
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<FuncGraphAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  auto inf_pair = evaluators_.find(func);
  if (inf_pair != evaluators_.end()) {
    return inf_pair->second;
  }
  std::shared_ptr<FuncGraphEvaluator> func_graph_evaluator =
    std::make_shared<FuncGraphEvaluator>(func->func_graph(), func->context());
  evaluators_[func] = func_graph_evaluator;
  return func_graph_evaluator;
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<MetaFuncGraphAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  auto inf_pair = evaluators_.find(func);
  if (inf_pair != evaluators_.end()) {
    return inf_pair->second;
  }

  std::shared_ptr<MetaFuncGraphEvaluator> evaluator =
    std::make_shared<MetaFuncGraphEvaluator>(func->meta_func_graph(), func->GetScope());
  evaluators_[func] = evaluator;
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
  MS_EXCEPTION_IF_NULL(func);
  MS_LOG(DEBUG) << "The func value: " << func->ToString();
  if (func->tracking_id() != nullptr) {
    MS_LOG(DEBUG) << "The tracking_id: " << func->tracking_id()->DebugString();
  }

  if (func->tracking_id() == nullptr || func->isa<abstract::MetaFuncGraphAbstractClosure>() ||
      func->isa<abstract::FuncGraphAbstractClosure>()) {
    EvaluatorPtr evaluator = _GetEvaluatorFor(func);
    return evaluator;
  }
  auto inf_pair = evaluators_.find(func);
  if (inf_pair != evaluators_.end()) {
    return inf_pair->second;
  }

  AbstractFunctionPtr func_generic = func->Copy();
  func_generic->set_tracking_id(nullptr);
  EvaluatorPtr eval = _GetEvaluatorFor(func_generic);
  auto tracked_eval = std::make_shared<TrackedEvaluator>(eval);
  evaluators_[func] = tracked_eval;

  return tracked_eval;
}

EvalResultPtr AnalysisEngine::ForwardConfig(const AnfNodeConfigPtr &orig_conf, const AnfNodeConfigPtr new_conf) {
  MS_EXCEPTION_IF_NULL(orig_conf);
  MS_EXCEPTION_IF_NULL(new_conf);
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
      MS_EXCEPTION_IF_NULL(old_cnode->func_graph());
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
  (void)forward_count_++;
  auto res = ObtainEvalResultWithCache(new_conf);
  (void)forward_count_--;
  return res;
}

EvalResultPtr AnalysisEngine::ExecuteEvaluators(const std::vector<EvaluatorPtr> &evaluators,
                                                const AnfNodeConfigPtr &out_conf, const ConfigPtrList &args_conf_list) {
  if (evaluators.size() == 1) {
    EvaluatorPtr eval = evaluators[0];
    MS_EXCEPTION_IF_NULL(eval);
    return eval->Run(shared_from_this(), args_conf_list, out_conf);
  }
  static bool enable_singleThread = (common::GetEnv("ENV_SINGLE_EVAL") == "1");
  if (enable_singleThread) {
    return ExecuteMultipleEvaluators(evaluators, out_conf, args_conf_list);
  } else {
    return ExecuteMultipleEvaluatorsMultiThread(evaluators, out_conf, args_conf_list);
  }
}

void AnalysisEngine::SetUndeterminedFlag(const EvaluatorPtr &evaluator, const FuncGraphPtr &possible_parent_fg) {
  MS_EXCEPTION_IF_NULL(evaluator);
  static std::mutex fg_lock;
  std::lock_guard<std::mutex> infer_lock(fg_lock);
  auto fg_eval = evaluator->cast<FuncGraphEvaluatorPtr>();
  if (fg_eval == nullptr) {
    return;
  }

  auto fg = fg_eval->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto undetermined_fgs = fg->recursive();
  if (undetermined_fgs) {
    auto fg_parent = fg->parent();
    if (fg_parent != nullptr) {
      fg_parent->set_flag(kFuncGraphFlagUndetermined, true);
      MS_LOG(DEBUG) << "Set graph undetermined: " << fg_parent->ToString() << " for fg: " << fg->ToString();
      return;
    } else if (possible_parent_fg != nullptr) {
      possible_parent_fg->set_flag(kFuncGraphFlagUndetermined, true);
      MS_LOG(DEBUG) << "Set graph undetermined: " << possible_parent_fg->ToString() << " for fg: " << fg->ToString();
    } else {
      MS_LOG(EXCEPTION) << "cannot find parent for fg: " << fg->ToString();
    }
  }
}

EvaluatorPtr AnalysisEngine::HandleNestedRecursion(const std::vector<EvaluatorPtr> &evaluators,
                                                   const EvaluatorPtr &eval, const AbstractBasePtrList &args_spec_list,
                                                   const EvalTraceRevIter &it, bool *continue_flag) {
  MS_EXCEPTION_IF_NULL(continue_flag);
  MS_EXCEPTION_IF_NULL(eval);
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
    auto eval_cache = alternate_evaluator->evaluator_cache_mgr();
    const auto &alt_eval_args = EvaluatorArgs(alternate_evaluator, args_spec_list);
    if ((!undetermined_evals.count(alt_eval_args)) &&
        (((!continued_evals_.count(u_eval)) && (eval_cache->GetValue(args_spec_list) != nullptr)) ||
         (eval_cache->GetValue(args_spec_list) == nullptr))) {
      MS_LOG(DEBUG) << u_eval.evaluator_->ToString() << "has undetermined.";
      has_undetermined = true;
      break;
    }
  }
  if (!has_undetermined) {
    MS_LOG(DEBUG) << eval->ToString() << "has no undetermined.";
    *continue_flag = true;
    return latest_entry;
  }

  return latest_entry;
}

std::string JoinBranchesFailedInfo(const AbstractBasePtr &spec, const AbstractBasePtr &last_spec,
                                   const AnfNodePtr &node, const std::string &error_info) {
  std::ostringstream buffer;
  buffer << "The return values of different branches do not join. \n"
         << error_info << "\nFor more details, please refer to the FAQ at https://www.mindspore.cn.\n"
         << "The abstract type of the return value of the current branch is " << spec->ToString()
         << ", and that of the previous branch is " << last_spec->ToString() << ".\n"
         << "The node " << node->DebugString();
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>()->input(0);
    if (IsPrimitiveCNode(cnode, prim::kPrimSwitch)) {
      // {prim::kPrimSwitch, cond, true_branch, false_branch}
      constexpr int true_index = 2;
      constexpr int false_index = 3;
      auto inputs = cnode->cast<CNodePtr>()->inputs();
      buffer << ", true branch: " << inputs.at(true_index)->ToString()
             << ", false branch: " << inputs.at(false_index)->ToString();
    } else if (IsPrimitiveCNode(cnode, prim::kPrimSwitchLayer)) {
      // {prim::kPrimSwitchLayer, X, {prim::kPrimMakeTuple, branch1, branch2, ...}}
      constexpr int branch_index = 2;
      auto tuple_node = cnode->cast<CNodePtr>()->input(branch_index);
      if (IsPrimitiveCNode(tuple_node, prim::kPrimMakeTuple)) {
        auto tuple_inputs = tuple_node->cast<CNodePtr>()->inputs();
        for (size_t i = 1; i < tuple_inputs.size(); i++) {
          buffer << ", branch" << i << ": " << tuple_inputs.at(i);
        }
      }
    }
  }
  buffer << ". trace: " << trace::DumpSourceLines(node);
  return buffer.str();
}

EvalResultPtr AnalysisEngine::ProcessEvalResults(const AbstractBasePtrList &out_specs, const AnfNodePtr &node) {
  if (out_specs.empty()) {
    MS_LOG(EXCEPTION) << "There is an endless loop for evaluator.";
  }

  if (out_specs.size() == 1) {
    MS_EXCEPTION_IF_NULL(out_specs[0]);
    // If only one result derived, then broaden it to avoid wrong constant propagation.
    return std::make_shared<EvalResult>(out_specs[0]->Broaden(), std::make_shared<AttrValueMap>());
  }
  MS_EXCEPTION_IF_NULL(node);

  AbstractBasePtr last_spec = out_specs[0];
  AbstractBasePtr joined_spec = out_specs[0];
  for (const auto &spec : out_specs) {
    MS_EXCEPTION_IF_NULL(spec);
    try {
      joined_spec = joined_spec->Join(spec);
    } catch (const py::type_error &ex) {
      auto error_info = ExtractLoggingInfo(ex.what());
      MS_EXCEPTION(TypeError) << JoinBranchesFailedInfo(spec, last_spec, node, error_info);
    } catch (const py::value_error &ex) {
      auto error_info = ExtractLoggingInfo(ex.what());
      MS_EXCEPTION(ValueError) << JoinBranchesFailedInfo(spec, last_spec, node, error_info);
    } catch (const std::exception &ex) {
      auto error_info = ExtractLoggingInfo(ex.what());
      MS_LOG(EXCEPTION) << JoinBranchesFailedInfo(spec, last_spec, node, error_info);
    }
    MS_EXCEPTION_IF_NULL(joined_spec);
    last_spec = spec;
  }

  MS_LOG(DEBUG) << "Multiple evaluators joined: " << joined_spec->ToString();
  return std::make_shared<EvalResult>(joined_spec, std::make_shared<AttrValueMap>());
}

bool NeedWaitForBranches(const AbstractBasePtr &abstract) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (abstract->isa<AbstractFunction>()) {
    return true;
  }
  if (abstract->isa<AbstractSequeue>()) {
    auto elements = abstract->cast<AbstractSequeuePtr>()->elements();
    if (std::any_of(elements.begin(), elements.end(),
                    [](const AbstractBasePtr &item) { return item->isa<AbstractFunction>(); })) {
      return true;
    }
  }
  return false;
}

void ExecEvaluator(EvaluatorPtr eval, AnalysisEnginePtr engine, ConfigPtrList args_conf_list, AnfNodeConfigPtr out_conf,
                   const std::string &threadID, AsyncAbstractPtr async_result_branch,
                   AsyncAbstractPtr async_result_main, AsyncInferTaskPtr async_run_flag,
                   const trace::TraceGraphEvalStack &graph_evals,
                   const trace::TraceCNodeEvalStack &trace_c_node_evals) {
  AnalysisSchedule::SetThreadID(threadID);
  // Restore trace stack for dump stack when there is exception.
  trace::TraceEvalCNodeStackPrepare(trace_c_node_evals);
  trace::TraceGraphEvalStackPrepare(graph_evals);

  try {
    // Wait for Signal to run
    MS_LOG(DEBUG) << async_run_flag.get() << "  " << eval->ToString() << " waiting.";
    (void)async_run_flag->GetResult();
    MS_LOG(DEBUG) << async_run_flag.get() << "  " << eval->ToString() << " running.";

    // Acquire GIL for eval to callback python.
    EvalResultPtr result;
    {
      py::gil_scoped_acquire pyGuard;
      result = eval->Run(engine, args_conf_list, out_conf);
    }
    MS_EXCEPTION_IF_NULL(result);
    MS_EXCEPTION_IF_NULL(result->abstract());

    // Broaden the result of switch(c,t,f)()
    auto broadAbstract = result->abstract()->Broaden();
    // Notify the thread of waiting for switch node and the main thread to continue.
    AnalysisResultCacheMgr::GetInstance().SetSwitchValue(out_conf, broadAbstract);
    async_result_branch->SetResult(broadAbstract);
    async_result_main->SetResult(broadAbstract);
    // Thread number will be drop when thread exits.
    AnalysisSchedule::GetInstance().DecreaseThreadCount();
    MS_LOG(DEBUG) << GetInferThread() << "async :" << eval->ToString()
                  << " asyncResult address = " << async_result_branch.get()
                  << " value = " << async_result_branch->TryGetResult()->ToString();
  } catch (const std::exception &e1) {
    auto abstractErrPtr = std::make_shared<AbstractError>(std::make_shared<StringImm>("Exception"), out_conf->node());
    AnalysisResultCacheMgr::GetInstance().SetSwitchValue(out_conf, abstractErrPtr);
    async_result_main->SetResult(abstractErrPtr);
    MS_LOG(INFO) << "Eval node: " << out_conf->node()->ToString() << "  " << eval->ToString() << " threw exception.";
    AnalysisSchedule::GetInstance().HandleException(e1);
    try {
      // Thread number will be drop when thread exits.
      AnalysisSchedule::GetInstance().DecreaseThreadCount();
    } catch (const std::exception &e2) {
      MS_LOG(DEBUG) << "AnalysisSchedule::GetInstance().DecreaseThreadCount() threw exception.";
    }
  }
}

EvalResultPtr AnalysisEngine::ExecuteMultipleEvaluatorsMultiThread(const std::vector<EvaluatorPtr> &evaluators,
                                                                   const AnfNodeConfigPtr &out_conf,
                                                                   const ConfigPtrList &args_conf_list) {
  MS_EXCEPTION_IF_NULL(out_conf);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  // Release GIL for C++
  py::gil_scoped_release infer_gil_release;
  // Wait for the last switch node to finish.
  MS_LOG(DEBUG) << GetInferThread() << "async : entry switch  " << out_conf->ToString();
  auto eval_result = AnalysisResultCacheMgr::GetInstance().GetSwitchValue(out_conf);
  if (eval_result == nullptr) {
    MS_LOG(DEBUG) << GetInferThread() << "async : Init switch  " << out_conf->node()->ToString();
    AnalysisResultCacheMgr::GetInstance().InitSwitchValue(out_conf);
  } else {
    return std::make_shared<EvalResult>(eval_result, nullptr);
  }
  auto possible_parent_fg = out_conf->node()->func_graph();

  // Eval result of the main.
  AsyncAbstractPtr asyncResult_main = std::make_shared<AsyncAbstract>();
  // Eval result of the branches
  std::vector<AsyncAbstractPtr> branchAsyncResults;

  for (auto &evaluator : evaluators) {
    static std::atomic<int> idCount{0};
    std::string threadId = AnalysisSchedule::GetThreadID() + "." + std::to_string(idCount.fetch_add(1));
    MS_EXCEPTION_IF_NULL(evaluator);
    SetUndeterminedFlag(evaluator, possible_parent_fg);
    AsyncAbstractPtr branchAsyncResult = std::make_shared<AsyncAbstract>();
    // Control the order to run.
    AsyncAbstractPtr asyncRunOrder = std::make_shared<AsyncAbstract>();
    AsyncInferTaskPtr asyncTask = AsyncInferTask::MakeShared(asyncRunOrder, threadId);
    // Add point to the async thread.
    AnalysisSchedule::GetInstance().IncreaseThreadCount();
    MS_LOG(DEBUG) << GetInferThread() << "async : " << evaluator->ToString();
    auto thread =
      std::thread(ExecEvaluator, evaluator, shared_from_this(), args_conf_list, out_conf, threadId, branchAsyncResult,
                  asyncResult_main, asyncTask, trace::GetCurrentGraphEvalStack(), trace::GetCNodeDebugStack());
    thread.detach();
    // Push to list of running loop
    asyncRunOrder->SetResult(std::make_shared<AbstractScalar>(1));
    MS_LOG(DEBUG) << " add to schedule: " << asyncTask.get();
    AnalysisSchedule::GetInstance().Add2Schedule(asyncTask);  // Activate order witch child thread.
    (void)branchAsyncResults.emplace_back(std::move(branchAsyncResult));
  }

  MS_LOG(DEBUG) << GetInferThread() << "async : wait for one of async to finish.  " << evaluators[0]->ToString()
                << " or  " << evaluators[1]->ToString() << "...";
  auto async_main = AsyncInferTask::MakeShared(asyncResult_main);
  MS_LOG(DEBUG) << " add to schedule: " << async_main.get();
  AnalysisSchedule::GetInstance().Add2Schedule(async_main);  // Third order
  auto firstResult = async_main->GetResult();
  MS_EXCEPTION_IF_NULL(firstResult);
  MS_LOG(DEBUG) << GetInferThread() << "async main thread result of " << out_conf->node()->ToString() << " = "
                << firstResult->ToString();

  AbstractBasePtrList out_specs;
  size_t len = evaluators.size();
  if (NeedWaitForBranches(firstResult)) {
    for (size_t i = 0; i < len; ++i) {
      MS_LOG(DEBUG) << GetInferThread() << "async waiting for " << evaluators[i]->ToString();
      auto async_branch = AsyncInferTask::MakeShared(branchAsyncResults[i]);
      MS_LOG(DEBUG) << " add to schedule: " << async_branch.get();
      AnalysisSchedule::GetInstance().Add2Schedule(async_branch);
      auto result = async_branch->GetResult();
      MS_EXCEPTION_IF_NULL(result);
      out_specs.push_back(result);
    }
  } else {
    // Give one more chance to wait for the result of the branches.
    auto async_tmp = AsyncInferTask::MakeShared(asyncResult_main);
    MS_LOG(DEBUG) << " add to schedule: " << async_tmp.get();
    AnalysisSchedule::GetInstance().Add2Schedule(async_tmp);
    (void)async_tmp->GetResult();
    for (size_t i = 0; i < len; ++i) {
      // Not wait to get the result of branch.
      auto result = branchAsyncResults[i]->TryGetResult();
      if (result) {
        MS_LOG(DEBUG) << GetInferThread() << "async get " << evaluators[i]->ToString()
                      << " result: " << result->ToString();
        out_specs.push_back(result);
      }
    }
  }

  return ProcessEvalResults(out_specs, out_conf->node());
}

EvalResultPtr AnalysisEngine::ExecuteMultipleEvaluators(const std::vector<EvaluatorPtr> &evaluators,
                                                        const AnfNodeConfigPtr &out_conf,
                                                        const ConfigPtrList &args_conf_list) {
  AbstractBasePtrList out_specs;
  const size_t evaluators_size = 2;
  if (evaluators.size() < evaluators_size) {
    MS_LOG(ERROR) << "evaluators size is less than 2";
  }
  multi_poss_[evaluators[0]] = evaluators[1];
  multi_poss_[evaluators[1]] = evaluators[0];
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->ObtainEvalResult()->abstract();
                       });
  MS_EXCEPTION_IF_NULL(out_conf);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  auto possible_parent_fg = out_conf->node()->func_graph();
  for (auto eval : evaluators) {
    MS_EXCEPTION_IF_NULL(eval);
    (void)SetUndeterminedFlag(eval, possible_parent_fg);
    const auto current_inf = EvaluatorArgs(eval, args_spec_list);
    MS_LOG(DEBUG) << "Check Evaluator " << eval->ToString();
    // If current evaluator is under tracing, then skip current evaluator to avoid recursively evaluating.
    auto it = std::find(eval_trace_.rbegin(), eval_trace_.rend(), current_inf);
    if (it == eval_trace_.rend()) {
      eval_trace_.push_back(current_inf);
      auto eval_result = eval->Run(shared_from_this(), args_conf_list, out_conf);
      auto eval_abstract = eval_result->abstract();
      MS_EXCEPTION_IF_NULL(eval_abstract);

      out_specs.push_back(eval_abstract);
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

  return ProcessEvalResults(out_specs, out_conf->node());
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
  MS_EXCEPTION_IF_NULL(value);
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
  } else {
    return value->ToAbstract();
  }
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
  if (evaluator == nullptr) {
    MS_LOG(EXCEPTION) << "The evaluator of the primitive is not defined (" << primitive->name() << ").";
  }
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
