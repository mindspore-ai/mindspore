/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "pipeline/static_analysis/static_analysis.h"

#include <algorithm>

#include "pipeline/static_analysis/utils.h"
#include "pipeline/static_analysis/prim.h"
#include "operator/ops.h"
#include "utils/symbolic.h"
#include "ir/meta_tensor.h"
#include "ir/func_graph_cloner.h"
#include "./common.h"
#include "pipeline/parse/data_converter.h"
#include "debug/draw.h"
#include "pipeline/static_analysis/evaluator.h"
#include "debug/trace.h"

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

void AnalysisCache::set_value(const AnfNodeConfigPtr &conf, const AbstractBasePtr &arg) {
  MS_LOG(DEBUG) << "AnalysisCache set for NodeConfig: " << conf->node()->DebugString()
                << ", Context: " << conf->context()->ToString() << ", Value: " << arg->ToString()
                << ", Pointer: " << arg.get();
  cache_[conf] = arg;

  // Set intermediate abstract value.
  if (IsIntermediateAbstract(arg)) {
    if (conf->node()->intermediate_abstract() == nullptr) {
      conf->node()->set_intermediate_abstract(arg);
      MS_LOG(DEBUG) << "Set intermediate abstract: " << arg->ToString();
    } else {
      auto old_spec = conf->node()->intermediate_abstract();
      auto joined_spec = IntermediateJoin(arg, old_spec);
      conf->node()->set_intermediate_abstract(joined_spec);
      MS_LOG(DEBUG) << "Set joined intermediate abstract:\nold_spec:\t\t" << old_spec->ToString() << "\nnew_spec:\t\t"
                    << arg->ToString() << "\njoined_spec:\t"
                    << (joined_spec != nullptr ? joined_spec->ToString() : "nullptr");
    }
  }
}

AbstractBasePtr AnalysisCache::GetValue(const AnfNodeConfigPtr &conf) {
  auto value = cache_.find(conf);
  if (value == cache_.end()) {
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
  if (conf->context() != nullptr && conf->context()->func_graph() != nullptr) {
    MS_LOG(DEBUG) << "NodeConfigHasher Node: " << conf->node()->DebugString()
                  << ", Graph: " << conf->context()->func_graph()->ToString() << " ### , hash value: " << hash_value;
  } else {
    MS_LOG(DEBUG) << "NodeConfigHasher Node: " << conf->node()->DebugString() << " ### , hash value: " << hash_value;
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
  AnalysisContextPtr root_context = Run(func_graph, empty_context, args_conf_list);
  MS_EXCEPTION_IF_NULL(root_context);
  MS_EXCEPTION_IF_NULL(root_context->func_graph());
  AnfNodeConfigPtr output_conf = MakeConfig(root_context->func_graph()->get_return(), root_context);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << func_graph->ToString() << ": Run finished.";

  AnalysisResult result;
  MS_EXCEPTION_IF_NULL(output_conf);
  result.inferred = output_conf->GetEvaluatedValue();
  result.context = root_context;
  return result;
}

AnalysisContextPtr AnalysisEngine::Run(const FuncGraphPtr &func_graph, const AnalysisContextPtr &context,
                                       const ConfigPtrList &args_conf_list) {
  std::shared_ptr<FuncGraphEvaluator> eval = std::make_shared<FuncGraphEvaluator>(func_graph, context);
  (void)eval->Run(shared_from_this(), args_conf_list, nullptr);
  return eval->graph_context();
}

AbstractBasePtr AnalysisEngine::GetEvaluatedValue(const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  auto value = cache_.GetValue(conf);
  if (value != nullptr) {
    MS_LOG(DEBUG) << "Evaluate cache hit for NodeConfig: " << conf->ToString() << ", Value: " << value.get() << ", "
                  << value->ToString();
    return value;
  }

  MS_LOG(DEBUG) << "Evaluate cache miss for NodeConfig: " << conf->ToString();
  value = Eval(conf);
  if (value == nullptr) {
    MS_LOG(EXCEPTION) << "Evaluate for NodeConfig " << conf->ToString() << " get nullptr";
  }
  cache_.set_value(conf, value);
  return value;
}

AbstractBasePtr AnalysisEngine::Eval(const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  AnfNodePtr node = conf->node();
  AbstractBasePtr ret_abstract = nullptr;
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
    ret_abstract = node->abstract();
  } else if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    ret_abstract = EvalValueNode(value_node, conf);
  } else if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    trace::TraceInferCNodeEnter(conf);
    ret_abstract = InferCNode(cnode, conf);
    trace::TraceInferCNodeLeave();
  } else {
    MS_LOG(EXCEPTION) << "Illegal AnfNode for inferring, " << node->DebugString()
                      << ". NodeInfo: " << trace::GetDebugInfo(node->debug_info());
  }

#ifdef DEBUG
  compute_conf_stack_.pop_back();
  if (ret_abstract == nullptr) {
    MS_LOG(EXCEPTION) << "Compute Config failed, node: " << node->DebugString()
                      << " NodeInfo: " << trace::GetDebugInfo(node->debug_info());
  }
#endif
  MS_LOG(DEBUG) << "End Eval NodeConfig " << conf->ToString() << ", res: " << ret_abstract->ToString();
  return ret_abstract;
}

AbstractBasePtr AnalysisEngine::EvalValueNode(const ValueNodePtr &value_node, const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  MS_EXCEPTION_IF_NULL(value_node);
  return ToAbstract(value_node->value(), conf->context(), conf);
}

AbstractBasePtr AnalysisEngine::InferCNode(const CNodePtr &cnode, const AnfNodeConfigPtr &conf) {
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
  AbstractBasePtr maybe_func = func_conf->GetEvaluatedValue();
  if (maybe_func == nullptr) {
    MS_LOG(EXCEPTION) << "func_conf.GetEvaluatedValue() return null, func_conf: " << func_conf->ToString()
                      << " NodeInfo: " << trace::GetDebugInfo(cnode->debug_info());
  }
  AbstractFunctionPtr func = dyn_cast<AbstractFunction>(maybe_func);
  if (func == nullptr) {
    MS_LOG(EXCEPTION) << "func_conf.GetEvaluatedValue() return not AbstractFunction: " << maybe_func->ToString()
                      << ", func_conf: " << func_conf->ToString()
                      << " NodeInfo: " << trace::GetDebugInfo(cnode->debug_info());
  }

  ConfigPtrList args_conf_list;
  // ignore the first node which is function name
  for (std::size_t i = 1; i < inputs.size(); i++) {
    const AnfNodePtr &node = inputs[i];
    args_conf_list.push_back(MakeConfig(node, context));
    MS_LOG(DEBUG) << "Current CNode args_conf_list[" << i << "] node: " << node->DebugString();
  }
  std::vector<EvaluatorPtr> infs;

  auto build_evaluator = [this, &infs, &cnode](const AbstractFuncAtomPtr &poss) {
    auto evaluator = this->GetEvaluatorFor(poss);
    evaluator->set_bound_node(cnode);
    infs.push_back(evaluator);
  };
  func->Visit(build_evaluator);

  return ExecuteEvaluators(infs, conf, args_conf_list);
}

AbstractBasePtr AnalysisEngine::Execute(const AbstractFunctionPtr &func, const AbstractBasePtrList &args_spec_list) {
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
    MS_EXCEPTION_IF_NULL(evaluator->cache());
    evaluator->cache()->clear();
  }
  for (auto &element : prim_constructors_) {
    EvaluatorPtr evaluator = element.second;
    MS_EXCEPTION_IF_NULL(evaluator);
    MS_EXCEPTION_IF_NULL(evaluator->cache());
    evaluator->cache()->clear();
  }
  for (auto &element : prim_py_evaluators_) {
    EvaluatorPtr evaluator = element.second;
    MS_EXCEPTION_IF_NULL(evaluator);
    MS_EXCEPTION_IF_NULL(evaluator->cache());
    evaluator->cache()->clear();
  }
}

void AnalysisEngine::Clear() {
  cache_.Clear();
  anfnode_config_map_.clear();
  eval_trace_.clear();
  constructors_.clear();
}

namespace {
EvaluatorPtr GetPrimEvaluator(const PrimitivePtr &prim, const AnalysisEnginePtr &engine) {
  // Custom Primitive with python infer_shape, infer_type
  EvaluatorPtr evaluator = nullptr;
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->isa<prim::DoSignaturePrimitive>()) {
    evaluator = std::make_shared<DoSignatureEvaluator>(prim);
    return evaluator;
  }
  if (prim->isa<prim::UnpackGraphPrimitive>()) {
    evaluator = std::make_shared<UnpackGraphEvaluator>(prim);
    return evaluator;
  }
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

  if (prim->isa<PrimitivePy>() || prim->HasAttr()) {
    if (engine == nullptr) {
      (void)GetPrimEvaluatorConstructors();
    }
    // If a primitive may have attr, try to create a new evaluator.
    StandardPrimitiveEvalImpl eval_impl = GetPrimitiveInferImpl(prim);
    if (eval_impl != nullptr) {
      return std::make_shared<StandardPrimEvaluator>(prim, eval_impl);
    }
  }

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
  std::shared_ptr<PartialAppEvaluator> partial_evaluator =
    std::make_shared<PartialAppEvaluator>(evaluator_orig, func->args());
  return partial_evaluator;
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<TypedPrimitiveAbstractClosure> &) {
  MS_LOG(EXCEPTION) << "Should not be called ";
}

// Forward to specific subclass of FunctionWrapper.
EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const AbstractFunctionPtr &func) {
  MS_EXCEPTION_IF_NULL(func);
  EvaluatorPtr evaluator = func->GetEvaluator(shared_from_this());
  return evaluator;
}

EvaluatorPtr AnalysisEngine::GetEvaluatorFor(const AbstractFunctionPtr &func) {
  MS_LOG(DEBUG) << "The func value: " << func->ToString();
  if (func->tracking_id() != nullptr) {
    MS_LOG(DEBUG) << "The tracking_id: " << func->tracking_id()->DebugString();
  }
  MS_EXCEPTION_IF_NULL(func);
  if (func->tracking_id() == nullptr) {
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

AbstractBasePtr AnalysisEngine::ExecuteEvaluators(const std::vector<EvaluatorPtr> &evaluators,
                                                  const AnfNodeConfigPtr &out_conf,
                                                  const ConfigPtrList &args_conf_list) {
  if (evaluators.size() == 1) {
    EvaluatorPtr eval = evaluators[0];
    MS_EXCEPTION_IF_NULL(eval);
    return eval->Run(shared_from_this(), args_conf_list, out_conf);
  }
  return ExecuteMultipleEvaluators(evaluators, out_conf, args_conf_list);
}

AbstractBasePtr AnalysisEngine::ExecuteMultipleEvaluators(const std::vector<EvaluatorPtr> &evaluators,
                                                          const AnfNodeConfigPtr &out_conf,
                                                          const ConfigPtrList &args_conf_list) {
  AbstractBasePtrList out_specs;
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->GetEvaluatedValue();
                       });
  for (auto eval : evaluators) {
    auto fg_eval = eval->cast<FuncGraphEvaluatorPtr>();
    if (fg_eval) {
      auto undetermined_fgs = fg_eval->func_graph()->recursive_graphs();
      if (undetermined_fgs) {
        for (auto undetermined_fg : *undetermined_fgs) {
          MS_LOG(DEBUG) << "Set graph undetermined: " << undetermined_fg->ToString();
          // As the current evaluator has multiple possibles, all the func_graphs which
          // are recursive with the current func_graph are undetermined in control flow.
          undetermined_fg->set_flags(kFuncGraphFlagUndetermined, true);
        }
      }
    }

    auto current_inf = std::make_pair(eval, args_spec_list);
    // If current evaluator is under tracing, then skip current evaluator to avoid recursively inferring.
    auto it = std::find(eval_trace_.begin(), eval_trace_.end(), current_inf);
    if (it == eval_trace_.end()) {
      eval_trace_.push_back(current_inf);
      MS_EXCEPTION_IF_NULL(eval);
      auto out_spec = eval->Run(shared_from_this(), args_conf_list, out_conf);
      MS_EXCEPTION_IF_NULL(out_spec);
      MS_LOG(DEBUG) << "Evaluator " << eval->ToString() << " return out_spec: " << out_spec->ToString();
      out_specs.push_back(out_spec);
      eval_trace_.pop_back();
    }
  }
  if (out_specs.size() == 0) {
    MS_LOG(EXCEPTION) << "There is an endless loop for evaluator.";
  }

  if (out_specs.size() == 1) {
    MS_EXCEPTION_IF_NULL(out_specs[0]);
    // If only one result derived, then broaden it to avoid wrong constant propagation.
    return out_specs[0]->Broaden();
  }
  auto joined_spec = AbstractJoin(out_specs);
  MS_EXCEPTION_IF_NULL(joined_spec);
  MS_LOG(DEBUG) << "Multiple evaluators joined: " << joined_spec->ToString();
  return joined_spec;
}

AbstractBasePtr AnfNodeConfig::GetEvaluatedValue() {
  AnfNodeConfigPtr self = shared_from_base<AnfNodeConfig>();
  return engine_.lock()->GetEvaluatedValue(self);
}

AbstractBasePtr ToAbstract(const ValuePtr &value, const AnalysisContextPtr &context, const AnfNodeConfigPtr &conf) {
  if (value->isa<FuncGraph>()) {
    auto func_graph = value->cast<FuncGraphPtr>();
    return func_graph->MakeAbstractClosure(context);
  }
  AnfNodePtr anf_node = nullptr;
  if (conf != nullptr) {
    anf_node = conf->node();
  }
  if (value->isa<MetaFuncGraph>()) {
    auto meta_func_graph = value->cast<MetaFuncGraphPtr>();
    return meta_func_graph->MakeAbstractClosure(anf_node);
  }
  if (value->isa<Primitive>()) {
    auto prim = value->cast<PrimitivePtr>();
    return prim->ToPrimAbstract(anf_node);
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

AbstractBasePtr InferOnePrim(const PrimitivePtr &primitive, const AbstractBasePtrList &arg_specs) {
  auto evaluator = GetPrimEvaluator(primitive, nullptr);
  MS_EXCEPTION_IF_NULL(evaluator);
  if (!evaluator->isa<TrivialPrimEvaluator>()) {
    MS_LOG(EXCEPTION) << "Prim " << primitive->ToString() << " should build a TrivialPrimEvaluator, but "
                      << evaluator->ToString();
  }
  auto trivial_evaluator = dyn_cast<TrivialPrimEvaluator>(evaluator);
  auto res_spec = trivial_evaluator->EvalPrim(nullptr, arg_specs);
  return res_spec;
}
}  // namespace abstract
}  // namespace mindspore
