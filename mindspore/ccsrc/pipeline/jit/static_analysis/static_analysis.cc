/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <memory>
#include <mutex>
#include <set>
#include <unordered_set>
#include <utility>
#include "abstract/abstract_value.h"
#include "pipeline/jit/parse/resolve.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "frontend/operator/ops.h"
#include "utils/symbolic.h"
#include "utils/ms_exception.h"
#include "ir/func_graph_cloner.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/static_analysis/evaluator.h"
#include "pipeline/jit/debug/trace.h"
#include "include/common/debug/anf_ir_dump.h"
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

EvalResultPtr PrimitiveEvalCache::Get(const PrimitivePtr &prim, const AbstractBasePtrList &args) const {
  MS_EXCEPTION_IF_NULL(prim);
  std::lock_guard<std::mutex> guard(mutex_);
  auto cache_iter = prim_cache_.find(prim->name());
  if (cache_iter == prim_cache_.end()) {
    return nullptr;
  }
  auto &cache = cache_iter->second;
  auto iter = cache.find(PrimitiveEvalCacheKey{prim->attrs(), args});
  if (iter == cache.end()) {
    return nullptr;
  }
  return iter->second;
}

void PrimitiveEvalCache::Put(const PrimitivePtr &prim, AttrValueMap &&attrs, const AbstractBasePtrList &args,
                             const EvalResultPtr &result) {
  MS_EXCEPTION_IF_NULL(prim);
  std::lock_guard<std::mutex> guard(mutex_);
  (void)prim_cache_[prim->name()].emplace(PrimitiveEvalCacheKey{std::move(attrs), args}, result);
}

void PrimitiveEvalCache::Clear() {
  std::lock_guard<std::mutex> guard(mutex_);
  prim_cache_.clear();
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
    // Create a new root dummy context for the new analysis session.
    AnalysisContextPtr dummy_context = AnalysisContext::NewDummyContext();
    MS_LOG(DEBUG) << func_graph->ToString() << ": Run begin.";
    AnalysisContextPtr root_context = Run(func_graph, dummy_context, args_conf_list);
    AnalysisSchedule::GetInstance().Wait();
    MS_EXCEPTION_IF_NULL(root_context);
    auto root_context_fg = root_context->func_graph();
    MS_EXCEPTION_IF_NULL(root_context_fg);
    AnfNodeConfigPtr output_conf = MakeConfig(root_context_fg->get_return(), root_context, root_context_fg);
    MS_EXCEPTION_IF_NULL(func_graph);
    MS_LOG(DEBUG) << func_graph->ToString() << ": Run finished.";

    MS_EXCEPTION_IF_NULL(output_conf);
    auto eval_result = output_conf->ObtainEvalResult();
    result.eval_result = eval_result;
    result.context = root_context;
  } catch (const std::exception &ex) {
    MS_LOG(INFO) << "Eval " << func_graph->ToString() << " threw exception.";
    AnalysisSchedule::GetInstance().HandleException(ex);
  }
  AnalysisSchedule::GetInstance().Wait();
  MS_LOG(DEBUG) << func_graph->ToString() << ": Run end.";
  // Set the sequence nodes' elements use flags all true.
  SetSequenceElementsUseFlagsRecursively(result.eval_result->abstract(), true);
  MS_LOG(DEBUG) << func_graph->ToString() << ":SetSequenceElementsUseFlagsRecursively Run end.";
  return result;
}

AnalysisContextPtr AnalysisEngine::Run(const FuncGraphPtr &func_graph, const AnalysisContextPtr &context,
                                       const ConfigPtrList &args_conf_list) {
  auto evaluator = std::make_shared<FuncGraphEvaluator>(func_graph, context);
  (void)evaluator->Run(shared_from_this(), args_conf_list, nullptr);
  return root_context_;
}

void AnalysisEngine::SaveEvalResultInCache(const AnfNodeConfigPtr &conf, const EvalResultPtr &result) const {
  MS_EXCEPTION_IF_NULL(conf);
  MS_EXCEPTION_IF_NULL(result);
  static AnalysisResultCacheMgr &cache_mgr = AnalysisResultCacheMgr::GetInstance();
  auto iter = cache_mgr.GetCache().find(conf);
  if (iter != cache_mgr.GetCache().end()) {
    MS_LOG(DEBUG) << "Found previous result for NodeConfig: " << conf->ToString()
                  << ", result: " << iter->second->abstract().get() << "/" << iter->second->abstract()->ToString();
    // Update sequence nodes info, if matched in cache.
    static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
    if (enable_eliminate_unused_element) {
      auto new_sequence = dyn_cast<AbstractSequence>(result->abstract());
      auto old_sequence = dyn_cast<AbstractSequence>(iter->second->abstract());
      if (old_sequence != nullptr && new_sequence != nullptr) {
        MS_LOG(DEBUG) << "Before synchronize sequence nodes use flags for NodeConfig: " << conf->ToString()
                      << ", old_sequence: " << old_sequence->ToString()
                      << ", new_sequence: " << new_sequence->ToString();
        SynchronizeSequenceElementsUseFlagsRecursively(old_sequence, new_sequence);
        MS_LOG(DEBUG) << "After synchronize sequence nodes use flags for NodeConfig: " << conf->ToString()
                      << ", old_sequence: " << old_sequence->ToString()
                      << ", new_sequence: " << new_sequence->ToString();
      }
    }
  }
  MS_LOG(DEBUG) << "Save result for NodeConfig: " << conf->ToString() << ", result: " << result->abstract().get() << "/"
                << result->abstract()->ToString();
  cache_mgr.SetValue(conf, result);
}

void SynchronizeSequenceElementsUseFlagsForFuncGraphArgs(const AnalysisEnginePtr &engine, const FuncGraphPtr &fg,
                                                         const CNodePtr &cnode,
                                                         const AbstractFunctionPtr &base_func_graph_func,
                                                         const AnalysisContextPtr &fg_context) {
  // Get the evaluator for func graph.
  auto evaluator = engine->GetEvaluatorFor(base_func_graph_func);
  MS_EXCEPTION_IF_NULL(evaluator);

  AbstractBasePtrList args_abs_list;
  auto &inputs = cnode->inputs();
  for (std::size_t i = 1; i < inputs.size(); i++) {
    auto config = engine->MakeConfig(inputs[i], fg_context, fg);
    auto result = config->ObtainEvalResult();
    MS_EXCEPTION_IF_NULL(result);
    auto abs = result->abstract();
    args_abs_list.push_back(abs);
  }

  // Check if already evaluated before.
  auto &cache = evaluator->evaluator_cache_mgr()->GetCache();
  auto iter = cache.find(args_abs_list);
  if (iter != cache.end()) {
    MS_LOG(DEBUG) << "Eval before, current_node: " << cnode->DebugString() << ", context: " << fg_context->ToString()
                  << ", args: " << args_abs_list;
    // Update inputs sequence nodes info, if matched in cache.
    for (std::size_t i = 0; i < args_abs_list.size(); ++i) {
      auto new_sequence = dyn_cast<AbstractSequence>(args_abs_list[i]);
      auto old_sequence = dyn_cast<AbstractSequence>(iter->first[i]);
      if (old_sequence != nullptr && new_sequence != nullptr) {
        MS_LOG(DEBUG) << "Before synchronize sequence nodes use flags, old_sequence: " << old_sequence->ToString()
                      << ", new_sequence: " << new_sequence->ToString();
        SynchronizeSequenceElementsUseFlagsRecursively(old_sequence, new_sequence);
        MS_LOG(DEBUG) << "After synchronize sequence nodes use flags, old_sequence: " << old_sequence->ToString()
                      << ", new_sequence: " << new_sequence->ToString();
      }
    }
  }
}

EvalResultPtr ObtainEvalResultFromCache(const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  static AnalysisResultCacheMgr &cache_mgr = AnalysisResultCacheMgr::GetInstance();
  auto result = cache_mgr.GetValue(conf);
  if (result != nullptr) {
    MS_LOG(DEBUG) << "Evaluate cache found for NodeConfig: " << conf->ToString()
                  << ", result: " << result->abstract().get() << "/" << result->abstract()->ToString();
    return result;
  }
  return nullptr;
}

EvalResultPtr AnalysisEngine::ObtainEvalResultWithCache(const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  auto result = ObtainEvalResultFromCache(conf);
  if (result != nullptr) {
    return result;
  }
  MS_LOG(DEBUG) << "Evaluate cache miss for NodeConfig: " << conf->ToString();
  result = ObtainEvalResultWithoutCache(conf);
  return result;
}

EvalResultPtr AnalysisEngine::ObtainEvalResultWithoutCache(const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  EvalResultPtr result = nullptr;
  result = Eval(conf);
  if (result == nullptr) {
    MS_LOG(EXCEPTION) << "Evaluate for NodeConfig " << conf->ToString() << " get nullptr";
  }
  MS_LOG(DEBUG) << "Always Evaluate node for NodeConfig: " << conf->ToString()
                << ", result: " << result->abstract().get() << "/" << result->abstract()->ToString();
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
    MS_LOG(EXCEPTION) << "Illegal AnfNode for evaluating, node: " << node->DebugString()
                      << "(type:" << node->type_name()
                      << "), fg: " << (node->func_graph() != nullptr ? node->func_graph()->ToString() : "nullgraph")
                      << " conf: " << conf->ToString();
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

AbstractBasePtr AnalysisEngine::EvalValueNode(const ValueNodePtr &value_node, const AnfNodeConfigPtr &conf) const {
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
  auto &func_node = inputs[0];
  MS_EXCEPTION_IF_NULL(func_node);
  MS_LOG(DEBUG) << "Current CNode function: " << func_node->DebugString();
  AnfNodeConfigPtr func_conf = MakeConfig(func_node, context, func_graph);
  MS_EXCEPTION_IF_NULL(func_conf);
  // Keep it in a local variable, otherwise smart pointer will free it.
  auto possible_func_eval_result = func_conf->ObtainEvalResult();
  MS_EXCEPTION_IF_NULL(possible_func_eval_result);
  auto &possible_func = possible_func_eval_result->abstract();
  if (possible_func == nullptr) {
    MS_LOG(EXCEPTION) << "No abstract, func_conf: " << func_conf->ToString();
  }
  return possible_func;
}

void CheckInterpretedObject(const AbstractBasePtr &abs) {
  static const auto support_fallback = common::GetEnv("MS_DEV_ENABLE_FALLBACK");
  static const auto use_fallback = (support_fallback != "0");
  if (!use_fallback) {
    return;
  }
  auto value = abs->BuildValue();
  if (value->isa<parse::InterpretedObject>()) {
    MS_LOG(ERROR) << "Do not support " << value->ToString() << ". "
                  << "\nIf you are using third-party modules, you can try setting: "
                  << "'export MS_DEV_SUPPORT_MODULES=module1,module2,...'.";
  }
}

EvalResultPtr ConvertClassToFunc(const CNodePtr &cnode, const AbstractBasePtr &abs, const AnfNodeConfigPtr &conf) {
  auto val = abs->BuildValue();
  auto class_val = dyn_cast_ptr<parse::ClassType>(val);
  const auto &class_name = class_val->name();
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  auto py_fn = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_CONVERT_CLASS_TO_FUNCTION, py::str(class_name));
  if (py::isinstance<py::none>(py_fn)) {
    MS_LOG(ERROR) << "Can not cast to a AbstractFunction from " << abs->ToString() << ".";
    MS_LOG(ERROR) << "It's called at: " << cnode->DebugString();
    MS_EXCEPTION(ValueError) << "Can not call " << class_name << " to create python object in graph mode. "
                             << "Try using 'jit_class' to decorate the class?";
  }
  auto list_func_fg = parse::ParsePythonCode(py_fn);
  auto fg = cnode->func_graph();
  list_func_fg->set_manager(fg->manager());

  auto &inputs = cnode->inputs();
  std::vector<AnfNodePtr> new_cnode_inputs;
  (void)new_cnode_inputs.emplace_back(NewValueNode(list_func_fg));
  for (std::size_t i = 1; i < inputs.size(); ++i) {
    (void)new_cnode_inputs.emplace_back(inputs[i]);
  }
  auto new_cnode = fg->NewCNodeInOrder(new_cnode_inputs);
  fg->ReplaceInOrder(cnode, new_cnode);

  AnalysisEnginePtr eng = conf->engine();
  MS_EXCEPTION_IF_NULL(eng);
  AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_cnode, conf->context(), conf->func_graph());
  return eng->ForwardConfig(conf, fn_conf);
}

EvalResultPtr AnalysisEngine::EvalCNode(const CNodePtr &cnode, const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  MS_EXCEPTION_IF_NULL(cnode);
  AbstractBasePtr possible_func = GetCNodeOperatorAbstract(cnode, conf->context(), conf->func_graph());
  if (possible_func->BuildType()->type_id() == kObjectTypeUndeterminedType) {
    MS_LOG(DEBUG) << "EvalCNode eval Undetermined";
    return std::make_shared<EvalResult>(possible_func->Clone(), std::make_shared<AttrValueMap>());
  }

  if (possible_func->isa<AbstractScalar>()) {
    // Convert class to function, such as list(xxx).
    auto val = possible_func->BuildValue();
    MS_EXCEPTION_IF_NULL(val);
    if (val->isa<parse::ClassType>()) {
      return ConvertClassToFunc(cnode, possible_func, conf);
    }
  }

  auto func = dyn_cast_ptr<AbstractFunction>(possible_func);
  if (func == nullptr) {
    CheckInterpretedObject(possible_func);
    MS_LOG(ERROR) << "Can not cast to a AbstractFunction from " << possible_func->ToString() << ".";
    MS_LOG(ERROR) << "It's called at: " << cnode->DebugString();
    MS_EXCEPTION(ValueError) << "The object is not callable. Please check code.";
  }

  // Make arguments config list.
  bool contains_isolated_side_effect = false;
  auto &inputs = cnode->inputs();
  const auto inputs_size = inputs.size();
  ConfigPtrList args_conf_list;
  args_conf_list.reserve(inputs_size - 1);
  // Ignore the first node which is function name.
  for (std::size_t i = 1; i < inputs_size; ++i) {
    const AnfNodePtr &node = inputs[i];
    (void)args_conf_list.emplace_back(MakeConfig(node, conf->context(), conf->func_graph()));
    if (check_isolated_side_effect()) {
      auto input_cnode = dyn_cast_ptr<CNode>(node);
      if (input_cnode != nullptr) {
        contains_isolated_side_effect = contains_isolated_side_effect || input_cnode->has_isolated_side_effect_node();
      }
    }
  }

  // Find evaluators.
  std::vector<EvaluatorPtr> evaluators;
  func->Visit([this, &evaluators, &cnode](const AbstractFuncAtomPtr &poss) {
    MS_EXCEPTION_IF_NULL(poss);
    auto resolved_atom = poss;
    auto async_abs_func = poss->cast_ptr<AsyncAbstractFuncAtom>();
    if (async_abs_func != nullptr) {
      auto resolved_func = async_abs_func->GetUnique();
      resolved_atom = dyn_cast<AbstractFuncAtom>(resolved_func);
      MS_EXCEPTION_IF_NULL(resolved_atom);
      MS_LOG(DEBUG) << "Resolved AsyncAbstractFuncAtom is: " << resolved_atom->ToString();
    }
    auto evaluator = this->GetEvaluatorFor(resolved_atom);
    evaluator->set_bound_node(cnode);
    (void)evaluators.emplace_back(std::move(evaluator));
  });

  // Run evaluators.
  auto eval_result = ExecuteEvaluators(evaluators, conf, args_conf_list);
  // Check if func graph contains isolated side-effect, and sync.
  if (check_isolated_side_effect()) {
    auto func_graph_abs = mindspore::cast<FuncGraphAbstractClosure>(func);
    if (func_graph_abs != nullptr) {
      contains_isolated_side_effect =
        contains_isolated_side_effect || func_graph_abs->func_graph()->has_isolated_side_effect_node();
    }
    auto meta_func_graph_abs = mindspore::cast<MetaFuncGraphAbstractClosure>(func);
    if (meta_func_graph_abs != nullptr) {
      contains_isolated_side_effect =
        contains_isolated_side_effect || meta_func_graph_abs->meta_func_graph()->has_isolated_side_effect_node();
    }
    if (contains_isolated_side_effect) {
      cnode->set_has_isolated_side_effect_node(true);
      conf->func_graph()->set_has_isolated_side_effect_node(true);
    }
  }
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
  py::gil_scoped_acquire gil;
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
  // Release exception to avoid hup at exit.
  StaticAnalysisException::Instance().ClearException();
  // Reset the EnvironGet sparse option.
  EnvSetSparseResultMgr::GetInstance().Set(false);
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

  // Find prim infer function in the prim function map return a standard evaluator
  auto eval_impl = GetPrimitiveInferImpl(prim);
  if (eval_impl.IsImplInferShapeAndType() && prim->name() != prim::kPrimMakeTuple->name() &&
      prim->name() != prim::kPrimMakeList->name()) {  // Refactoring infer routine soon.
    return std::make_shared<StandardPrimEvaluator>(prim, eval_impl);
  }

  // Use python infer function if the infer function not founded in the map return a python evaluator
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

  // Return a default evaluator
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
  const auto &primitive = func->prim();
  if (func->tracking_id() == 0) {
    // Create primitive evaluator if tracking_id == 0.
    auto [iter, is_new] = evaluators_.emplace(func, nullptr);
    if (is_new) {
      iter->second = GetPrimEvaluator(primitive, shared_from_this());
      if (iter->second == nullptr) {
        MS_LOG(EXCEPTION) << "The evaluator of the primitive is not defined (" << primitive->name() << ").";
      }
    }
    return iter->second;
  }
  // Use TrackedEvaluator if tracking_id != 0.
  auto iter = evaluators_.find(func);
  if (iter != evaluators_.end()) {
    return iter->second;
  }
  auto prim_without_tracking_id = std::make_shared<PrimitiveAbstractClosure>(primitive, 0);
  EvaluatorPtr prim_evaluator = _GetEvaluatorFor(prim_without_tracking_id);
  auto tracked_evaluator = std::make_shared<TrackedEvaluator>(prim_evaluator);
  auto result = evaluators_.emplace(func, std::move(tracked_evaluator));
  return result.first->second;
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<FuncGraphAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  auto [iter, is_new] = evaluators_.emplace(func, nullptr);
  if (is_new) {
    iter->second = std::make_shared<FuncGraphEvaluator>(func->func_graph(), func->context());
  }
  return iter->second;
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<MetaFuncGraphAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  auto [iter, is_new] = evaluators_.emplace(func, nullptr);
  if (is_new) {
    iter->second = std::make_shared<MetaFuncGraphEvaluator>(func->meta_func_graph(), func->GetScope());
  }
  return iter->second;
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<JTransformedAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  const auto &primal_func = func->fn();
  auto primal_evaluator = GetEvaluatorFor(primal_func);
  return std::make_shared<JEvaluator>(primal_evaluator, primal_func);
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<VmapTransformedAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  const auto &primal_func = func->fn();
  const auto &in_axes = func->in_axes();
  const auto &out_axes = func->out_axes();
  auto primal_evaluator = GetEvaluatorFor(primal_func);
  return std::make_shared<VmapEvaluator>(primal_evaluator, primal_func, in_axes, out_axes);
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<TaylorTransformedAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  const auto &primal_func = func->fn();
  auto primal_evaluator = GetEvaluatorFor(primal_func);
  return std::make_shared<TaylorEvaluator>(primal_evaluator, primal_func);
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<ShardTransformedAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  const auto &primal_func = func->fn();
  auto primal_evaluator = GetEvaluatorFor(primal_func);
  return std::make_shared<ShardEvaluator>(primal_evaluator, primal_func);
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<VirtualAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  return std::make_shared<VirtualEvaluator>(func->args_spec_list(), func->output());
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<PartialAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  auto primal_func = func->fn();
  auto part_pair = std::make_pair(primal_func, func->args());
  auto iter = constructors_app_.find(part_pair);
  if (iter != constructors_app_.end()) {
    return iter->second;
  }
  auto primal_evaluator = GetEvaluatorFor(primal_func);
  auto partial_evaluator = std::make_shared<PartialAppEvaluator>(primal_evaluator, func->args());
  auto result = constructors_app_.emplace(std::move(part_pair), std::move(partial_evaluator));
  return result.first->second;
}

EvaluatorPtr AnalysisEngine::GetEvaluatorFor(const AbstractFunctionPtr &func) {
  MS_EXCEPTION_IF_NULL(func);
  MS_LOG(DEBUG) << "GetEvaluatorFor: " << func->ToString() << " tracking_id: " << func->tracking_id();
  if (func->isa<PrimitiveAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<PrimitiveAbstractClosure>(func));
  }
  if (func->isa<FuncGraphAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<FuncGraphAbstractClosure>(func));
  }
  if (func->isa<MetaFuncGraphAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<MetaFuncGraphAbstractClosure>(func));
  }
  if (func->isa<JTransformedAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<JTransformedAbstractClosure>(func));
  }
  if (func->isa<VmapTransformedAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<VmapTransformedAbstractClosure>(func));
  }
  if (func->isa<TaylorTransformedAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<TaylorTransformedAbstractClosure>(func));
  }
  if (func->isa<ShardTransformedAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<ShardTransformedAbstractClosure>(func));
  }
  if (func->isa<VirtualAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<VirtualAbstractClosure>(func));
  }
  if (func->isa<PartialAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<PartialAbstractClosure>(func));
  }
  MS_LOG(EXCEPTION) << "Cannot GetEvaluator from " << func->type_name();
}

EvalResultPtr AnalysisEngine::ForwardConfig(const AnfNodeConfigPtr &orig_conf, const AnfNodeConfigPtr new_conf) {
  MS_EXCEPTION_IF_NULL(orig_conf);
  MS_EXCEPTION_IF_NULL(new_conf);
  // If always_eval_flag is true in BaseFuncGraphEvaluaotr, then the CNode with same orig_conf may be forwarded
  // again, so update the config_map with new_conf;
  anfnode_config_map_[orig_conf] = new_conf;
  MS_LOG(DEBUG) << "Forward orig_conf: " << orig_conf->ToString() << ", to new_conf: " << new_conf->ToString();
  auto old_cnode = orig_conf->node()->cast_ptr<CNode>();
  auto new_cnode = new_conf->node()->cast<CNodePtr>();
  if (old_cnode != nullptr && new_cnode != nullptr) {
    if (old_cnode->func_graph() == new_cnode->func_graph()) {
      MS_LOG(DEBUG) << "Try to remove forward node from order list, forward node: " << new_cnode->DebugString()
                    << ", as origin node should be in order list, origin_node: " << old_cnode->DebugString();
      old_cnode->func_graph()->EraseUnusedNodeInOrder(new_cnode);
    } else {
      MS_LOG(EXCEPTION) << "Forward orig_node to different func_graph, old_node: " << old_cnode->DebugString()
                        << ", new_node: " << new_cnode->DebugString();
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
    auto &eval = evaluators[0];
    MS_EXCEPTION_IF_NULL(eval);
    return eval->Run(shared_from_this(), args_conf_list, out_conf);
  }
  static const bool enable_single_thread = (common::GetEnv("MS_DEV_SINGLE_EVAL") == "1");
  if (enable_single_thread) {
    return ExecuteMultipleEvaluators(evaluators, out_conf, args_conf_list);
  }
  return ExecuteMultipleEvaluatorsMultiThread(evaluators, out_conf, args_conf_list);
}

void AnalysisEngine::SetUndeterminedFlag(const FuncGraphPtr &possible_parent_fg) const {
  MS_EXCEPTION_IF_NULL(possible_parent_fg);
  static std::mutex fg_lock;
  std::lock_guard<std::mutex> infer_lock(fg_lock);
  possible_parent_fg->set_flag(kFuncGraphFlagUndetermined, true);
  MS_LOG(DEBUG) << "Set graph undetermined flag for " << possible_parent_fg->ToString();
}

EvaluatorPtr AnalysisEngine::HandleNestedRecursion(const std::vector<EvaluatorPtr> &evaluators,
                                                   const EvaluatorPtr &eval, const AbstractBasePtrList &args_spec_list,
                                                   const EvalTraceRevIter &it, bool *continue_flag) {
  MS_EXCEPTION_IF_NULL(continue_flag);
  MS_EXCEPTION_IF_NULL(eval);
  *continue_flag = false;
  // Find latest entry function to handle nested recursion.
  EvaluatorPtr latest_entry = eval;
  auto latest_entry_iter = eval_trace_.crbegin();
  for (auto r_it = eval_trace_.crbegin(); *r_it != *it;) {
    auto it_temp = std::find(evaluators.cbegin(), evaluators.cend(), r_it->evaluator_);
    if (it_temp != evaluators.cend()) {
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
  for (auto r_it = eval_trace_.crbegin(); r_it != latest_entry_iter; r_it++) {
    (void)undetermined_evals.insert(*r_it);
  }
  MS_LOG(DEBUG) << "undetermined_evals size(): " << undetermined_evals.size();

  for (const auto &u_eval : undetermined_evals) {
    MS_LOG(DEBUG) << u_eval.evaluator_->ToString() << "check undetermined.";
    auto &alternate_evaluator = multi_poss_[u_eval.evaluator_];
    auto eval_cache = alternate_evaluator->evaluator_cache_mgr();
    const auto &alt_eval_args = EvaluatorArgs(alternate_evaluator, args_spec_list);
    auto is_not_undetermined_eval = (undetermined_evals.find(alt_eval_args) == undetermined_evals.cend());
    auto is_not_continued_eval = (continued_evals_.find(u_eval) == continued_evals_.cend());
    auto args_not_evaluated = (eval_cache->GetValue(args_spec_list) == nullptr);
    if (is_not_undetermined_eval && (args_not_evaluated || is_not_continued_eval)) {
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
  constexpr int recursive_level = 2;
  std::ostringstream buffer;
  buffer << "Cannot join the return values of different branches, perhaps you need to make them equal.\n"
         << error_info << "\n\nInner Message:\nThe abstract type of the return value of the current branch is "
         << spec->ToString() << ", and that of the previous branch is " << last_spec->ToString() << ".\n"
         << "The node is " << node->DebugString(recursive_level);
  if (node->isa<CNode>()) {
    auto cnode = node->cast_ptr<CNode>()->input(0);
    if (IsPrimitiveCNode(cnode, prim::kPrimSwitch)) {
      // {prim::kPrimSwitch, cond, true_branch, false_branch}
      constexpr int true_index = 2;
      constexpr int false_index = 3;
      const auto &inputs = cnode->cast_ptr<CNode>()->inputs();
      buffer << ", true branch: " << inputs.at(true_index)->ToString()
             << ", false branch: " << inputs.at(false_index)->ToString();
    } else if (IsPrimitiveCNode(cnode, prim::kPrimSwitchLayer)) {
      // {prim::kPrimSwitchLayer, X, {prim::kPrimMakeTuple, branch1, branch2, ...}}
      constexpr int branch_index = 2;
      const auto &tuple_node = cnode->cast_ptr<CNode>()->input(branch_index);
      if (IsPrimitiveCNode(tuple_node, prim::kPrimMakeTuple)) {
        const auto &tuple_inputs = tuple_node->cast_ptr<CNode>()->inputs();
        for (size_t i = 1; i < tuple_inputs.size(); i++) {
          buffer << ", branch" << i << ": " << tuple_inputs.at(i);
        }
      }
    }
  }
  buffer << "\n";
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
  for (size_t i = 1; i < out_specs.size(); ++i) {
    const auto &spec = out_specs[i];
    MS_EXCEPTION_IF_NULL(spec);
    try {
      MS_LOG(DEBUG) << "Join node: " << node->DebugString() << ", " << joined_spec->ToString() << ", and "
                    << spec->ToString();
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

namespace {
bool NeedWaitForBranches(const AbstractBasePtr &abstract) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (abstract->isa<AbstractFunction>()) {
    return true;
  }
  if (abstract->isa<AbstractSequence>()) {
    auto elements = abstract->cast_ptr<AbstractSequence>()->elements();
    if (std::any_of(elements.begin(), elements.end(),
                    [](const AbstractBasePtr &item) { return NeedWaitForBranches(item); })) {
      return true;
    }
  }
  return false;
}

void ExecEvaluator(EvaluatorPtr eval, AnalysisEnginePtr engine, ConfigPtrList args_conf_list, AnfNodeConfigPtr out_conf,
                   std::string thread_id, AsyncAbstractPtr async_result_branch, AsyncAbstractPtr async_result_main,
                   AsyncInferTaskPtr async_task, trace::TraceGraphEvalStack graph_evals,
                   trace::TraceCNodeEvalStack trace_c_node_evals) {
  AnalysisSchedule::set_thread_id(thread_id);
  // Restore trace stack for dump stack when there is exception.
  trace::TraceEvalCNodeStackPrepare(trace_c_node_evals);
  trace_c_node_evals.clear();
  trace::TraceGraphEvalStackPrepare(graph_evals);
  graph_evals.clear();

  try {
    // Wait for Signal to run
    MS_LOG(DEBUG) << async_task.get() << "  " << eval->ToString() << " waiting.";
    (void)async_task->GetResult();
    MS_LOG(DEBUG) << async_task.get() << "  " << eval->ToString() << " running.";

    // Acquire GIL for eval to callback python.
    EvalResultPtr result;
    {
      MS_LOG(DEBUG) << std::this_thread::get_id() << " begin.";
      py::gil_scoped_acquire py_guard;
      result = eval->Run(engine, args_conf_list, out_conf);
    }
    MS_LOG(DEBUG) << std::this_thread::get_id() << " end.";
    MS_EXCEPTION_IF_NULL(result);
    MS_EXCEPTION_IF_NULL(result->abstract());

    // Check the branch value to be compatible with the other branch value.
    AnalysisResultCacheMgr::GetInstance().CheckSwitchValueJoinable(out_conf, result->abstract());
    // Broaden the result of switch(c,t,f)()
    auto broaden_abstract = result->abstract()->Broaden();
    // Notify the thread of waiting for branch value and the main thread to continue.
    async_result_branch->set_result(broaden_abstract);
    async_result_main->set_result(broaden_abstract);
    MS_LOG(DEBUG) << GetInferThread() << " async :" << eval->ToString()
                  << " asyncResult address = " << async_result_branch.get()
                  << " value = " << async_result_branch->TryGetResult()->ToString();
  } catch (const std::exception &ex) {
    MS_LOG(INFO) << "Eval node: " << out_conf->node()->ToString() << "  " << eval->ToString() << " threw exception.";
    AnalysisSchedule::GetInstance().HandleException(ex);
  }
  trace::ClearTraceStack();
  ClearThreadLocal();
  MS_LOG(DEBUG) << AnalysisSchedule::thread_id() << " exited.";
  // Thread number will be drop when thread exits.
  AnalysisSchedule::GetInstance().DecreaseThreadCount();
}

AbstractBasePtr BuildAsyncAbstractRecursively(const AbstractBasePtr &orig_abs,
                                              const std::vector<AsyncAbstractPtr> &pending_async_abstract_list,
                                              const std::vector<std::size_t> &index) {
  auto sequence_abs = dyn_cast_ptr<AbstractSequence>(orig_abs);
  if (sequence_abs != nullptr) {
    const auto &orig_elements = sequence_abs->elements();
    AbstractBasePtrList new_elements;
    for (size_t i = 0; i < orig_elements.size(); ++i) {
      if (orig_elements[i]->isa<AbstractFuncAtom>()) {
        AbstractFuncAtomPtrList abs_func_list{orig_elements[i]->cast<AbstractFuncAtomPtr>()};
        for (size_t j = 0; j < pending_async_abstract_list.size(); ++j) {
          std::vector<std::size_t> new_index(index);
          new_index.push_back(i);
          auto async_func = AsyncAbstractFuncAtom::MakeShared(pending_async_abstract_list[j], new_index);
          abs_func_list.push_back(async_func);
        }
        new_elements.push_back(AbstractFunction::MakeAbstractFunction(abs_func_list));
      } else if (orig_elements[i]->isa<AbstractSequence>()) {
        std::vector<std::size_t> new_index(index);
        new_index.push_back(i);
        new_elements.push_back(BuildAsyncAbstractRecursively(orig_elements[i], pending_async_abstract_list, new_index));
      } else {
        new_elements.push_back(orig_elements[i]);
      }
    }
    static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
    AbstractBasePtr new_abs;
    if (orig_abs->isa<AbstractTuple>()) {
      new_abs = std::make_shared<AbstractTuple>(
        new_elements, (enable_eliminate_unused_element ? sequence_abs->sequence_nodes() : nullptr));
    } else if (orig_abs->isa<AbstractList>()) {
      new_abs = std::make_shared<AbstractList>(
        new_elements, (enable_eliminate_unused_element ? sequence_abs->sequence_nodes() : nullptr));
    } else {
      MS_LOG(EXCEPTION) << "FirstResult is not AbstractTuple or AbstractList, but: " << orig_abs->ToString();
    }
    return new_abs;
  }
  MS_LOG(EXCEPTION) << "Orig abstract is not AbstractTuple or AbstractList, but: " << orig_abs->ToString();
}

void BuildPossibleSpecs(const AbstractBasePtr &first_result,
                        const std::vector<AsyncAbstractPtr> &branch_async_abstract_list,
                        AbstractBasePtrList *out_specs) {
  MS_EXCEPTION_IF_NULL(out_specs);
  MS_EXCEPTION_IF_NULL(first_result);
  std::vector<AsyncAbstractPtr> pending_async_abstract_list;
  std::size_t len = branch_async_abstract_list.size();

  for (size_t i = 0; i < len; ++i) {
    auto result = branch_async_abstract_list[i]->TryGetResult();
    if (result) {
      out_specs->push_back(result);
    } else {
      pending_async_abstract_list.push_back(branch_async_abstract_list[i]);
    }
  }
  if (first_result->isa<AbstractFunction>()) {
    for (std::size_t j = 0; j < pending_async_abstract_list.size(); ++j) {
      auto async_func = AsyncAbstractFuncAtom::MakeShared(pending_async_abstract_list[j], std::vector<size_t>{0});
      out_specs->push_back(async_func);
    }
  } else if (first_result->isa<AbstractSequence>()) {
    const auto &new_first_result =
      BuildAsyncAbstractRecursively(first_result, pending_async_abstract_list, std::vector<size_t>());
    MS_LOG(DEBUG) << GetInferThread() << " Try to replace old first with new one, old: " << first_result->ToString()
                  << ", new: " << new_first_result->ToString();
    std::replace_if(
      out_specs->begin(), out_specs->end(), [first_result](const auto &element) { return element == first_result; },
      new_first_result);
  } else {
    MS_LOG(DEBUG) << GetInferThread() << " wait for normal async result";
  }
}
}  // namespace

EvalResultPtr AnalysisEngine::ExecuteMultipleEvaluatorsMultiThread(const std::vector<EvaluatorPtr> &evaluators,
                                                                   const AnfNodeConfigPtr &out_conf,
                                                                   const ConfigPtrList &args_conf_list) {
  MS_EXCEPTION_IF_NULL(out_conf);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  // Release GIL for C++
  MS_LOG(DEBUG) << std::this_thread::get_id() << " begin.";
  py::gil_scoped_release infer_gil_release;

  // Only one thread to run
  AnalysisSchedule::GetInstance().WaitForRun();

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
  AsyncAbstractPtr async_result_main = std::make_shared<AsyncAbstract>();
  // Eval result of the branches
  std::vector<AsyncAbstractPtr> async_result_branches;

  for (auto &evaluator : evaluators) {
    static std::atomic<int> id_count{0};
    std::string thread_id = AnalysisSchedule::thread_id() + "." + std::to_string(id_count.fetch_add(1));
    MS_EXCEPTION_IF_NULL(evaluator);
    SetUndeterminedFlag(possible_parent_fg);
    AsyncAbstractPtr async_result_branch = std::make_shared<AsyncAbstract>();
    // Control the order to run.
    AsyncAbstractPtr control_run_order = std::make_shared<AsyncAbstract>();
    control_run_order->set_result(std::make_shared<AbstractScalar>(1));
    AsyncInferTaskPtr async_task = AsyncInferTask::MakeShared(control_run_order, thread_id);
    AnalysisSchedule::GetInstance().IncreaseThreadCount();
    MS_LOG(DEBUG) << GetInferThread() << "async : " << evaluator->ToString();
    auto thread = std::thread(ExecEvaluator, evaluator, shared_from_this(), args_conf_list, out_conf, thread_id,
                              async_result_branch, async_result_main, async_task, trace::GetCurrentGraphEvalStack(),
                              trace::GetCNodeDebugStack());
    thread.detach();

    // Push to list of running loop
    MS_LOG(DEBUG) << " add to schedule: " << async_task.get();
    AnalysisSchedule::GetInstance().Add2Schedule(async_task);  // Activate order witch child thread.
    (void)async_result_branches.emplace_back(std::move(async_result_branch));
  }

  MS_LOG(DEBUG) << GetInferThread() << "async : wait for one of async to finish.  " << evaluators[0]->ToString()
                << " or  " << evaluators[1]->ToString() << "...";

  auto first_result = async_result_main->GetResult();
  MS_EXCEPTION_IF_NULL(first_result);
  MS_LOG(DEBUG) << GetInferThread() << "async main thread result of " << out_conf->node()->ToString() << " = "
                << first_result->ToString();

  AbstractBasePtrList out_specs;
  size_t len = evaluators.size();
  if (NeedWaitForBranches(first_result)) {
    BuildPossibleSpecs(first_result, async_result_branches, &out_specs);
  } else {
    for (size_t i = 0; i < len; ++i) {
      // Not wait to get the result of branch.
      auto result = async_result_branches[i]->TryGetResult();
      if (result) {
        MS_LOG(DEBUG) << "#" << i << ": " << GetInferThread() << " async get " << evaluators[i]->ToString()
                      << ", result: " << result->ToString() << ", args: " << args_conf_list;
        out_specs.push_back(result);
      }
    }
  }

  MS_LOG(DEBUG) << std::this_thread::get_id() << " finish.";
  const auto &processed_result = ProcessEvalResults(out_specs, out_conf->node());
  if (processed_result != nullptr) {
    // This is the final switch()() value.
    AnalysisResultCacheMgr::GetInstance().SetSwitchValue(out_conf, processed_result->abstract());
  }
  return processed_result;
}

EvalResultPtr AnalysisEngine::ExecuteMultipleEvaluators(const std::vector<EvaluatorPtr> &evaluators,
                                                        const AnfNodeConfigPtr &out_conf,
                                                        const ConfigPtrList &args_conf_list) {
  AbstractBasePtrList out_specs;
  const size_t evaluators_size = 2;
  if (evaluators.size() < evaluators_size) {
    MS_LOG(EXCEPTION) << "Evaluators size is less than 2.";
  }
  multi_poss_[evaluators[0]] = evaluators[1];
  multi_poss_[evaluators[1]] = evaluators[0];
  AbstractBasePtrList args_spec_list = EvaluateArguments(args_conf_list);
  MS_EXCEPTION_IF_NULL(out_conf);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  auto possible_parent_fg = out_conf->node()->func_graph();
  for (auto eval : evaluators) {
    MS_EXCEPTION_IF_NULL(eval);
    SetUndeterminedFlag(possible_parent_fg);
    const auto current_inf = EvaluatorArgs(eval, args_spec_list);
    MS_LOG(DEBUG) << "Check Evaluator " << eval->ToString();
    // If current evaluator is under tracing, then skip current evaluator to avoid recursively evaluating.
    auto it = std::find(eval_trace_.crbegin(), eval_trace_.crend(), current_inf);
    if (it == eval_trace_.crend()) {
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

AbstractBasePtr MakeAbstractClosure(const FuncGraphPtr &func_graph, const AnalysisContextPtr &context,
                                    const AnfNodePtr &anf_node) {
  AnalysisContextPtr temp_context = context;
  if (temp_context == nullptr) {
    temp_context = AnalysisContext::DummyContext();
  }
  return std::make_shared<FuncGraphAbstractClosure>(func_graph, temp_context, anf_node);
}

AbstractBasePtr MakeAbstractClosure(const MetaFuncGraphPtr &meta_func_graph, const AnfNodePtr &anf_node) {
  MetaFuncGraphAbstractClosurePtr meta_func_graph_fn;
  if (anf_node == nullptr) {
    meta_func_graph_fn = std::make_shared<MetaFuncGraphAbstractClosure>(meta_func_graph);
  } else {
    meta_func_graph_fn = std::make_shared<MetaFuncGraphAbstractClosure>(meta_func_graph, anf_node, anf_node->scope());
  }
  return meta_func_graph_fn;
}

AbstractBasePtr MakeAbstractClosure(const PrimitivePtr &primitive, const AnfNodePtr &anf_node) {
  auto prim_func = std::make_shared<PrimitiveAbstractClosure>(primitive, anf_node);
  return prim_func;
}

AbstractBasePtr ToAbstract(const ValuePtr &value, const AnalysisContextPtr &context, const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(value);
  AnfNodePtr anf_node = nullptr;
  if (conf != nullptr) {
    anf_node = conf->node();
  }
  if (value->isa<Primitive>()) {
    auto prim = value->cast<PrimitivePtr>();
    return MakeAbstractClosure(prim, anf_node);
  }
  if (value->isa<FuncGraph>()) {
    auto func_graph = value->cast<FuncGraphPtr>();
    return MakeAbstractClosure(func_graph, context, anf_node);
  }
  if (value->isa<MetaFuncGraph>()) {
    auto meta_func_graph = value->cast<MetaFuncGraphPtr>();
    return MakeAbstractClosure(meta_func_graph, anf_node);
  }
  static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
  if (enable_eliminate_unused_element && value->isa<ValueSequence>()) {
    auto abs = value->ToAbstract();
    auto sequence_abs = dyn_cast_ptr<AbstractSequence>(abs);
    MS_EXCEPTION_IF_NULL(sequence_abs);
    if (anf_node != nullptr) {
      SetSequenceNodeElementsUseFlags(anf_node, std::make_shared<std::vector<bool>>(sequence_abs->elements().size()));
      std::shared_ptr<AnfNodeWeakPtrList> sequence_nodes = std::make_shared<AnfNodeWeakPtrList>();
      (void)sequence_nodes->emplace_back(AnfNodeWeakPtr(anf_node));
      sequence_abs->set_sequence_nodes(sequence_nodes);
    }
    return abs;
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
  if (evaluator == nullptr) {
    MS_LOG(EXCEPTION) << "The evaluator of the primitive is not defined (" << primitive->name() << ").";
  }
  auto trivial_evaluator = dyn_cast_ptr<TrivialPrimEvaluator>(evaluator);
  if (trivial_evaluator != nullptr) {
    return trivial_evaluator->EvalPrim(nullptr, arg_specs);
  }
  // Support MakeTuple/MakeList ops in PyNative mode.
  auto transition_evaluator = dyn_cast_ptr<TransitionPrimEvaluator>(evaluator);
  if (transition_evaluator != nullptr &&
      (transition_evaluator->isa<MakeTupleEvaluator>() || transition_evaluator->isa<MakeListEvaluator>())) {
    return transition_evaluator->EvalPrim(nullptr, arg_specs, nullptr, nullptr);
  }
  MS_LOG(EXCEPTION) << "The primitive '" << primitive->ToString() << "' should be built as a TrivialPrimEvaluator, but "
                    << evaluator->ToString();
}
}  // namespace abstract
}  // namespace mindspore
