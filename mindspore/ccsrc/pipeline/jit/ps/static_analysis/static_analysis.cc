/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/ps/static_analysis/static_analysis.h"
#include <algorithm>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_set>
#include <utility>
#include <atomic>
#include "mindspore/core/ops/structure_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "abstract/abstract_value.h"
#include "pipeline/jit/ps/fallback.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "pipeline/jit/ps/static_analysis/prim.h"
#include "frontend/operator/ops.h"
#include "utils/ms_exception.h"
#include "ir/func_graph_cloner.h"
#include "pipeline/jit/ps/static_analysis/evaluator.h"
#include "pipeline/jit/ps/debug/trace.h"
#include "include/common/fallback.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/ps/static_analysis/async_eval_result.h"
#include "frontend/operator/ops_front_infer_function.h"

namespace mindspore {
namespace abstract {
// Record current depth of function call stack, including `stack_frame_depth`.
std::atomic<size_t> function_call_depth;
// Record current depth of stack frames call.
std::atomic<size_t> stack_frame_depth;

void ResetFunctionCallDepth() { function_call_depth = 0; }

void IncreaseFunctionCallDepth() { (void)(++function_call_depth); }

void DecreaseFunctionCallDepth() {
  if (function_call_depth == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "Current function call depth is already 0, can not decrease it.";
  }
  function_call_depth--;
}

size_t FunctionCallDepth() { return function_call_depth; }

void ResetStackFrameDepth() { stack_frame_depth = 0; }

void IncreaseStackFrameDepth() { (void)(++stack_frame_depth); }

void DecreaseStackFrameDepth() {
  if (stack_frame_depth == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "Current stack frame depth is already 0, can not decrease it.";
  }
  stack_frame_depth--;
}

size_t StackFrameDepth() { return stack_frame_depth; }

namespace {
void ExecEvaluator(EvaluatorPtr eval, AnalysisEnginePtr engine, ConfigPtrList args_conf_list, AnfNodeConfigPtr out_conf,
                   std::string thread_id, AsyncAbstractPtr async_result_branch, AsyncAbstractPtr async_result_main,
                   AsyncInferTaskPtr async_task, trace::TraceGraphEvalStack graph_evals,
                   trace::TraceCNodeEvalStack trace_c_node_evals) {
  MS_EXCEPTION_IF_NULL(eval);
  MS_EXCEPTION_IF_NULL(async_task);
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
      MS_LOG(DEBUG) << eval->ToString() << "_" << AnalysisSchedule::thread_id() << " begin.";
      py::gil_scoped_acquire py_guard;
      result = eval->Run(engine, args_conf_list, out_conf);
    }
    MS_LOG(DEBUG) << eval->ToString() << "_" << AnalysisSchedule::thread_id() << " end.";
    MS_EXCEPTION_IF_NULL(result);
    MS_EXCEPTION_IF_NULL(result->abstract());

    // Check the branch value to be compatible with the other branch value.
    AnalysisResultCacheMgr::GetInstance().CheckSwitchValueJoinable(out_conf, result->abstract());
    // Broaden the result of switch(c,t,f)()
    auto broaden_abstract = result->abstract()->Broaden();

    MS_EXCEPTION_IF_NULL(async_result_branch);
    MS_EXCEPTION_IF_NULL(async_result_main);
    // Notify the thread of waiting for branch value and the main thread to continue.
    async_result_branch->set_result(broaden_abstract);
    async_result_main->set_result(broaden_abstract);
    MS_LOG(DEBUG) << GetInferThread() << " async :" << eval->ToString()
                  << " asyncResult address = " << async_result_branch.get();
    if (async_result_branch->TryGetResult()) {
      MS_LOG(DEBUG) << "value = " << (async_result_branch->TryGetResult())->ToString();
    } else {
      MS_LOG(DEBUG) << "value = null.";
    }
  } catch (const std::exception &ex) {
    MS_EXCEPTION_IF_NULL(out_conf->node());
    MS_LOG(INFO) << GetInferThread() << "Eval node: " << out_conf->node()->ToString() << "  " << eval->ToString()
                 << " threw exception: " << ex.what();
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
  MS_EXCEPTION_IF_NULL(orig_abs);
  auto sequence_abs = dyn_cast_ptr<AbstractSequence>(orig_abs);
  if (sequence_abs != nullptr) {
    const auto &orig_elements = sequence_abs->elements();
    AbstractBasePtrList new_elements;
    for (size_t i = 0; i < orig_elements.size(); ++i) {
      MS_EXCEPTION_IF_NULL(orig_elements[i]);
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
      MS_LOG(INTERNAL_EXCEPTION) << "FirstResult is not AbstractTuple or AbstractList, but: " << orig_abs->ToString();
    }
    return new_abs;
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Orig abstract is not AbstractTuple or AbstractList, but: " << orig_abs->ToString();
}

void BuildPossibleSpecs(const AbstractBasePtr &first_result,
                        const std::vector<AsyncAbstractPtr> &branch_async_abstract_list,
                        AbstractBasePtrList *out_abs_list) {
  MS_EXCEPTION_IF_NULL(out_abs_list);
  MS_EXCEPTION_IF_NULL(first_result);
  std::vector<AsyncAbstractPtr> pending_async_abstract_list;
  std::size_t len = branch_async_abstract_list.size();

  for (size_t i = 0; i < len; ++i) {
    AbstractBasePtr result;
    MS_EXCEPTION_IF_NULL(branch_async_abstract_list[i]);
    if (enable_waiting_branch_eval()) {
      result = branch_async_abstract_list[i]->GetResult();
    } else {
      result = branch_async_abstract_list[i]->TryGetResult();
    }

    if (result) {
      if (result->isa<AsyncAbstractFuncAtom>()) {
        branch_async_abstract_list[i]->ClearPossibleResult();
        pending_async_abstract_list.push_back(branch_async_abstract_list[i]);
        MS_LOG(DEBUG) << "Pending add: " << branch_async_abstract_list[i].get() << "_"
                      << branch_async_abstract_list[i]->ToString();
      } else {
        out_abs_list->push_back(result);
      }
    } else {
      pending_async_abstract_list.push_back(branch_async_abstract_list[i]);
      MS_LOG(DEBUG) << "Pending add: " << branch_async_abstract_list[i].get() << "_"
                    << branch_async_abstract_list[i]->ToString();
    }
  }

  if (first_result->isa<AbstractFunction>()) {
    for (std::size_t j = 0; j < pending_async_abstract_list.size(); ++j) {
      auto async_func = AsyncAbstractFuncAtom::MakeShared(pending_async_abstract_list[j], std::vector<size_t>{0});
      out_abs_list->push_back(async_func);
      MS_LOG(DEBUG) << "out_abs_list add: " << async_func.get() << "_" << async_func->ToString();
    }
  } else if (first_result->isa<AbstractSequence>()) {
    const auto &new_first_result =
      BuildAsyncAbstractRecursively(first_result, pending_async_abstract_list, std::vector<size_t>());
    MS_LOG(DEBUG) << GetInferThread() << " Try to replace old first with new one, old: " << first_result->ToString()
                  << ", new: " << new_first_result->ToString();
    std::replace_if(
      out_abs_list->begin(), out_abs_list->end(),
      [first_result](const auto &element) { return element == first_result; }, new_first_result);
  } else {
    MS_LOG(DEBUG) << GetInferThread() << " wait for normal async result";
  }
}

EvalResultPtr ConvertToPyExecuteCall(const CNodePtr &cnode, const AnfNodeConfigPtr &conf) {
  constexpr auto internal_callable_obj_str = "__internal_callable_obj__";
  std::stringstream script_buffer;
  script_buffer << internal_callable_obj_str << "(";

  std::vector<ValuePtr> key_list;
  const auto callable_obj_name_str = std::make_shared<StringImm>(internal_callable_obj_str);
  (void)key_list.emplace_back(callable_obj_name_str);
  constexpr auto internal_callable_input_str = "__internal_callable_obj_input__";
  const auto &inputs = cnode->inputs();
  for (size_t i = 1; i < inputs.size(); ++i) {
    std::stringstream key_input_buffer;
    key_input_buffer << internal_callable_input_str << i;
    (void)key_list.emplace_back(std::make_shared<StringImm>(key_input_buffer.str()));
    script_buffer << key_input_buffer.str();
    if (i < inputs.size() - 1) {
      script_buffer << ", ";
    }
  }
  script_buffer << ")";
  const auto key_tuple = std::make_shared<ValueTuple>(key_list);
  const auto script_call_str = std::make_shared<StringImm>(script_buffer.str());

  std::vector<AnfNodePtr> value_list{NewValueNode(prim::kPrimMakeTuple)};
  (void)std::copy(inputs.begin(), inputs.end(), std::back_inserter(value_list));
  auto fg = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  const auto value_tuple_node = fg->NewCNode(value_list);

  const auto obj_call_node =
    fallback::CreatePyExecuteCNode(cnode, NewValueNode(script_call_str), NewValueNode(key_tuple), value_tuple_node);
  constexpr auto recursive_level = 2;
  MS_LOG(DEBUG) << "Created obj_call_node: " << obj_call_node->DebugString(recursive_level);
  AnalysisEnginePtr eng = conf->engine();
  MS_EXCEPTION_IF_NULL(eng);
  AnfNodeConfigPtr fn_conf = eng->MakeConfig(obj_call_node, conf->context(), conf->func_graph());
  return eng->ForwardConfig(conf, fn_conf);
}

EvalResultPtr ConvertClassTypeToFunc(const CNodePtr &cnode, const AbstractBasePtr &abs, const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(abs);
  auto val = abs->BuildValue();
  MS_EXCEPTION_IF_NULL(val);
  auto class_val = dyn_cast_ptr<parse::ClassType>(val);
  MS_EXCEPTION_IF_NULL(class_val);
  const auto &class_name = class_val->name();
  auto class_obj = class_val->obj();
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  auto py_fn =
    python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_CONVERT_CLASS_TO_FUNCTION, py::str(class_name), class_obj);
  if (py::isinstance<py::none>(py_fn)) {
    const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() == kLax);
    if (allow_fallback_runtime) {
      return ConvertToPyExecuteCall(cnode, conf);
    }
    MS_LOG(ERROR) << "Can not cast to a AbstractFunction from " << abs->ToString() << ".";
    MS_LOG(ERROR) << "It's called at: " << cnode->DebugString();
    MS_EXCEPTION(ValueError) << "Can not call " << class_name << " to create python object in graph mode. "
                             << "Try using 'jit_class' to decorate the class?";
  }
  auto list_func_fg = parse::ParsePythonCode(py_fn);
  MS_EXCEPTION_IF_NULL(list_func_fg);
  auto fg = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  list_func_fg->set_manager(fg->manager());

  auto &inputs = cnode->inputs();
  std::vector<AnfNodePtr> new_cnode_inputs;
  (void)new_cnode_inputs.emplace_back(NewValueNode(list_func_fg));
  for (std::size_t i = 1; i < inputs.size(); ++i) {
    (void)new_cnode_inputs.emplace_back(inputs[i]);
  }
  auto new_cnode = fg->NewCNodeInOrder(new_cnode_inputs);
  new_cnode->set_debug_info(cnode->debug_info());

  AnalysisEnginePtr eng = conf->engine();
  MS_EXCEPTION_IF_NULL(eng);
  AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_cnode, conf->context(), conf->func_graph());
  return eng->ForwardConfig(conf, fn_conf);
}

EvalResultPtr ConvertMsClassObjToFunc(const CNodePtr &cnode, const AbstractBasePtr &abs, const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(abs);
  auto val = abs->BuildValue();
  MS_EXCEPTION_IF_NULL(val);
  auto class_val = dyn_cast_ptr<parse::MsClassObject>(val);
  MS_EXCEPTION_IF_NULL(class_val);
  py::object cls_obj = class_val->obj();
  const std::string call_func_name = "__call__";
  if (!py::hasattr(cls_obj, common::SafeCStr(call_func_name))) {
    MS_EXCEPTION(ValueError) << class_val->name() << " has no " << call_func_name
                             << " function, please check the code.";
  }
  py::object call_obj = py::getattr(cls_obj, common::SafeCStr(call_func_name));
  FuncGraphPtr call_func_graph = parse::ConvertToFuncGraph(call_obj);
  if (call_func_graph == nullptr) {
    MS_EXCEPTION(TypeError) << "Expect a function type, but got " << py::str(call_obj) << ".";
  }
  auto fg = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  call_func_graph->set_manager(fg->manager());

  auto &inputs = cnode->inputs();
  std::vector<AnfNodePtr> new_cnode_inputs;
  (void)new_cnode_inputs.emplace_back(NewValueNode(call_func_graph));
  for (std::size_t i = 1; i < inputs.size(); ++i) {
    (void)new_cnode_inputs.emplace_back(inputs[i]);
  }
  auto new_cnode = fg->NewCNodeInOrder(new_cnode_inputs);
  new_cnode->set_debug_info(cnode->debug_info());

  AnalysisEnginePtr eng = conf->engine();
  MS_EXCEPTION_IF_NULL(eng);
  AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_cnode, conf->context(), conf->func_graph());
  return eng->ForwardConfig(conf, fn_conf);
}

bool CheckFuncSideEffect(const AbstractFunctionPtr &func) {
  // Check if func graph contains isolated side-effect, and sync.
  auto func_graph_abs = dyn_cast_ptr<FuncGraphAbstractClosure>(func);
  if (func_graph_abs != nullptr) {
    MS_EXCEPTION_IF_NULL(func_graph_abs->func_graph());
    return func_graph_abs->func_graph()->has_side_effect_node();
  } else {
    auto meta_func_graph_abs = dyn_cast_ptr<MetaFuncGraphAbstractClosure>(func);
    if (meta_func_graph_abs != nullptr) {
      MS_EXCEPTION_IF_NULL(meta_func_graph_abs->meta_func_graph());
      return meta_func_graph_abs->meta_func_graph()->has_side_effect_node();
    }
  }
  return false;
}

AbstractFuncAtomPtr GetRealFuncAtom(const AbstractFuncAtomPtr &possible_func) {
  MS_EXCEPTION_IF_NULL(possible_func);
  auto real_atom = possible_func;
  const auto &async_abs_func = possible_func->cast_ptr<AsyncAbstractFuncAtom>();
  if (async_abs_func != nullptr) {
    auto real_func = async_abs_func->GetUnique();
    real_atom = dyn_cast<AbstractFuncAtom>(real_func);
    MS_EXCEPTION_IF_NULL(real_atom);
    MS_LOG(DEBUG) << "Real AsyncAbstractFuncAtom is: " << real_atom->ToString();
  }
  return real_atom;
}
}  // namespace

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

AnalysisResult AnalysisEngine::Run(const FuncGraphPtr &func_graph, const AbstractBasePtrList &args_abs_list) {
  StaticAnalysisException::Instance().ClearException();
  AnalysisResult result;
  try {
    MS_EXCEPTION_IF_NULL(func_graph);
    ConfigPtrList args_conf_list;
    (void)std::transform(args_abs_list.begin(), args_abs_list.end(), std::back_inserter(args_conf_list),
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

EvalResultPtr ObtainEvalResultFromCache(const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  static AnalysisResultCacheMgr &cache_mgr = AnalysisResultCacheMgr::GetInstance();
  auto result = cache_mgr.GetValue(conf);
  if (result != nullptr) {
    MS_EXCEPTION_IF_NULL(result->abstract());
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
    MS_LOG(INTERNAL_EXCEPTION) << "Evaluate for NodeConfig " << conf->ToString() << " get nullptr";
  }
  MS_EXCEPTION_IF_NULL(result->abstract());
  MS_LOG(DEBUG) << "Always Evaluate node for NodeConfig: " << conf->ToString()
                << ", result: " << result->abstract().get() << "/" << result->abstract()->ToString();
  SaveEvalResultInCache(conf, result);
  return result;
}

void AnalysisEngine::SaveEvalResultInCache(const AnfNodeConfigPtr &conf, const EvalResultPtr &result) const {
  MS_EXCEPTION_IF_NULL(conf);
  MS_EXCEPTION_IF_NULL(result);
  static AnalysisResultCacheMgr &cache_mgr = AnalysisResultCacheMgr::GetInstance();
  auto iter = cache_mgr.GetCache().find(conf);
  if (iter != cache_mgr.GetCache().end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    MS_EXCEPTION_IF_NULL(iter->second->abstract());
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
  MS_EXCEPTION_IF_NULL(result->abstract());
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
  MS_EXCEPTION_IF_NULL(evaluator->evaluator_cache_mgr());
  auto &cache = evaluator->evaluator_cache_mgr()->GetCache();
  auto iter = cache.find(args_abs_list);
  if (iter != cache.end()) {
    MS_EXCEPTION_IF_NULL(fg_context);
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
    MS_LOG(DEBUG) << "Begin Eval CNode: " << cnode->DebugString();
    eval_result = EvalCNode(cnode, conf);
    MS_LOG(DEBUG) << "End Eval CNode: " << cnode->DebugString();
    trace::TraceEvalCNodeLeave();
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Illegal AnfNode for evaluating, node: " << node->DebugString()
                               << "(type:" << node->type_name() << "), fg: "
                               << (node->func_graph() != nullptr ? node->func_graph()->ToString() : "nullgraph")
                               << " conf: " << conf->ToString();
  }

#ifdef DEBUG
  compute_conf_stack_.pop_back();
  if (eval_result == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Compute Config failed, node: " << node->DebugString()
                               << " NodeInfo: " << trace::GetDebugInfo(node->debug_info());
  }
#endif
  MS_EXCEPTION_IF_NULL(eval_result->abstract());
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

AnfNodeConfigPtr AnalysisEngine::GetForwardConfig(const AnfNodeConfigPtr &conf) const {
  MS_EXCEPTION_IF_NULL(conf);
  AnfNodeConfigPtr new_conf = conf;
  auto conf_iter = anfnode_config_map().find(conf);
  while (conf_iter != anfnode_config_map().end()) {
    new_conf = conf_iter->second;
    MS_EXCEPTION_IF_NULL(new_conf);
    conf_iter = anfnode_config_map().find(new_conf);
  }
  return new_conf;
}

EvalResultPtr AnalysisEngine::InterpretedNodeCall(const CNodePtr &cnode, const AnfNodeConfigPtr &conf) {
  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() == kLax);
  if (!allow_fallback_runtime) {
    return nullptr;
  }

  MS_EXCEPTION_IF_NULL(cnode);
  auto &inputs = cnode->inputs();
  if (inputs.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "CNode inputs should not be empty, CNode: " << cnode->DebugString();
  }

  // Check if the operator input is PyExecute CNode.
  auto &func_node = inputs[0];
  MS_EXCEPTION_IF_NULL(func_node);
  constexpr auto recursive_level = 2;
  MS_LOG(DEBUG) << "Current CNode: " << cnode->DebugString(recursive_level)
                << ", func_node: " << func_node->DebugString(recursive_level);
  auto prim = GetCNodePrimitiveWithoutDoSignature(func_node);
  if (!IsPrimitiveEquals(prim, prim::kPrimGetAttr) && !IsPrimitiveEquals(prim, prim::kPrimPyExecute) &&
      !IsPrimitiveEquals(prim, prim::kPrimPyInterpret)) {
    // Optimize the performance.
    return nullptr;
  }
  AnfNodeConfigPtr func_conf = MakeConfig(func_node, conf->context(), conf->func_graph());
  MS_EXCEPTION_IF_NULL(func_conf);
  const auto &forwarded_conf = GetForwardConfig(func_conf);
  if (!IsPrimitiveCNode(forwarded_conf->node(), prim::kPrimPyExecute) &&
      !IsPrimitiveCNode(forwarded_conf->node(), prim::kPrimPyInterpret)) {
    return nullptr;
  }

  // Forward getattr CNode call to py_execute CNode.
  return ConvertToPyExecuteCall(cnode, conf);
}

AbstractBasePtr AnalysisEngine::GetCNodeOperatorAbstract(const CNodePtr &cnode, const AnalysisContextPtr &context,
                                                         const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto &inputs = cnode->inputs();
  if (inputs.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "CNode inputs should not be empty, CNode: " << cnode->DebugString();
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
    MS_LOG(INTERNAL_EXCEPTION) << "No abstract, func_conf: " << func_conf->ToString();
  }
  return possible_func;
}

EvalResultPtr AnalysisEngine::EvalCNode(const CNodePtr &cnode, const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  MS_EXCEPTION_IF_NULL(cnode);

  // Handle the interpreted node call here.
  const auto &interpreted_eval_result = InterpretedNodeCall(cnode, conf);
  if (interpreted_eval_result != nullptr) {
    return interpreted_eval_result;
  }

  AbstractBasePtr possible_func = GetCNodeOperatorAbstract(cnode, conf->context(), conf->func_graph());
  MS_EXCEPTION_IF_NULL(possible_func->BuildType());
  if (possible_func->BuildType()->type_id() == kObjectTypeUndeterminedType) {
    MS_LOG(DEBUG) << "EvalCNode eval Undetermined";
    return std::make_shared<EvalResult>(possible_func->Clone(), std::make_shared<AttrValueMap>());
  }

  if (possible_func->isa<AbstractClass>()) {
    return ConvertMsClassObjToFunc(cnode, possible_func, conf);
  }
  if (possible_func->isa<AbstractScalar>()) {
    // Convert class to function, such as list(xxx).
    auto val = possible_func->BuildValue();
    MS_EXCEPTION_IF_NULL(val);
    if (val->isa<parse::ClassType>()) {
      return ConvertClassTypeToFunc(cnode, possible_func, conf);
    }
    if (val->isa<parse::InterpretedObject>()) {
      MS_LOG(ERROR) << "Do not support " << val->ToString() << " as a function.\n"
                    << "If it is a function from a module outside the project root directory and it needs to be run as "
                    << "a static computation graph, try setting: 'export MS_JIT_MODULES=module1_name,module2_name,...'";
    }
  }

  auto func = dyn_cast_ptr<AbstractFunction>(possible_func);
  if (func == nullptr) {
    MS_LOG(ERROR) << "Can not cast to a AbstractFunction from " << possible_func->ToString() << ".";
    MS_LOG(ERROR) << "It's called at: " << cnode->DebugString();
    MS_EXCEPTION(ValueError) << "The object is not callable. Please check code.";
  }

  // Make arguments config list.
  bool contains_side_effect = false;
  auto &inputs = cnode->inputs();
  const auto inputs_size = inputs.size();
  ConfigPtrList args_conf_list;
  args_conf_list.reserve(inputs_size - 1);
  // Ignore the first node which is function name.
  for (std::size_t i = 1; i < inputs_size; ++i) {
    const AnfNodePtr &node = inputs[i];
    (void)args_conf_list.emplace_back(MakeConfig(node, conf->context(), conf->func_graph()));
    if (check_side_effect()) {
      auto input_cnode = dyn_cast_ptr<CNode>(node);
      if (input_cnode != nullptr) {
        contains_side_effect = contains_side_effect || input_cnode->has_side_effect_node();
      }
    }
  }

  // Find evaluators.
  std::vector<EvaluatorPtr> evaluators;
  func->Visit([this, &evaluators, &cnode](const AbstractFuncAtomPtr &possible_func) {
    const auto &real_func_atom = GetRealFuncAtom(possible_func);
    auto evaluator = this->GetEvaluatorFor(real_func_atom);
    evaluator->set_bound_node(cnode);
    (void)evaluators.emplace_back(std::move(evaluator));
  });

  // Run evaluators.
  auto eval_result = ExecuteEvaluators(evaluators, conf, args_conf_list);
  // Check if func graph contains isolated side-effect, and sync.
  if (check_side_effect()) {
    func->Visit([&contains_side_effect](const AbstractFuncAtomPtr &possible_func) {
      const auto &real_func_atom = GetRealFuncAtom(possible_func);
      bool func_has_side_effect = CheckFuncSideEffect(real_func_atom);
      if (func_has_side_effect) {
        contains_side_effect = true;
      }
    });
    if (contains_side_effect) {
      MS_EXCEPTION_IF_NULL(conf->func_graph());
      MS_LOG(DEBUG) << "Found side-effect, cnode: " << cnode->DebugString()
                    << ", func_graph: " << conf->func_graph()->ToString();
      cnode->set_has_side_effect_node(true);
      conf->func_graph()->set_has_side_effect_node(true);
      eval_result->set_has_side_effect_node(true);
    }
  }
  return eval_result;
}

EvalResultPtr AnalysisEngine::Execute(const AbstractFunctionPtr &func, const AbstractBasePtrList &args_abs_list) {
  MS_EXCEPTION_IF_NULL(func);
  ConfigPtrList args_conf_list;
  (void)std::transform(args_abs_list.begin(), args_abs_list.end(), std::back_inserter(args_conf_list),
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
    if (evaluator == nullptr || evaluator->evaluator_cache_mgr() == nullptr) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(evaluator->evaluator_cache_mgr());
    evaluator->evaluator_cache_mgr()->Clear();
  }
  for (auto &element : prim_constructors_) {
    EvaluatorPtr evaluator = element.second;
    if (evaluator == nullptr || evaluator->evaluator_cache_mgr() == nullptr) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(evaluator->evaluator_cache_mgr());
    evaluator->evaluator_cache_mgr()->Clear();
  }
  for (auto &element : prim_py_evaluators_) {
    EvaluatorPtr evaluator = element.second;
    if (evaluator == nullptr || evaluator->evaluator_cache_mgr() == nullptr) {
      continue;
    }
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
  prim_py_evaluators_.clear();
  constructors_app_.clear();
  continued_evals_.clear();
  root_context_ = nullptr;
}

EvaluatorPtr GetPyEvaluator(const PrimitivePtr &prim, const AnalysisEnginePtr &engine) {
  auto prim_py = dyn_cast<PrimitivePy>(prim);
  if (prim_py != nullptr) {
    auto is_constexpr = prim_py->HasAttr(GRAPH_FLAG_CONSTEXPR_PRIM);
    if (is_constexpr) {
      return std::make_shared<ConstexprEvaluator>(prim_py);
    }
    if (engine == nullptr) {
      return std::make_shared<PythonPrimEvaluator>(prim_py);
    }

    const auto &iter = engine->prim_py_evaluators_.find(prim_py);
    if (iter != engine->prim_py_evaluators_.end()) {
      return iter->second;
    }
    auto evaluator = std::make_shared<PythonPrimEvaluator>(prim_py);
    engine->prim_py_evaluators_[prim_py] = evaluator;
    return evaluator;
  }
  MS_LOG(ERROR) << "The primitive with python evaluator should be a python primitive.";
  return nullptr;
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
  if (IsPrimitiveEquals(prim, prim::kPrimMixedPrecisionCast)) {
    return std::make_shared<MixedPrecisionCastEvaluator>(prim);
  }
  if (IsPrimitiveEquals(prim, prim::kPrimPyExecute)) {
    return std::make_shared<PyExecuteEvaluator>();
  }
  static const bool enable_pre_lift = (common::GetEnv("MS_DEV_PRE_LIFT") == "1");
  if (enable_pre_lift && IsPrimitiveEquals(prim, prim::kPrimSwitch)) {
    return std::make_shared<SwitchEvaluator>();
  }

  // Find prim infer function in the prim function map return a standard evaluator
  auto eval_impl_opt = GetFrontendPrimitiveInferImpl(prim);
  if (eval_impl_opt.has_value()) {
    auto eval_impl = eval_impl_opt.value();
    if (eval_impl.IsImplInferShapeAndType() && !IsPrimitiveEquals(prim, prim::kPrimMakeTuple) &&
        !IsPrimitiveEquals(prim, prim::kPrimMakeList)) {
      return std::make_shared<StandardPrimEvaluator>(prim, eval_impl);
    }
  }

  // Use python infer function if the infer function not founded in the map return a python evaluator
  EvaluatorPtr evaluator = nullptr;
  if (prim->HasPyEvaluator()) {
    return GetPyEvaluator(prim, engine);
  }

  // Delete this when the infer value can be mapped to the CPU backend operator.
  if (PrimNeedFrontendInferValue(prim)) {
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
        MS_LOG(EXCEPTION) << "Operator '" << primitive->name() << "' is invalid.";
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
  static const bool enable_pre_lift = (common::GetEnv("MS_DEV_PRE_LIFT") == "1");
  if (enable_pre_lift && IsPrimitiveEquals(primitive, prim::kPrimSwitch)) {
    auto result = evaluators_.emplace(func, prim_evaluator);
    return result.first->second;
  } else {
    auto tracked_evaluator = std::make_shared<TrackedEvaluator>(prim_evaluator);
    auto result = evaluators_.emplace(func, std::move(tracked_evaluator));
    return result.first->second;
  }
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
  size_t cell_size = func->cell_size();
  auto primal_evaluator = GetEvaluatorFor(primal_func);
  return std::make_shared<VmapEvaluator>(primal_evaluator, primal_func, in_axes, out_axes, cell_size);
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
  return std::make_shared<VirtualEvaluator>(func->args_abs_list(), func->output());
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

  MS_LOG(INTERNAL_EXCEPTION) << "Cannot GetEvaluator from " << func->type_name();
}

EvalResultPtr AnalysisEngine::ForwardConfig(const AnfNodeConfigPtr &orig_conf, const AnfNodeConfigPtr new_conf) {
  MS_EXCEPTION_IF_NULL(orig_conf);
  MS_EXCEPTION_IF_NULL(new_conf);
  // If always_eval_flag is true in BaseFuncGraphEvaluaotr, then the CNode with same orig_conf may be forwarded
  // again, so update the config_map with new_conf;
  anfnode_config_map_[orig_conf] = new_conf;
  MS_LOG(DEBUG) << "Forward orig_conf: " << orig_conf->ToString() << ", to new_conf: " << new_conf->ToString();
  MS_EXCEPTION_IF_NULL(orig_conf->node());
  MS_EXCEPTION_IF_NULL(new_conf->node());
  auto old_cnode = orig_conf->node()->cast_ptr<CNode>();
  auto new_cnode = new_conf->node()->cast<CNodePtr>();
  if (old_cnode != nullptr && new_cnode != nullptr) {
    if (old_cnode->func_graph() == new_cnode->func_graph()) {
      MS_LOG(DEBUG) << "Try to remove forward node from order list, forward node: " << new_cnode->DebugString()
                    << ", as origin node should be in order list, origin_node: " << old_cnode->DebugString();
      old_cnode->func_graph()->EraseUnusedNodeInOrder(new_cnode);
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "Forward orig_node to different func_graph, old_node: " << old_cnode->DebugString()
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

void AnalysisEngine::SetUndeterminedFlag(const std::string &thread_id, const FuncGraph &fg) {
  static std::mutex fg_lock;
  std::lock_guard<std::mutex> infer_lock(fg_lock);
  MS_LOG(DEBUG) << "Record undetermined flag of fg:" << fg.ToString() << ", thread id:" << thread_id;
  func_graph_undetermined_flags_[&fg].push_front(thread_id);
}

void AnalysisEngine::SetIgnoreValueFlag(const std::string &thread_id, FuncGraph *fg) {
  MS_EXCEPTION_IF_NULL(fg);
  auto it = func_graph_undetermined_flags_.find(fg);
  if (it == func_graph_undetermined_flags_.cend()) {
    return;
  }
  for (const auto &id : it->second) {
    if (thread_id.find(id) != std::string::npos && thread_id != id) {
      MS_LOG(DEBUG) << "Set ignore value of fg:" << fg->ToString() << ", thread id:" << thread_id;
      fg->set_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE, true);
      return;
    }
  }
}

EvaluatorPtr AnalysisEngine::HandleNestedRecursion(const std::vector<EvaluatorPtr> &evaluators,
                                                   const EvaluatorPtr &eval, const AbstractBasePtrList &args_abs_list,
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
    MS_EXCEPTION_IF_NULL(u_eval.evaluator_);
    MS_LOG(DEBUG) << u_eval.evaluator_->ToString() << "check undetermined.";
    auto &alternate_evaluator = multi_poss_[u_eval.evaluator_];
    MS_EXCEPTION_IF_NULL(alternate_evaluator);
    auto eval_cache = alternate_evaluator->evaluator_cache_mgr();
    MS_EXCEPTION_IF_NULL(eval_cache);
    const auto &alt_eval_args = EvaluatorArgs(alternate_evaluator, args_abs_list);
    auto is_not_undetermined_eval = (undetermined_evals.find(alt_eval_args) == undetermined_evals.cend());
    auto is_not_continued_eval = (continued_evals_.find(u_eval) == continued_evals_.cend());
    auto args_not_evaluated = (eval_cache->GetValue(args_abs_list) == nullptr);
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

FuncGraphPtr GetFuncGraphFromBranchNode(const AnfNodePtr &branch_node) {
  MS_EXCEPTION_IF_NULL(branch_node);
  auto fg = GetValueNode<FuncGraphPtr>(branch_node);
  if (fg != nullptr) {
    return fg;
  }
  if (IsPrimitiveCNode(branch_node, prim::kPrimPartial)) {
    fg = GetValueNode<FuncGraphPtr>(branch_node->cast<CNodePtr>()->input(kPartialGraphIndex));
  }
  if (fg != nullptr) {
    return fg;
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Unexpected branch node: " << branch_node->DebugString();
}

std::string JoinBranchesFailedInfo(const AbstractBasePtr &abs, const AbstractBasePtr &last_out_abs,
                                   const AnfNodePtr &node, const std::string &error_info) {
  constexpr int recursive_level = 2;
  std::ostringstream buffer;
  buffer << "Cannot join the return values of different branches, perhaps you need to make them equal.\n"
         << error_info
         << "#dmsg#Framework Error Message:#dmsg#The abstract type of the return value of the current branch is:\n"
         << abs->ToString() << ",\n and that of the previous branch is:\n"
         << last_out_abs->ToString() << ".\n"
         << "The node is " << node->DebugString(recursive_level);
  if (node->isa<CNode>()) {
    auto cnode = node->cast_ptr<CNode>()->input(0);
    if (IsPrimitiveCNode(cnode, prim::kPrimSwitch)) {
      // {prim::kPrimSwitch, cond, true_branch, false_branch}
      const auto &inputs = cnode->cast_ptr<CNode>()->inputs();
      auto true_out = GetFuncGraphFromBranchNode(inputs[kSwitchTrueBranchIndex])->get_return();
      auto false_out = GetFuncGraphFromBranchNode(inputs[kSwitchFalseBranchIndex])->get_return();
      buffer << ", true branch: " << inputs.at(kSwitchTrueBranchIndex)->ToString() << "\n"
             << trace::GetDebugInfoStr(true_out->debug_info())
             << "\n, false branch: " << inputs.at(kSwitchFalseBranchIndex)->ToString() << "\n"
             << trace::GetDebugInfoStr(false_out->debug_info());
    } else if (IsPrimitiveCNode(cnode, prim::kPrimSwitchLayer)) {
      // {prim::kPrimSwitchLayer, X, {prim::kPrimMakeTuple, branch1, branch2, ...}}
      constexpr int branch_index = 2;
      const auto &tuple_node = cnode->cast_ptr<CNode>()->input(branch_index);
      if (IsPrimitiveCNode(tuple_node, prim::kPrimMakeTuple)) {
        const auto &tuple_inputs = tuple_node->cast_ptr<CNode>()->inputs();
        for (size_t i = 1; i < tuple_inputs.size(); i++) {
          auto out_node = GetValueNode<FuncGraphPtr>(tuple_inputs.at(i))->get_return();
          MS_EXCEPTION_IF_NULL(out_node);
          buffer << ", branch" << i << ": " << tuple_inputs.at(i)->ToString() << "\n"
                 << trace::GetDebugInfoStr(out_node->debug_info());
        }
      }
    } else {
      buffer << trace::GetDebugInfoStr(node->debug_info());
    }
  }
  buffer << "\n";
  return buffer.str();
}

void SetUseFlagsForJoinedAny(const AbstractBasePtrList &out_abs_list) {
  for (const auto &abs : out_abs_list) {
    SetSequenceElementsUseFlagsRecursively(abs, true);
  }
}

EvalResultPtr AnalysisEngine::ProcessEvalResults(const AbstractBasePtrList &out_abs_list, const AnfNodePtr &node) {
  if (out_abs_list.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "There is an endless loop for evaluator.";
  }

  if (out_abs_list.size() == 1) {
    MS_EXCEPTION_IF_NULL(out_abs_list[0]);
    // If only one result derived, then broaden it to avoid wrong constant propagation.
    return std::make_shared<EvalResult>(out_abs_list[0]->Broaden(), std::make_shared<AttrValueMap>());
  }
  MS_EXCEPTION_IF_NULL(node);

  // Return Any if some branch returns Any.
  if (std::any_of(out_abs_list.cbegin(), out_abs_list.cend(), [](const AbstractBasePtr &abs) {
        MS_EXCEPTION_IF_NULL(abs);
        return abs->isa<AbstractAny>() && !abs->isa<AbstractNegligible>();
      })) {
    MS_LOG(INFO) << "The branches outputs contain Any output.\nJoin them to Any output.";
    return std::make_shared<EvalResult>(std::make_shared<AbstractAny>(), std::make_shared<AttrValueMap>());
  }

  AbstractBasePtr last_out_abs = out_abs_list[0];
  MS_EXCEPTION_IF_NULL(last_out_abs);
  AbstractBasePtr joined_abs = out_abs_list[0];
  for (size_t i = 1; i < out_abs_list.size(); ++i) {
    const auto &abs = out_abs_list[i];
    MS_EXCEPTION_IF_NULL(abs);
    try {
      MS_LOG(DEBUG) << "Join node: " << node->DebugString() << ", " << joined_abs->ToString() << ", and "
                    << abs->ToString();
      MS_LOG_TRY_CATCH_SCOPE;
      joined_abs = joined_abs->Join(abs);
    } catch (const py::type_error &ex) {
      auto error_info = ExtractLoggingInfo(ex.what());
      const auto info = JoinBranchesFailedInfo(abs, last_out_abs, node, error_info);
      MS_LOG(INFO) << info;
      auto joined_any = std::make_shared<AbstractJoinedAny>();
      joined_any->set_exception(AbstractJoinedAny::ExceptionType::kTypeError);
      joined_any->set_message(info);
      SetUseFlagsForJoinedAny(out_abs_list);
      return std::make_shared<EvalResult>(joined_any, std::make_shared<AttrValueMap>());
    } catch (const py::value_error &ex) {
      auto error_info = ExtractLoggingInfo(ex.what());
      const auto info = JoinBranchesFailedInfo(abs, last_out_abs, node, error_info);
      MS_LOG(INFO) << info;
      auto joined_any = std::make_shared<AbstractJoinedAny>();
      joined_any->set_exception(AbstractJoinedAny::ExceptionType::kValueError);
      joined_any->set_message(info);
      SetUseFlagsForJoinedAny(out_abs_list);
      return std::make_shared<EvalResult>(joined_any, std::make_shared<AttrValueMap>());
    } catch (const std::exception &ex) {
      auto error_info = ExtractLoggingInfo(ex.what());
      const auto info = JoinBranchesFailedInfo(abs, last_out_abs, node, error_info);
      MS_LOG(INFO) << info;
      auto joined_any = std::make_shared<AbstractJoinedAny>();
      joined_any->set_exception(AbstractJoinedAny::ExceptionType::kDefault);
      joined_any->set_message(info);
      // Remove it when the transform form dict to tuple is disabled in Compatible or Lax mode.
      if (joined_abs->isa<AbstractDictionary>()) {
        joined_any->set_user_data<bool>("from_dict", std::make_shared<bool>(true));
      }
      SetUseFlagsForJoinedAny(out_abs_list);
      return std::make_shared<EvalResult>(joined_any, std::make_shared<AttrValueMap>());
    }
    MS_EXCEPTION_IF_NULL(joined_abs);
    last_out_abs = abs;
  }

  MS_LOG(DEBUG) << "Multiple evaluators joined: " << joined_abs->ToString();
  return std::make_shared<EvalResult>(joined_abs, std::make_shared<AttrValueMap>());
}

EvalResultPtr AnalysisEngine::ExecuteMultipleEvaluatorsMultiThread(const std::vector<EvaluatorPtr> &evaluators,
                                                                   const AnfNodeConfigPtr &out_conf,
                                                                   const ConfigPtrList &args_conf_list) {
  MS_EXCEPTION_IF_NULL(out_conf);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  MS_EXCEPTION_IF_NULL(out_conf->func_graph());
  // Release GIL for C++
  MS_LOG(DEBUG) << out_conf->func_graph()->ToString() << "_" << std::this_thread::get_id() << " begin.";
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
  MS_EXCEPTION_IF_NULL(possible_parent_fg);
  // Eval result of the main.
  AsyncAbstractPtr async_result_main = std::make_shared<AsyncAbstract>();
  if (possible_parent_fg->has_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE)) {
    async_result_main->set_ignore_value(true);
  }
  // Eval result of the branches
  std::vector<AsyncAbstractPtr> async_result_branches;
  SetUndeterminedFlag(AnalysisSchedule::thread_id(), *possible_parent_fg);
  for (auto &evaluator : evaluators) {
    static std::atomic<int> id_count{0};
    std::string thread_id = AnalysisSchedule::thread_id() + "." + std::to_string(id_count.fetch_add(1));
    MS_EXCEPTION_IF_NULL(evaluator);
    AsyncAbstractPtr async_result_branch = std::make_shared<AsyncAbstract>(async_result_main);
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
    MS_LOG(DEBUG) << "Add to schedule: " << async_task.get();
    AnalysisSchedule::GetInstance().Add2Schedule(async_task);  // Activate order witch child thread.
    (void)async_result_branches.emplace_back(std::move(async_result_branch));
  }

  size_t len = evaluators.size();
  size_t min_size = 2;
  if (len < min_size) {
    MS_LOG(EXCEPTION) << "There are at least 2 evaluators in multi thread, but got " << len << " evaluator.";
  }

  MS_LOG(DEBUG) << GetInferThread() << "async : wait for one of async to finish.  " << evaluators[0]->ToString()
                << " or  " << evaluators[1]->ToString() << "...";

  auto first_result = async_result_main->GetResult();
  MS_EXCEPTION_IF_NULL(first_result);
  MS_LOG(DEBUG) << GetInferThread() << "async main thread result of " << out_conf->node()->ToString() << " = "
                << first_result->ToString();

  AbstractBasePtrList out_abs_list;
  if (NeedWaitForBranches(first_result)) {
    MS_LOG(DEBUG) << GetInferThread() << " BuildPossibleSpecs.";
    BuildPossibleSpecs(first_result, async_result_branches, &out_abs_list);
  } else {
    for (size_t i = 0; i < len; ++i) {
      AbstractBasePtr result;
      MS_EXCEPTION_IF_NULL(async_result_branches[i]);
      if (enable_waiting_branch_eval()) {
        // wait to get the result of branch.
        result = async_result_branches[i]->GetResult();
      } else {
        // Not wait to get the result of branch.
        result = async_result_branches[i]->TryGetResult();
      }

      if (result) {
        MS_EXCEPTION_IF_NULL(evaluators[i]);
        MS_EXCEPTION_IF_NULL(result);
        MS_LOG(DEBUG) << "#" << i << ": " << GetInferThread() << " async get " << evaluators[i]->ToString()
                      << ", result: " << result->ToString() << ", args: " << args_conf_list;
        out_abs_list.push_back(result);
      }
    }
  }
  MS_LOG(DEBUG) << GetInferThread() << " finish.";
  const auto &processed_result = ProcessEvalResults(out_abs_list, out_conf->node());
  if (processed_result != nullptr) {
    // This is the final switch()() value.
    AnalysisResultCacheMgr::GetInstance().SetSwitchValue(out_conf, processed_result->abstract());
  }
  MS_LOG(DEBUG) << GetInferThread() << " join finish.";
  return processed_result;
}

EvalResultPtr AnalysisEngine::ExecuteMultipleEvaluators(const std::vector<EvaluatorPtr> &evaluators,
                                                        const AnfNodeConfigPtr &out_conf,
                                                        const ConfigPtrList &args_conf_list) {
  AbstractBasePtrList out_abs_list;
  const size_t evaluators_size = 2;
  if (evaluators.size() < evaluators_size) {
    MS_LOG(INTERNAL_EXCEPTION) << "Evaluators size is less than 2.";
  }
  multi_poss_[evaluators[0]] = evaluators[1];
  multi_poss_[evaluators[1]] = evaluators[0];
  AbstractBasePtrList args_abs_list = EvaluateArguments(args_conf_list);
  MS_EXCEPTION_IF_NULL(out_conf);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  auto possible_parent_fg = out_conf->node()->func_graph();
  MS_EXCEPTION_IF_NULL(possible_parent_fg);
  possible_parent_fg->set_flag(kFuncGraphFlagUndetermined, true);
  MS_LOG(DEBUG) << "Set graph undetermined flag for " << possible_parent_fg->ToString();
  for (const auto &eval : evaluators) {
    MS_EXCEPTION_IF_NULL(eval);
    const auto current_inf = EvaluatorArgs(eval, args_abs_list);
    MS_LOG(DEBUG) << "Check evaluator " << eval->ToString();
    // If current evaluator is under tracing, then skip current evaluator to avoid recursively evaluating.
    auto it = std::find(eval_trace_.crbegin(), eval_trace_.crend(), current_inf);
    if (it == eval_trace_.crend()) {
      eval_trace_.push_back(current_inf);
      auto eval_result = eval->Run(shared_from_this(), args_conf_list, out_conf);
      MS_EXCEPTION_IF_NULL(eval_result);
      auto eval_abstract = eval_result->abstract();
      MS_EXCEPTION_IF_NULL(eval_abstract);

      out_abs_list.push_back(eval_abstract);
      eval_trace_.pop_back();
      if (eval_trace_.empty()) {
        multi_poss_.clear();
      }
    } else {
      bool continue_flag = false;
      auto latest_entry = HandleNestedRecursion(evaluators, eval, args_abs_list, it, &continue_flag);
      if (continue_flag) {
        MS_EXCEPTION_IF_NULL(current_inf.evaluator_);
        MS_LOG(DEBUG) << "The continued_evals_ insert " << current_inf.evaluator_.get() << "/"
                      << current_inf.evaluator_->ToString();
        continued_evals_.insert(current_inf);
        continue;
      }

      // Try to travel the latest undetermined.
      if (latest_entry != eval_trace_.rbegin()->evaluator_) {
        MS_LOG(DEBUG) << "Direct run evaluator " << eval.get() << "/" << eval->ToString();
        auto eval_result = latest_entry->Run(shared_from_this(), args_conf_list, out_conf);
        MS_EXCEPTION_IF_NULL(eval_result);
        MS_EXCEPTION_IF_NULL(eval_result->abstract());
        MS_LOG(DEBUG) << "End direct evaluator " << latest_entry->ToString()
                      << ", return out_abs: " << eval_result->abstract()->ToString();
        possible_parent_fg->set_flag(kFuncGraphFlagUndetermined, false);
        return eval_result;
      }
    }
  }
  possible_parent_fg->set_flag(kFuncGraphFlagUndetermined, false);
  return ProcessEvalResults(out_abs_list, out_conf->node());
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
  if (value->isa<ValueSequence>() && anf_node != nullptr) {
    auto abs = value->ToAbstract();
    MS_EXCEPTION_IF_NULL(abs);
    // Attach corresponding python sequence object to AbstractSequence.
    py::object py_list_obj =
      fallback::HasPyObjectInNode(anf_node) ? fallback::GetPyObjectFromNode(anf_node) : ValueToPyData(value);
    fallback::AttachPyObjToAbs(abs, py_list_obj, !fallback::HasPyObjectInNode(anf_node));
    MS_LOG(DEBUG) << "Attach python list object " << fallback::GetPyObjectPtrStr(py_list_obj)
                  << " to new abstract: " << abs->ToString();
    // Set sequence node for new AbstractSequence.
    static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
    if (enable_eliminate_unused_element) {
      auto sequence_abs = abs->cast<AbstractSequencePtr>();
      MS_EXCEPTION_IF_NULL(sequence_abs);
      SetSequenceNodeElementsUseFlags(anf_node, std::make_shared<std::vector<bool>>(sequence_abs->elements().size()));
      std::shared_ptr<AnfNodeWeakPtrList> sequence_nodes = std::make_shared<AnfNodeWeakPtrList>();
      (void)sequence_nodes->emplace_back(AnfNodeWeakPtr(anf_node));
      sequence_abs->set_sequence_nodes(sequence_nodes);
    }
    return abs;
  }
  if (value->isa<ValueDictionary>() && anf_node != nullptr) {
    auto abs = value->ToAbstract();
    MS_EXCEPTION_IF_NULL(abs);
    // Attach corresponding python dictionary object to AbstractDictionary.
    py::object py_dict_obj =
      fallback::HasPyObjectInNode(anf_node) ? fallback::GetPyObjectFromNode(anf_node) : fallback::GeneratePyObj(abs);
    fallback::AttachPyObjToAbs(abs, py_dict_obj, !fallback::HasPyObjectInNode(anf_node));
    MS_LOG(DEBUG) << "Attach python dict object " << fallback::GetPyObjectPtrStr(py_dict_obj)
                  << " to new abstract: " << abs->ToString();
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
