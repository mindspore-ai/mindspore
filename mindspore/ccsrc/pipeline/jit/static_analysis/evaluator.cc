/**
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

#include "pipeline/jit/static_analysis/evaluator.h"

#include <algorithm>
#include <utility>
#include <unordered_set>

#include "ir/func_graph_cloner.h"
#include "abstract/utils.h"
#include "debug/trace.h"
#include "utils/ms_context.h"
#include "pipeline/jit/static_analysis/stack_frame.h"
#include "pipeline/jit/static_analysis/async_eval_result.h"

namespace mindspore {
namespace abstract {
namespace {
string EvalEntryLogging(const EvaluatorPtr &evaluator, const AbstractBasePtrList &arg_spec_list,
                        const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(evaluator);
  std::stringstream ss;
  if (out_conf != nullptr) {
    ss << "Evaluator " << evaluator->ToString() << " run for " << out_conf->node()->scope()->name();
  }
  for (size_t i = 0; i < arg_spec_list.size(); i++) {
    ss << evaluator->ToString() << " input[" << i
       << "] abstract value: " << (arg_spec_list[i] ? arg_spec_list[i]->ToString() : "null abstract.");
  }
  return ss.str();
}

void EvalFailLogging(const EvaluatorPtr &evaluator, const AbstractBasePtrList &, const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(evaluator);
  if (out_conf != nullptr) {
    auto node = out_conf->node();
    if (IsValueNode<Primitive>(node)) {
      MS_LOG(ERROR) << "Evaluator " << evaluator->ToString() << " run failed for node " << node->fullname_with_scope()
                    << ", with debug info: " << trace::GetDebugInfo(node->debug_info());
    } else {
      MS_LOG(ERROR) << "Evaluator " << evaluator->ToString() << " run failed for node " << node->DebugString()
                    << ", with debug info: " << trace::GetDebugInfo(node->debug_info());
    }
  }
}
}  // namespace

void BaseFuncGraphEvaluator::EnterStackFrame(const AnalysisEnginePtr &engine, const StackFramePtr &current_stack_frame,
                                             const StackFramePtr &new_stack_frame) {
  MS_EXCEPTION_IF_NULL(current_stack_frame);
  MS_EXCEPTION_IF_NULL(new_stack_frame);
  MS_EXCEPTION_IF_NULL(engine);
  // Enter new func graph.
  auto &current_node = current_stack_frame->CurrentNode();
  auto current_context = current_stack_frame->current_context();
  AnfNodeConfigPtr call_conf = engine->MakeConfig(current_node, current_context, current_context->func_graph());
  auto evaluator = new_stack_frame->evaluator();
  MS_EXCEPTION_IF_NULL(evaluator);
  auto new_context = new_stack_frame->current_context();
  trace::TraceGraphEvalEnter(new_context, call_conf);

  // Increase & Check the func graph call depth.
  IncreaseFunctionCallDepth();
  IncreaseStackFrameDepth();
  const uint32_t max_depth = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_MAX_CALL_DEPTH);
  if (FunctionCallDepth() > max_depth) {
    MS_LOG(EXCEPTION) << "Exceed function call depth limit " << max_depth
                      << ", (function call depth: " << FunctionCallDepth()
                      << ", simulate call depth: " << StackFrameDepth() << ").\n"
                      << "It's always happened with complex construction of code or infinite recursion or loop.\n"
                      << "Please check the code if it's has the infinite recursion "
                      << "or call 'context.set_context(max_call_depth=value)' to adjust this value.\n"
                      << "If max_call_depth is set larger, the system max stack depth should be set larger too "
                      << "to avoid stack overflow.\n"
                      << "For more details, please refer to the FAQ at https://www.mindspore.cn.";
  }
  MS_LOG(DEBUG) << evaluator << "(" << evaluator->type_name() << "/" << evaluator->ToString()
                << "), enter, function call depth: " << FunctionCallDepth() << " - " << StackFrameDepth();
}

void BaseFuncGraphEvaluator::LeaveStackFrame(const AnalysisEnginePtr &, const StackFramePtr &current_stack_frame) {
  MS_EXCEPTION_IF_NULL(current_stack_frame);
  // Leave current func graph.
  auto current_context = current_stack_frame->current_context();
  trace::TraceGraphEvalLeave(current_context);

  // Decrease the func graph call depth.
  DecreaseFunctionCallDepth();
  DecreaseStackFrameDepth();

  auto evaluator = current_stack_frame->evaluator();
  MS_EXCEPTION_IF_NULL(evaluator);
  MS_LOG(DEBUG) << evaluator << "(" << evaluator->type_name() << "/" << evaluator->ToString()
                << "), leave, function call depth: " << FunctionCallDepth() << " - " << StackFrameDepth();
}

// Start running stack frames in a Evaluator.
AbstractBasePtr BaseFuncGraphEvaluator::LaunchStackFrame(const AnalysisEnginePtr &engine, const FuncGraphPtr &fg,
                                                         const AnalysisContextPtr &context) {
  EvalResultPtr eval_result = nullptr;
  AbstractBasePtr res_base = nullptr;
  std::stack<StackFramePtr> stack_frames;
  auto current_stack_frame = std::make_shared<StackFrame>(shared_from_base<Evaluator>(), fg, context, parent_context_);
  MS_LOG(DEBUG) << "[" << this << "/StackFrame] Start at func graph, " << current_stack_frame;
  stack_frames.push(current_stack_frame);
  while (true) {
    current_stack_frame = stack_frames.top();
    if (current_stack_frame->Done()) {
      MS_EXCEPTION_IF_NULL(res_base);
      MS_LOG(DEBUG) << "[" << this << "/StackFrame] Leave from func graph, " << current_stack_frame;
      stack_frames.pop();
      if (stack_frames.empty()) {
        MS_LOG(DEBUG) << "[" << this << "/StackFrame] Finish at func graph, " << current_stack_frame
                      << ", res_base: " << res_base->ToString();
        break;
      }
      // Leave current func graph.
      LeaveStackFrame(engine, current_stack_frame);
      // Switch the stack frame.
      auto last_stack_frame = current_stack_frame;
      current_stack_frame = stack_frames.top();
      MS_LOG(DEBUG) << "[" << this << "/StackFrame] Back to func graph, " << current_stack_frame;
      current_stack_frame->Back(engine, last_stack_frame, eval_result);
      continue;
    }

    auto new_stack_frame = current_stack_frame->Jump(engine);
    if (new_stack_frame != nullptr) {
      // Enter new func graph.
      EnterStackFrame(engine, current_stack_frame, new_stack_frame);
      // Update current stack frame.
      stack_frames.push(new_stack_frame);
      current_stack_frame = new_stack_frame;
      MS_LOG(DEBUG) << "[" << this << "/StackFrame] Jump to new func graph, " << new_stack_frame;
      continue;
    }

    eval_result = current_stack_frame->Step(engine);
    MS_EXCEPTION_IF_NULL(eval_result);
    res_base = eval_result->abstract();
  }
  return res_base;
}

AbstractBasePtr BaseFuncGraphEvaluator::LaunchRecursiveEval(const AnalysisEnginePtr &engine, const FuncGraphPtr &fg,
                                                            const AnalysisContextPtr &context) {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(engine);
  const AnfNodePtr &func_node = fg->get_return();
  const auto &all_nodes = TopoSort(func_node, SuccIncoming, [](const AnfNodePtr &node) -> IncludeType {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<ValueNode>() || node->isa<Parameter>()) {
      return EXCLUDE;
    }
    return FOLLOW;
  });
  AbstractBasePtr res_base = nullptr;
  for (const auto &node : all_nodes) {
    AnfNodeConfigPtr node_conf = engine->MakeConfig(node, context, fg);
    MS_LOG(DEBUG) << "Analysis node begin, func graph: " << fg << "/" << fg->ToString()
                  << ", node_conf: " << node_conf->ToString();
    auto node_eval_result = engine->ObtainEvalResultWithCache(node_conf);
    MS_EXCEPTION_IF_NULL(node_eval_result);
    res_base = node_eval_result->abstract();
    MS_EXCEPTION_IF_NULL(res_base);
    MS_LOG(DEBUG) << GetInferThread() << "Eval ( " << node_conf->ToString() << ") = " << res_base->ToString();
  }
  MS_EXCEPTION_IF_NULL(res_base);
  return res_base;
}

EvalResultPtr BaseFuncGraphEvaluator::Eval(AnalysisEnginePtr engine, const AbstractBasePtrList &args_abs_list,
                                           const AnfNodeConfigPtr &out_conf) {
  auto eval_result = evaluator_cache_mgr_->GetValue(args_abs_list);
  if (eval_result != nullptr) {
    MS_LOG(ERROR) << ToString() << ArgsToString(args_abs_list) << " entered again. There is something wrong.";
    return eval_result;
  } else {
    MS_LOG(DEBUG) << ToString() << " entered first.";
  }
  MS_EXCEPTION_IF_NULL(engine);

  // Increase & Check the func graph call depth.
  IncreaseFunctionCallDepth();
  const uint32_t max_depth = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_MAX_CALL_DEPTH);
  if (FunctionCallDepth() > max_depth) {
    MS_LOG(EXCEPTION) << "Exceed function call depth limit " << max_depth
                      << ", (function call depth: " << FunctionCallDepth()
                      << ", simulate call depth: " << StackFrameDepth() << ").\n"
                      << "It's always happened with complex construction of code or infinite recursion or loop.\n"
                      << "Please check the code if it's has the infinite recursion "
                      << "or call 'context.set_context(max_call_depth=value)' to adjust this value.\n"
                      << "If max_call_depth is set larger, the system max stack depth should be set larger too "
                      << "to avoid stack overflow.\n"
                      << "For more details, please refer to the FAQ at https://www.mindspore.cn.";
  }
  MS_LOG(DEBUG) << this << "(" << type_name() << "/" << ToString()
                << "), enter, function call depth: " << FunctionCallDepth() << " - " << StackFrameDepth();

  FuncGraphPtr fg = GetFuncGraph(engine, args_abs_list);
  MS_EXCEPTION_IF_NULL(fg);
  auto context = parent_context_->NewContext(fg, args_abs_list);
  trace::TraceGraphEvalEnter(context, out_conf);

  std::size_t nargs = fg->parameters().size();
  if (args_abs_list.size() != nargs) {
    MS_EXCEPTION(TypeError) << "The parameters number of the function is " << fg->parameters().size()
                            << ", but the number of provided arguments is " << args_abs_list.size() << ".\n"
                            << "FunctionGraph : " << fg->ToString()
                            << "\nNodeInfo: " << trace::GetDebugInfo(fg->debug_info());
  }
  MS_EXCEPTION_IF_NULL(parent_context_);
  MS_LOG(DEBUG) << GetInferThread() << "@" << fg->ToString() << ArgsToString(args_abs_list) << " { ";
  if (parent_context_->func_graph() != nullptr) {
    MS_LOG(DEBUG) << GetInferThread() << "graph_: " << AnalysisSchedule::GetThreadID() << ":"
                  << parent_context_->func_graph()->ToString() << "()->" << AnalysisSchedule::GetThreadID() << ":"
                  << fg->ToString() << "();";
  }

  auto func_graph_evaluator = dyn_cast<FuncGraphEvaluator>(shared_from_base<BaseFuncGraphEvaluator>());
  if (func_graph_evaluator != nullptr) {
    if (engine->root_func_graph() == func_graph_evaluator->func_graph()) {
      engine->set_root_context(context);
    }
  }
  const auto &parameters = fg->parameters();
  for (size_t i = 0; i < nargs; i++) {
    const auto &arg = args_abs_list[i];
    const auto &node = parameters[i];
    AnfNodeConfigPtr conf = engine->MakeConfig(node, context, fg);
    engine->SaveEvalResultInCache(conf, std::make_shared<EvalResult>(arg, nullptr));
    MS_LOG(DEBUG) << GetInferThread() << "Set Param: " << conf->ToString() << "   =   " << arg->ToString();
  }
  MS_LOG(DEBUG) << "Analysis FuncGraph begin, func graph: " << fg << "/" << fg->ToString()
                << ", context: " << context->ToString() << ", return node: " << fg->get_return()->DebugString()
                << ", parent: " << (parent_context_->func_graph() ? parent_context_->func_graph()->ToString() : "NULL")
                << ", current function call depth: " << FunctionCallDepth();
  AbstractBasePtr res_base = nullptr;
  if (engine->enable_recursive_eval()) {
    res_base = LaunchRecursiveEval(engine, fg, context);
  } else {
    res_base = LaunchStackFrame(engine, fg, context);
  }

  MS_EXCEPTION_IF_NULL(res_base);
  MS_LOG(DEBUG) << "Analysis FuncGraph end, " << fg << "/" << fg->ToString()
                << ", evaluated abstract: " << res_base->ToString() << ", is stub: " << fg->stub();
  if (fg->stub()) {
    res_base = std::make_shared<AbstractUndetermined>();
  }
  MS_LOG(DEBUG) << GetInferThread() << "} //" << fg->ToString() << " = " << res_base->ToString();

  trace::TraceGraphEvalLeave(context);
  // Decrease the func graph call depth.
  DecreaseFunctionCallDepth();
  MS_LOG(DEBUG) << this << "(" << type_name() << "/" << ToString()
                << "), leave, function call depth: " << FunctionCallDepth() << " - " << StackFrameDepth();
  auto res = std::make_shared<EvalResult>(res_base, nullptr);
  return res;
}

void BroadenArgs(const AbstractBasePtrList &args_spec_list, AbstractBasePtrList *broaded_args) {
  MS_EXCEPTION_IF_NULL(broaded_args);
  (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(*broaded_args),
                       [](const AbstractBasePtr &arg) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(arg);
                         if (arg->GetValueTrack() != kAnyValue) {
                           return arg->Broaden();
                         }
                         return arg;
                       });
}

AbstractBasePtrList FuncGraphEvaluator::NormalizeArgs(const AbstractBasePtrList &args_spec_list) const {
  MS_EXCEPTION_IF_NULL(func_graph_);
  if (func_graph_->has_flag(FUNC_GRAPH_FLAG_IGNORE_VALUES)) {
    AbstractBasePtrList broaded_list;
    BroadenArgs(args_spec_list, &broaded_list);
    MS_LOG(DEBUG) << func_graph_->ToString() << " original: " << mindspore::ToString(args_spec_list)
                  << ", broaded: " << mindspore::ToString(broaded_list);
    return broaded_list;
  }
  return args_spec_list;
}

AbstractBasePtrList FuncGraphEvaluator::BroadenUndeterminedArgs(const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(func_graph_);
  if (func_graph_->has_flag(FUNC_GRAPH_FLAG_IGNORE_VALUES)) {
    return args_spec_list;
  }

  if (func_graph_->has_flag(kFuncGraphFlagUndetermined)) {
    func_graph_->set_flag(FUNC_GRAPH_FLAG_IGNORE_VALUES, true);
    auto normalized_args_spec_list = NormalizeArgs(args_spec_list);
    MS_LOG(DEBUG) << "Set " << func_graph_->ToString() << " with IGNORE_VALUES flag.";
    MS_LOG(DEBUG) << "Normalized args " << mindspore::ToString(normalized_args_spec_list);
    return normalized_args_spec_list;
  }
  return args_spec_list;
}

FuncGraphPtr FuncGraphEvaluator::GetFuncGraph(AnalysisEnginePtr engine, const AbstractBasePtrList &args_spec_list) {
  auto iter = func_graph_cache_.find(args_spec_list);
  FuncGraphPtr res;
  if (iter == func_graph_cache_.end()) {
    auto fg = func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    FuncGraphPtr generated_graph = fg->GenerateGraph(args_spec_list);
    func_graph_cache_[args_spec_list] = generated_graph;
    MS_EXCEPTION_IF_NULL(engine);
    engine->func_graph_manager()->AddFuncGraph(generated_graph);
    res = generated_graph;
  } else {
    res = iter->second;
  }

  // For the top graph, if it is replaced by generated graph, update the top graph to the new one.
  if (parse::Parser::GetTopFuncGraph() == func_graph()) {
    if (res != func_graph()) {
      parse::Parser::UpdateTopFuncGraph(res);
    }
  }
  return res;
}

FuncGraphPtr MetaFuncGraphEvaluator::GetFuncGraph(AnalysisEnginePtr engine, const AbstractBasePtrList &args_spec_list) {
  auto iter = func_graph_cache_.find(args_spec_list);
  if (iter != func_graph_cache_.end()) {
    return iter->second;
  }

  MS_EXCEPTION_IF_NULL(meta_func_graph_);
  FuncGraphPtr generated_func_graph;
  if (this->bound_node() != nullptr) {
    TraceGuard trace_guard(std::make_shared<TraceGenMetaFuncGraph>(bound_node()->debug_info()));
    generated_func_graph = meta_func_graph_->GenerateFuncGraph(args_spec_list);
  } else {
    generated_func_graph = meta_func_graph_->GenerateFuncGraph(args_spec_list);
  }

  FuncGraphPtr cloned_func_graph = BasicClone(generated_func_graph);
  func_graph_cache_[args_spec_list] = cloned_func_graph;
  MS_EXCEPTION_IF_NULL(engine);
  engine->func_graph_manager()->AddFuncGraph(cloned_func_graph);
  return cloned_func_graph;
}

EvalResultPtr Evaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                             const AnfNodeConfigPtr &out_conf) {
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->ObtainEvalResult()->abstract();
                       });
  args_spec_list = NormalizeArgs(args_spec_list);
  args_spec_list = BroadenUndeterminedArgs(args_spec_list);

  MS_LOG(DEBUG) << EvalEntryLogging(shared_from_base<Evaluator>(), args_spec_list, out_conf);
  const std::string &evaluator_name = ToString();
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr_);
  auto eval_result = evaluator_cache_mgr_->GetValue(args_spec_list);
  if (eval_result == nullptr) {
    MS_LOG(DEBUG) << evaluator_name << " cache miss, call Eval().";
    eval_result = Eval(engine, args_spec_list, out_conf);
    MS_EXCEPTION_IF_NULL(eval_result);
    if (eval_result->abstract() == nullptr) {
      EvalFailLogging(shared_from_base<Evaluator>(), args_spec_list, out_conf);
      MS_LOG(EXCEPTION) << "Evaluator " << evaluator_name << " result is nullptr.";
    }
    MS_LOG(DEBUG) << evaluator_name << " set cache. return: " << eval_result->abstract()->ToString() << ".";
    evaluator_cache_mgr_->SetValue(args_spec_list, eval_result);
  } else {
    MS_EXCEPTION_IF_NULL(eval_result);
    MS_EXCEPTION_IF_NULL(eval_result->abstract());
    MS_LOG(DEBUG) << evaluator_name << " cache hit. return: " << eval_result->abstract()->ToString() << ".";
  }
  return eval_result;
}

EvalResultPtr TrivialPrimEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                        const AnfNodeConfigPtr &) {
  AbstractBasePtrList args_spec_list;
  auto is_py_eval = (identifier_ == "PythonPrimEvaluator");
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [is_py_eval](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         auto abstract = conf->ObtainEvalResult()->abstract();
                         MS_EXCEPTION_IF_NULL(abstract);
                         // broaden the ref_key, while infer python prim for cache
                         if (is_py_eval && abstract->isa<AbstractRef>()) {
                           auto abs_ref = abstract->cast<AbstractRefPtr>();
                           abstract = std::make_shared<AbstractRef>(abs_ref->ref_key()->Broaden(), abs_ref);
                         }
                         return abstract;
                       });
  return EvalPrim(engine, args_spec_list);
}

EvalResultPtr TransitionPrimEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                           const AnfNodeConfigPtr &out_conf) {
  if (args_conf_list.empty()) {
    MS_LOG(EXCEPTION) << "Size should greater than 0";
  }
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->ObtainEvalResult()->abstract();
                       });
  EvalResultPtr res = EvalPrim(engine, args_spec_list, args_conf_list[0], out_conf);
  // No need to cache.
  return res;
}

EvalResultPtr SymbolicPrimEvaluator::Run(AnalysisEnginePtr, const ConfigPtrList &args_conf_list,
                                         const AnfNodeConfigPtr &) {
  return EvalPrim(args_conf_list);
}

EvalResultPtr TrackedEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                    const AnfNodeConfigPtr &out_conf) {
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->ObtainEvalResult()->abstract();
                       });
  EvalResultPtr res = sub_evaluator_->Run(engine, args_conf_list, out_conf);
  // Don't lookup from cache, as different out_conf with same node but different context
  // may add different entry to anfnode_config_map_, like getattr primitive.
  evaluator_cache_mgr_->SetValue(args_spec_list, res);
  return res;
}

EvalResultPtr PartialAppEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                       const AnfNodeConfigPtr &out_conf) {
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->ObtainEvalResult()->abstract();
                       });
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr_);
  auto eval_result = evaluator_cache_mgr_->GetValue(args_spec_list);
  if (eval_result != nullptr) {
    return eval_result;
  }

  ConfigPtrList partial_args_conf_list;
  // Join arguments in partial and the rest arguments from args_conf_list.
  (void)std::transform(args_spec_list_.begin(), args_spec_list_.end(), std::back_inserter(partial_args_conf_list),
                       [](const AbstractBasePtr &arg) -> ConfigPtr { return std::make_shared<VirtualConfig>(arg); });

  (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(partial_args_conf_list),
                       [](const AbstractBasePtr &arg) -> ConfigPtr { return std::make_shared<VirtualConfig>(arg); });
  EvalResultPtr res = evaluator_->Run(engine, partial_args_conf_list, out_conf);
  evaluator_cache_mgr_->SetValue(args_spec_list, res);
  return res;
}

EvalResultPtr JEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list, const AnfNodeConfigPtr &) {
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->ObtainEvalResult()->abstract();
                       });
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr_);
  auto eval_result = evaluator_cache_mgr_->GetValue(args_spec_list);
  if (eval_result != nullptr) {
    return eval_result;
  }

  // Call the original evaluator, get the result: y = f(x)
  EvalResultPtr result = evaluator_->Run(engine, args_conf_list, nullptr);
  MS_EXCEPTION_IF_NULL(result);
  // Build a virtual function: bprop_f which use sense of y as input, return sense of function free variable and input
  // parameters. (sense_f, sense_x, ...)(*bpro_f) (sense_y)
  AbstractBasePtrList bparams;
  bparams.push_back(SensitivityTransform(orig_func_));
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool enable_sparse = context->get_param<bool>(MS_CTX_ENABLE_SPARSE);
  (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(bparams),
                       [&enable_sparse](const AbstractBasePtr &arg_spec) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(arg_spec);
                         if (enable_sparse && arg_spec->isa<AbstractTensor>()) {
                           return std::make_shared<AbstractUndetermined>();
                         }
                         return SensitivityTransform(arg_spec);
                       });
  AbstractBasePtr bparams_final = std::make_shared<AbstractTuple>(bparams);
  AbstractFunctionPtr bprop =
    std::make_shared<VirtualAbstractClosure>(SensitivityTransform(result->abstract()), bparams_final);

  // J(f)(J(x)) return a tuple (y, bprop_f)
  AbstractBasePtrList jargs = {result->abstract(), bprop};
  AbstractBasePtr jtuple = std::make_shared<AbstractTuple>(jargs);
  auto res = std::make_shared<EvalResult>(jtuple, std::make_shared<AttrValueMap>());
  evaluator_cache_mgr_->SetValue(args_spec_list, res);
  return res;
}

EvalResultPtr VirtualEvaluator::Eval(AnalysisEnginePtr, const AbstractBasePtrList &args_spec_list,
                                     const AnfNodeConfigPtr &out_conf) {
  if (args_spec_list.size() != args_spec_list_.size()) {
    MS_LOG(EXCEPTION) << "Arguments mismatch, parameters no: " << args_spec_list_.size()
                      << ", arguments no: " << args_spec_list.size();
  }
  // Check each parameter and argument match;
  for (std::size_t i = 0; i < args_spec_list.size(); i++) {
    MS_EXCEPTION_IF_NULL(args_spec_list[i]);
    (void)args_spec_list[i]->Join(args_spec_list_[i]);
  }
  return std::make_shared<EvalResult>(output_, std::make_shared<AttrValueMap>());
}
EvalResultPtr Evaluator::SingleRun(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                   const AnfNodeConfigPtr &out_conf) {
  EvalResultPtr result;
  try {
    result = this->Run(engine, args_conf_list, out_conf);
  } catch (const std::exception &ex) {
    MS_LOG(INFO) << "Eval " << ToString() << " throw exception.";
    AnalysisSchedule::GetInstance().HandleException(ex);
  }
  AnalysisSchedule::GetInstance().Wait();
  return result;
}
}  // namespace abstract
}  // namespace mindspore
