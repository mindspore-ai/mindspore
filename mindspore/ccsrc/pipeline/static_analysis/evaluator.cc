/**
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

#include "pipeline/static_analysis/evaluator.h"

#include <algorithm>
#include <unordered_set>

#include "ir/func_graph_cloner.h"
#include "pipeline/static_analysis/utils.h"
#include "debug/trace.h"

namespace mindspore {
namespace abstract {
namespace {
void InferEntryLogging(const EvaluatorPtr &evaluator, const AbstractBasePtrList &arg_spec_list,
                       const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(evaluator);
  if (out_conf != nullptr) {
    MS_LOG(DEBUG) << "Evaluator " << evaluator->ToString() << " run for " << out_conf->node()->scope()->name();
  }
  for (size_t i = 0; i < arg_spec_list.size(); i++) {
    MS_LOG(DEBUG) << evaluator->ToString() << " input[" << i << "] abstract value: " << arg_spec_list[i]->ToString();
  }
}

void InferFailLogging(const EvaluatorPtr &evaluator, const AbstractBasePtrList &, const AnfNodeConfigPtr &out_conf) {
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

AnalysisContextPtr BaseFuncGraphEvaluator::MakeContext(const AnalysisEnginePtr &engine,
                                                       const AbstractBasePtrList &args_spec_list) {
  AbstractBasePtrList normalized_args_spec_list = NormalizeArgs(args_spec_list);
  FuncGraphPtr fg = GetFuncGraph(engine, normalized_args_spec_list);
  MS_EXCEPTION_IF_NULL(parent_context_);
  AnalysisContextPtr context = parent_context_->NewFuncGraphContext(fg, normalized_args_spec_list);
  return context;
}

static std::vector<AnfNodePtr> FastShadowSort(const AnfNodePtr &ret_node) {
  auto ori_func_graph = ret_node->func_graph();
  MS_EXCEPTION_IF_NULL(ori_func_graph);

  std::vector<AnfNodePtr> sorted_nodes;
  std::unordered_set<AnfNodePtr> checked_cnodes;
  std::size_t index = 0;
  sorted_nodes.emplace_back(ret_node);
  while (index < sorted_nodes.size()) {
    auto current = sorted_nodes[index];
    index++;
    MS_EXCEPTION_IF_NULL(current);
    if (current->isa<CNode>()) {
      auto &inputs = current->cast<CNodePtr>()->inputs();
      for (auto it = inputs.begin(); it != inputs.end(); it++) {
        AnfNodePtr input = *it;
        if (input != nullptr && input->isa<CNode>() && checked_cnodes.find(input) == checked_cnodes.end() &&
            input->func_graph() == ori_func_graph) {
          sorted_nodes.emplace_back(input);
          (void)checked_cnodes.insert(input);
        }
      }
    }
  }
  return sorted_nodes;
}

AbstractBasePtr BaseFuncGraphEvaluator::Infer(AnalysisEnginePtr engine, const AbstractBasePtrList &args_spec_list) {
  FuncGraphPtr fg = GetFuncGraph(engine, args_spec_list);
  MS_EXCEPTION_IF_NULL(fg);
  std::size_t nargs = fg->parameters().size();
  if (args_spec_list.size() != nargs) {
    MS_EXCEPTION(TypeError) << "Function " << fg->ToString() << ", The number of parameters of this function is "
                            << fg->parameters().size() << ", but the number of provided arguments is "
                            << args_spec_list.size() << ". NodeInfo: " << trace::GetDebugInfo(fg->debug_info());
  }
  MS_EXCEPTION_IF_NULL(parent_context_);
  MS_EXCEPTION_IF_NULL(engine);
  graph_context_ = parent_context_->NewFuncGraphContext(fg, args_spec_list);
  const auto &parameters = fg->parameters();
  for (size_t i = 0; i < nargs; i++) {
    const auto &arg = args_spec_list[i];
    const auto &node = parameters[i];
    AnfNodeConfigPtr conf = engine->MakeConfig(node, graph_context_);
    engine->cache().set_value(conf, arg);
  }
  const AnfNodePtr &func_node = fg->get_return();

  MS_LOG(DEBUG) << "Analysis FuncGraph begin, func graph: " << fg->ToString()
                << ", context: " << graph_context_->ToString() << ", return node: " << func_node->DebugString();
  AbstractBasePtr ret_base = nullptr;
  std::vector<AnfNodePtr> nodes = FastShadowSort(func_node);
  for (auto it = nodes.crbegin(); it != nodes.crend(); it++) {
    const auto &node = *it;
    AnfNodeConfigPtr node_conf = engine->MakeConfig(node, graph_context_);
    MS_LOG(DEBUG) << "Analysis node begin, func graph: " << fg->ToString() << ", node_conf: " << node_conf->ToString();
    ret_base = engine->GetEvaluatedValue(node_conf);
    MS_LOG(DEBUG) << "Analysis node end, func graph: " << fg->ToString() << ", node_conf: " << node_conf->ToString()
                  << ", abstract: " << ret_base->ToString();
  }

  MS_EXCEPTION_IF_NULL(ret_base);
  MS_LOG(DEBUG) << "BaseFuncGraph " << fg->ToString() << " infer end, inferred abstract: " << ret_base->ToString();
  return ret_base;
}

AbstractBasePtrList FuncGraphEvaluator::NormalizeArgs(const AbstractBasePtrList &args_spec_list) const {
  MS_EXCEPTION_IF_NULL(func_graph_);
  if (func_graph_->has_flag(FUNC_GRAPH_FLAG_IGNORE_VALUES)) {
    AbstractBasePtrList broaded_list;
    (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(broaded_list),
                         [](const AbstractBasePtr &arg) -> AbstractBasePtr {
                           MS_EXCEPTION_IF_NULL(arg);
                           return arg->Broaden();
                         });
    MS_LOG(DEBUG) << func_graph_->ToString() << " original: " << mindspore::ToString(args_spec_list)
                  << ", broaded: " << mindspore::ToString(broaded_list);
    return broaded_list;
  }

  if (func_graph_->has_flag(kFuncGraphFlagUndetermined)) {
    if (parent_context_) {
      MS_LOG(DEBUG) << "Undeterminate FuncGraphEvaluator " << ToString()
                    << ", context: " << parent_context_->ToString();
      auto last_context = parent_context_->Filter(func_graph_);
      if (last_context && last_context->func_graph() == func_graph_) {
        MS_LOG(DEBUG) << "Find last infer context: " << last_context->ToString();
        MS_LOG(DEBUG) << "Current eval args: " << ::mindspore::ToString(args_spec_list);
        MS_LOG(DEBUG) << "Last eval args: " << ::mindspore::ToString(last_context->args_spec_list());
        // Join the last eval arguments and current arguments to check if there are loop variant.
        auto joined_args_spec_list = AbstractJoin(args_spec_list, last_context->args_spec_list());
        MS_LOG(DEBUG) << "Joined args: " << ::mindspore::ToString(joined_args_spec_list);
        // If there is loop variant, all arguments need to be broaden to avoid wrong constant propagation.
        if (!(joined_args_spec_list == args_spec_list)) {
          func_graph_->set_flags(FUNC_GRAPH_FLAG_IGNORE_VALUES, true);
        }
        return joined_args_spec_list;
      }
    }
  }
  return args_spec_list;
}

FuncGraphPtr FuncGraphEvaluator::GetFuncGraph(AnalysisEnginePtr engine, const AbstractBasePtrList &args_spec_list) {
  auto iter = func_graph_cache_.find(args_spec_list);
  FuncGraphPtr ret = nullptr;
  if (iter == func_graph_cache_.end()) {
    auto fg = func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    TraceManager::DebugTrace(std::make_shared<TraceEvaluatorGenGraph>(fg->debug_info()));
    FuncGraphPtr generated_graph = fg->GenerateGraph(args_spec_list);
    TraceManager::EndTrace();
    func_graph_cache_[args_spec_list] = generated_graph;
    MS_EXCEPTION_IF_NULL(engine);
    engine->func_graph_manager()->AddFuncGraph(generated_graph);
    ret = generated_graph;
  } else {
    ret = iter->second;
  }

  // For the top graph, if it is replaced by generated graph, update the top graph to the new one.
  if (parse::Parser::GetTopFuncGraph() == func_graph()) {
    if (ret != func_graph()) {
      parse::Parser::UpdateTopFuncGraph(ret);
    }
  }
  return ret;
}

FuncGraphPtr MetaFuncGraphEvaluator::GetFuncGraph(AnalysisEnginePtr engine, const AbstractBasePtrList &args_spec_list) {
  auto iter = func_graph_cache_.find(args_spec_list);
  if (iter != func_graph_cache_.end()) {
    return iter->second;
  }

  MS_EXCEPTION_IF_NULL(meta_func_graph_);
  FuncGraphPtr generated_func_graph = nullptr;
  if (this->bound_node() != nullptr) {
    TraceManager::DebugTrace(std::make_shared<TraceGenMetaFuncGraph>(bound_node()->debug_info()));
    generated_func_graph = meta_func_graph_->GenerateFuncGraph(args_spec_list);
    TraceManager::EndTrace();
  } else {
    generated_func_graph = meta_func_graph_->GenerateFuncGraph(args_spec_list);
  }

  FuncGraphPtr cloned_func_graph = BasicClone(generated_func_graph);
  func_graph_cache_[args_spec_list] = cloned_func_graph;
  MS_EXCEPTION_IF_NULL(engine);
  engine->func_graph_manager()->AddFuncGraph(cloned_func_graph);
  return cloned_func_graph;
}

AbstractBasePtr Evaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                               AnfNodeConfigPtr out_conf) {
  const std::string &evaluator_name = ToString();

  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->GetEvaluatedValue();
                       });
  args_spec_list = NormalizeArgs(args_spec_list);
  trace::TraceGraphInferEnter(shared_from_base<Evaluator>(), out_conf);
  InferEntryLogging(shared_from_base<Evaluator>(), args_spec_list, out_conf);
  MS_EXCEPTION_IF_NULL(cache_);
  auto iter = cache_->find(args_spec_list);
  if (iter == cache_->end()) {
    MS_LOG(DEBUG) << evaluator_name << " cache miss, call Infer().";
    AbstractBasePtr ret = Infer(engine, args_spec_list);
    if (ret == nullptr) {
      InferFailLogging(shared_from_base<Evaluator>(), args_spec_list, out_conf);
      MS_LOG(EXCEPTION) << "Evaluator " << evaluator_name << " result is nullptr.";
    }
    MS_EXCEPTION_IF_NULL(ret);
    MS_LOG(DEBUG) << evaluator_name << " set cache. return: " << ret->ToString() << ".";
    (*cache_)[args_spec_list] = ret;
    trace::TraceGraphInferLeave(shared_from_base<Evaluator>());
    return ret;
  } else {
    MS_EXCEPTION_IF_NULL(iter->second);
    MS_LOG(DEBUG) << evaluator_name << " cache hit. return: " << iter->second->ToString() << ".";
    trace::TraceGraphInferLeave(shared_from_base<Evaluator>());
    return iter->second;
  }
}

AbstractBasePtr TrivialPrimEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                          AnfNodeConfigPtr) {
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->GetEvaluatedValue();
                       });
  AbstractBasePtr ret = EvalPrim(engine, args_spec_list);
  return ret;
}

AbstractBasePtr TransitionPrimEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                             AnfNodeConfigPtr out_conf) {
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->GetEvaluatedValue();
                       });
  if (args_conf_list.size() == 0) {
    MS_LOG(EXCEPTION) << "Size should greater than 0";
  }
  AbstractBasePtr ret = EvalPrim(engine, args_spec_list, args_conf_list[0], out_conf);
  // No need to cache.
  return ret;
}

AbstractBasePtr SymbolicPrimEvaluator::Run(AnalysisEnginePtr, const ConfigPtrList &args_conf_list, AnfNodeConfigPtr) {
  AbstractBasePtr ret = EvalPrim(args_conf_list);
  return ret;
}

AbstractBasePtr TrackedEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                      AnfNodeConfigPtr out_conf) {
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->GetEvaluatedValue();
                       });
  AbstractBasePtr ret = sub_evaluator_->Run(engine, args_conf_list, out_conf);
  // Don't lookup from cache, as different out_conf with same node but different context
  // may add different entry to anfnode_config_map_, like getattr primitive.
  (*cache_)[args_spec_list] = ret;
  return ret;
}

AbstractBasePtr PartialAppEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                         AnfNodeConfigPtr out_conf) {
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->GetEvaluatedValue();
                       });
  MS_EXCEPTION_IF_NULL(cache_);
  auto iter = cache_->find(args_spec_list);
  if (iter != cache_->end()) {
    return iter->second;
  }

  ConfigPtrList partial_args_conf_list;
  // Join arguments in partial and the rest arguments from args_conf_list.
  (void)std::transform(args_spec_list_.begin(), args_spec_list_.end(), std::back_inserter(partial_args_conf_list),
                       [](const AbstractBasePtr &arg) -> ConfigPtr { return std::make_shared<VirtualConfig>(arg); });

  (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(partial_args_conf_list),
                       [](const AbstractBasePtr &arg) -> ConfigPtr { return std::make_shared<VirtualConfig>(arg); });
  AbstractBasePtr ret = evaluator_->Run(engine, partial_args_conf_list, out_conf);
  (*cache_)[args_spec_list] = ret;
  return ret;
}

AbstractBasePtr JEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list, AnfNodeConfigPtr) {
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->GetEvaluatedValue();
                       });
  MS_EXCEPTION_IF_NULL(cache_);
  auto iter = cache_->find(args_spec_list);
  if (iter != cache_->end()) {
    return iter->second;
  }

  // Call the original evaluator, get the result: y = f(x)
  AbstractBasePtr result = evaluator_->Run(engine, args_conf_list, nullptr);
  // Build a virtual function: bprop_f which use sense of y as input, return sense of function free variable and input
  // parameters. (sense_f, sense_x, ...)(*bpro_f) (sense_y)
  AbstractBasePtrList bparams;
  bparams.push_back(SensitivityTransform(orig_func_));
  (void)std::transform(
    args_spec_list.begin(), args_spec_list.end(), std::back_inserter(bparams),
    [](const AbstractBasePtr &arg_spec) -> AbstractBasePtr { return SensitivityTransform(arg_spec); });
  AbstractBasePtr bparams_final = std::make_shared<AbstractTuple>(bparams);
  AbstractFunctionPtr bprop = std::make_shared<VirtualAbstractClosure>(SensitivityTransform(result), bparams_final);

  // J(f)(J(x)) return a tuple (y, bprop_f)
  AbstractBasePtrList jargs = {result, bprop};
  AbstractBasePtr jtuple = std::make_shared<AbstractTuple>(jargs);
  (*cache_)[args_spec_list] = jtuple;
  return jtuple;
}

AbstractBasePtr VirtualEvaluator::Infer(AnalysisEnginePtr, const AbstractBasePtrList &args_spec_list) {
  if (args_spec_list.size() != args_spec_list_.size()) {
    MS_LOG(EXCEPTION) << "Arguments mismatch, parameters no: " << args_spec_list_.size()
                      << ", arguments no: " << args_spec_list.size();
  }
  // Check each parameter and argument match;
  for (std::size_t i = 0; i < args_spec_list.size(); i++) {
    MS_EXCEPTION_IF_NULL(args_spec_list[i]);
    (void)args_spec_list[i]->Join(args_spec_list_[i]);
  }
  return output_;
}
}  // namespace abstract
}  // namespace mindspore
