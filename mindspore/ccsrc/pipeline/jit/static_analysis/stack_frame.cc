/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/static_analysis/stack_frame.h"
#include "debug/trace.h"

namespace mindspore {
namespace abstract {
AbstractBasePtrList StackFrame::GenerateArgsAbsList(const AnalysisEnginePtr &engine, const EvaluatorPtr &evaluator,
                                                    const CNodePtr current_cnode) {
  AbstractBasePtrList args_abs_list;
  auto &inputs = current_cnode->inputs();
  for (std::size_t i = 1; i < inputs.size(); i++) {
    auto config = engine->MakeConfig(inputs[i], current_context_);
    auto abs = config->ObtainEvalResult()->abstract();
    args_abs_list.push_back(abs);
  }
  args_abs_list = evaluator->NormalizeArgs(args_abs_list);
  args_abs_list = evaluator->BroadenUndeterminedArgs(args_abs_list);
  return args_abs_list;
}

StackFramePtr StackFrame::DoJump(const AnalysisEnginePtr &engine, const CNodePtr current_cnode,
                                 const FuncGraphAbstractClosurePtr &graph_func) {
  // Get the evaluator for func graph.
  auto evaluator = engine->GetEvaluatorFor(graph_func);
  auto fg_evaluator = dyn_cast<BaseFuncGraphEvaluator>(evaluator);
  if (fg_evaluator == nullptr) {
    MS_LOG(EXCEPTION) << "Evaluator should be a BaseGraphEvaluator, but got " << evaluator->ToString();
  }
  fg_evaluator->set_context(current_context_);

  // Evaluate the inputs firstly. Build arguments for the func graph.
  AbstractBasePtrList args_abs_list = GenerateArgsAbsList(engine, evaluator, current_cnode);

  // Generate func graph with arguments.
  auto fg = fg_evaluator->GetFuncGraph(engine, args_abs_list);
  MS_EXCEPTION_IF_NULL(fg);
  std::size_t nargs = fg->parameters().size();
  if (args_abs_list.size() != nargs) {
    MS_EXCEPTION(TypeError) << "Function " << fg->ToString() << ", The number of parameters of this function is "
                            << fg->parameters().size() << ", but the number of provided arguments is "
                            << args_abs_list.size() << ". NodeInfo: " << trace::GetDebugInfo(fg->debug_info());
  }

  MS_LOG(DEBUG) << "current_node: " << current_cnode->DebugString() << ", fg: " << fg->ToString()
                << ", current_context_: " << current_context_->ToString();

  // Find parent context and create new context.
  auto branch_fg = graph_func->func_graph();
  auto parent_context = graph_func->context()->FindParentContext(branch_fg);
  auto new_context = parent_context->NewFuncGraphContext(fg, args_abs_list);
  // Evaluate the parameters with new context.
  for (size_t i = 0; i < nargs; i++) {
    const auto &arg_abs = args_abs_list[i];
    const auto &node = fg->parameters()[i];
    AnfNodeConfigPtr conf = engine->MakeConfig(node, new_context);
    engine->SaveEvalResultInCache(conf, std::make_shared<EvalResult>(arg_abs, nullptr));
  }

  // Create a new stack frame and set arguments for it.
  auto new_stack_frame = std::make_shared<StackFrame>(fg_evaluator, fg, new_context, current_context());
  new_stack_frame->set_args_abs_list(std::move(args_abs_list));
  return new_stack_frame;
}

// Check if we need branch to another func graph.
StackFramePtr StackFrame::Jump(const AnalysisEnginePtr &engine) {
  auto &current_node = CurrentNode();
  if (!current_node->isa<CNode>()) {
    return nullptr;
  }
  auto cnode = current_node->cast<CNodePtr>();
  auto func = engine->GetCNodeOperatorAbstract(cnode, current_context_);
  auto graph_func = dyn_cast<FuncGraphAbstractClosure>(func);  // Not handle MetaFuncGraphAbstractClosure by now.
  if (graph_func == nullptr) {
    return nullptr;  // Not call FuncGraph.
  }

  // It's FuncGraph Call.
  return DoJump(engine, cnode, graph_func);
}

EvalResultPtr StackFrame::Step(const AnalysisEnginePtr &engine) {
  auto &current_node = NextNode();
  MS_LOG(DEBUG) << "current_node: " << current_node->DebugString()
                << ", current_context_: " << current_context_->ToString();
  AnfNodeConfigPtr node_conf = engine->MakeConfig(current_node, current_context_);
  auto node_eval_result = engine->ObtainEvalResultWithCache(node_conf);
  return node_eval_result;
}

void StackFrame::Back(const AnalysisEnginePtr &engine, const EvalResultPtr &result) {
  auto &current_node = NextNode();
  MS_LOG(DEBUG) << "current_node: " << current_node->DebugString()
                << ", current_context_: " << current_context_->ToString();
  AnfNodeConfigPtr node_conf = engine->MakeConfig(current_node, current_context_);
  engine->SaveEvalResultInCache(node_conf, result);
}
}  // namespace abstract
}  // namespace mindspore
