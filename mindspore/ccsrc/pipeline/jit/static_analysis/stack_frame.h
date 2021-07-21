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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_STACK_FRAME_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_STACK_FRAME_H_

#include <utility>
#include <memory>
#include <string>
#include <vector>

#include "pipeline/jit/static_analysis/evaluator.h"

namespace mindspore {
namespace abstract {
class StackFrame;
using StackFramePtr = std::shared_ptr<StackFrame>;
using EvaluatorWeakPtr = std::weak_ptr<Evaluator>;
using BaseFuncGraphEvaluatorPtr = std::shared_ptr<BaseFuncGraphEvaluator>;

class StackFrame : public Base {
 public:
  StackFrame(const EvaluatorPtr &evaluator, const FuncGraphPtr &func_graph, const AnalysisContextPtr &current_context,
             const AnalysisContextPtr &parent_context)
      : evaluator_(EvaluatorWeakPtr(evaluator)),
        func_graph_(func_graph),
        current_context_(current_context),
        parent_context_(parent_context),
        slot_index_(0),
        done_(false) {
    Load();
  }
  virtual ~StackFrame() = default;

  void Load() {
    node_slots_ = TopoSort(func_graph_->get_return(), SuccIncoming, [this](const AnfNodePtr &node) -> IncludeType {
      if (node->isa<ValueNode>() || node->isa<Parameter>()) {
        return EXCLUDE;
      }
      return FOLLOW;
    });
    slot_index_ = 0;
    args_abs_list_.clear();
  }

  // Check if we need branch to another func graph.
  StackFramePtr Jump(const AnalysisEnginePtr &engine);
  // Run one step in current func graph.
  EvalResultPtr Step(const AnalysisEnginePtr &engine);
  // Return back from branch func graph.
  void Back(const AnalysisEnginePtr &engine, const StackFramePtr &last_stack_frame, const EvalResultPtr &eval_result);

  bool Done() { return done_; }

  AnfNodePtr &CurrentNode() {
    if (slot_index_ >= node_slots_.size()) {
      MS_LOG(EXCEPTION) << "The stack frame of " << func_graph_->ToString()
                        << " is invalid. Try to access frame sequence by index " << slot_index_
                        << ", while the size is " << node_slots_.size() << ".";
    }
    return node_slots_[slot_index_];
  }

  AnfNodePtr &NextNode() {
    auto &current_node = CurrentNode();
    // Set `done_` true, if the stack frames is being exhausted.
    if (current_node == func_graph_->get_return()) {
      done_ = true;
    }
    // Move cursor to next node.
    slot_index_++;
    return current_node;
  }

  EvaluatorPtr evaluator() const { return evaluator_.lock(); }
  FuncGraphPtr func_graph() const { return func_graph_; }
  AnalysisContextPtr current_context() const { return current_context_; }
  AnalysisContextPtr parent_context() const { return parent_context_; }

  const AbstractBasePtrList &args_abs_list() { return args_abs_list_; }
  void set_args_abs_list(const AbstractBasePtrList &&args_abs_list) { args_abs_list_ = args_abs_list; }

  std::string ToString() const override {
    MS_EXCEPTION_IF_NULL(func_graph_);
    std::ostringstream buffer;
    buffer << "StackFrame: " << this << ", " << func_graph_->ToString();
    if (slot_index_ < node_slots_.size()) {
      auto current_node = node_slots_[slot_index_];
      buffer << "(#" << slot_index_ << " / Running " << current_node->DebugString() << ")";
    } else {
      buffer << "(Exhausted..)";
    }
    buffer << ", parent: ";
    auto parent_graph = parent_context_->func_graph();
    if (parent_graph != nullptr) {
      buffer << parent_graph << "/" << parent_graph->ToString();
    } else {
      buffer << "NULL";
    }
    return buffer.str();
  }

  friend std::ostream &operator<<(std::ostream &os, const StackFramePtr &frame) {
    MS_EXCEPTION_IF_NULL(frame);
    os << frame->ToString();
    return os;
  }

 private:
  AbstractBasePtrList GenerateArgsAbsList(const AnalysisEnginePtr &engine, const EvaluatorPtr &evaluator,
                                          const CNodePtr current_cnode);
  AnalysisContextPtr GetParentContext(const BaseFuncGraphEvaluatorPtr &fg_evaluator,
                                      const AbstractFunctionPtr &graph_func);
  StackFramePtr DoJump(const AnalysisEnginePtr &engine, const CNodePtr current_cnode,
                       const AbstractFunctionPtr &graph_func);

  EvaluatorWeakPtr evaluator_;
  FuncGraphPtr func_graph_;
  AnalysisContextPtr current_context_;
  AnalysisContextPtr parent_context_;
  AbstractBasePtrList args_abs_list_;
  std::vector<AnfNodePtr> node_slots_;
  size_t slot_index_;
  bool done_;
};
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_STACK_FRAME_H_
