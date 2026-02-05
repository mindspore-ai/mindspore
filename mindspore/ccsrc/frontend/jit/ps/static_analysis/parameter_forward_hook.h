/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PS_STATIC_ANALYSIS_PARAMETER_FORWARD_HOOK_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_PS_STATIC_ANALYSIS_PARAMETER_FORWARD_HOOK_H_

#include <utility>
#include <stack>

#include "ir/anf.h"
#include "base/base.h"
#include "frontend/jit/ps/resource.h"
#include "frontend/jit/ps/static_analysis/stack_frame.h"
#include "include/frontend/jit/ps/resource_interface.h"

namespace mindspore {
namespace abstract {
constexpr auto kParameterForwardHook = "parameter_forward_hook";

class ParameterForwardHookStack {
 public:
  static ParameterForwardHookStack &Instance() {
    static ParameterForwardHookStack instance;
    return instance;
  }

  ParameterForwardHookStack(const ParameterForwardHookStack &) = delete;
  ParameterForwardHookStack &operator=(const ParameterForwardHookStack &) = delete;

  void push(const StackFramePtr &frame, bool active) { stk_.push({frame, active}); }
  void push(StackFramePtr &&frame, bool active) { stk_.push({std::move(frame), active}); }
  void pop() { stk_.pop(); }
  const std::pair<StackFramePtr, bool> &top() const { return stk_.top(); }
  bool empty() const { return stk_.empty(); }
  size_t size() const { return stk_.size(); }
  void clear() { stk_ = std::stack<std::pair<StackFramePtr, bool>>(); }

 private:
  ParameterForwardHookStack() = default;

  std::stack<std::pair<StackFramePtr, bool>> stk_;
};

void ClearParameterForwardHookStack();
bool IsHookedParameterCNode(const AnfNodePtr &anf_node);
AnfNodePtr ApplyParameterForwardHook(const FuncGraphPtr &func_graph, const ParameterPtr &param_node);
bool ShouldSkipParameterForwardHook(const AnfNodePtr &origin_node, const ParameterPtr &para_node);
void ReOrderParameterForwardHookNodes(const FuncGraphPtr &new_fg, const pipeline::ResourcePtr &resource);
}  // namespace abstract
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PS_STATIC_ANALYSIS_PARAMETER_FORWARD_HOOK_H_
