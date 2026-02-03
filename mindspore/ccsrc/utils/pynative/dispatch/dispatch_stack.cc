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

#include "include/utils/pynative/dispatch/dispatch_stack.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace pynative {
namespace {
thread_local bool in_dispatch = false;
}

const std::string &GetDispatchGuardCallbackStr() {
  static const std::string kDispatchGuardCallback = "__ms_dispatch__";
  return kDispatchGuardCallback;
}

DispatchGuardStack &DispatchGuardStack::Get() {
  static DispatchGuardStack instance;
  return instance;
}

void DispatchGuardStack::Push(const py::object &guard) {
  if (guard.is_none()) {
    MS_LOG(WARNING) << "Attempting to push None as dispatch guard, ignored";
    return;
  }
  MS_LOG(DEBUG) << "Push dispatch guard, depth: " << dispatch_guard_stack.size() << " -> "
                << (dispatch_guard_stack.size() + 1);
  dispatch_guard_stack.push_back(guard);
}

void DispatchGuardStack::Pop() {
  if (dispatch_guard_stack.empty()) {
    MS_LOG(EXCEPTION) << "Cannot pop from empty DispatchGuardStack";
  }
  MS_LOG(DEBUG) << "Pop dispatch guard, depth: " << dispatch_guard_stack.size() << " -> "
                << (dispatch_guard_stack.size() - 1);
  dispatch_guard_stack.pop_back();
}

bool DispatchGuardStack::HasDispatchGuard() const { return !dispatch_guard_stack.empty(); }

size_t DispatchGuardStack::Depth() const { return dispatch_guard_stack.size(); }

void DispatchGuardStack::Clear() {
  MS_LOG(DEBUG) << "Clear dispatch guard stack";
  dispatch_guard_stack.clear();
  dispatch_guard_current_index = -1;
  dispatch_guard_chain_started = false;
}

bool DispatchGuardStack::NoMoreGuard() {
  return dispatch_guard_current_index < 0 ||
         static_cast<size_t>(dispatch_guard_current_index) >= dispatch_guard_stack.size();
}

const py::object &DispatchGuardStack::GetCurrentGuardAndMoveNext() {
  const py::object &guard = dispatch_guard_stack[static_cast<size_t>(dispatch_guard_current_index)];
  --dispatch_guard_current_index;
  MS_LOG(DEBUG) << "GetCurrentGuardAndMoveNext: returned guard at index " << (dispatch_guard_current_index + 1)
                << ", next index: " << dispatch_guard_current_index;
  return guard;
}

GuardChainScope::GuardChainScope() {
  auto &stack = DispatchGuardStack::Get();
  stack.StartChain();
  stack.MarkChainStarted();
  MS_LOG(DEBUG) << "GuardChainScope: started chain, index=" << (stack.Depth() - 1);
}

GuardChainScope::~GuardChainScope() {
  auto &stack = DispatchGuardStack::Get();
  stack.EndChain();
  stack.MarkChainEnded();
  MS_LOG(DEBUG) << "GuardChainScope: ended chain";
}

InMsDispatchGuard::InMsDispatchGuard() {
  pre_in_dispatch_ = in_dispatch;
  in_dispatch = true;
}

InMsDispatchGuard::~InMsDispatchGuard() { in_dispatch = pre_in_dispatch_; }

bool IsDispatchEnable() { return in_dispatch; }

bool HasActiveDispatchGuard() {
  auto &stack = DispatchGuardStack::Get();
  // Only trigger guard chain if:
  // 1. There are guards on the stack
  // 2. Chain hasn't been started yet
  if (!stack.HasDispatchGuard()) {
    return false;
  }
  if (stack.ChainStarted()) {
    MS_LOG(DEBUG) << "HasActiveDispatchGuard: chain already started, returning false";
    return false;
  }
  MS_LOG(DEBUG) << "HasActiveDispatchGuard: guards available, returning true";
  return true;
}

}  // namespace pynative
}  // namespace mindspore
