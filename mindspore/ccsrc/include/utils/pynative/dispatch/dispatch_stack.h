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

#ifndef MINDSPORE_CCSRC_INCLUDE_UTILS_PYNATIVE_DISPATCH_DISPATCH_STACK_H_
#define MINDSPORE_CCSRC_INCLUDE_UTILS_PYNATIVE_DISPATCH_DISPATCH_STACK_H_

#include <string>
#include <vector>
#include "pybind11/pybind11.h"
#include "include/utils/visible.h"

namespace py = pybind11;
namespace mindspore {
namespace pynative {

// thread_local state for dispatch guard chain, declared at namespace scope
// to avoid MSVC C2492 (thread_local in dllexport class).
// Using inline to allow header-only definition without ODR violations (C++17).
inline thread_local std::vector<py::object> dispatch_guard_stack;
inline thread_local int64_t dispatch_guard_current_index{-1};
inline thread_local bool dispatch_guard_chain_started{false};

/// \brief Stack-based manager for nested dispatch guards.
///
/// Design:
/// - Guards are stored in a stack (bottom to top order)
/// - dispatch_guard_current_index tracks which guard to execute next
/// - When dispatch_guard_current_index >= 0, there are guards to process
/// - When dispatch_guard_current_index < 0, guard chain is complete
///
/// Example:
///   with GuardA():     # stack: [A], index not set
///       with GuardB(): # stack: [A, B], index not set
///           op()       # StartChain() sets index=1, then B->A->op
class COMMON_EXPORT DispatchGuardStack {
 public:
  static DispatchGuardStack &Get();

  void Push(const py::object &guard);
  void Pop();
  bool HasDispatchGuard() const;
  size_t Depth() const;
  void Clear();

  /// \brief Start guard chain dispatch. Sets current_index to top of stack.
  void StartChain() { dispatch_guard_current_index = static_cast<int64_t>(dispatch_guard_stack.size()) - 1; }

  /// \brief End guard chain dispatch. Resets current_index to -1.
  void EndChain() { dispatch_guard_current_index = -1; }

  /// \brief Check if currently in guard chain (index >= 0 or was started).
  bool InChain() const { return dispatch_guard_current_index >= -1 && dispatch_guard_chain_started; }

  /// \brief Check if there are more guards to process.
  bool HasMoreGuards() const { return dispatch_guard_current_index >= 0; }

  /// \brief Get current guard and move to next.
  /// \return Current guard, or py::none() if no more guards.
  const py::object &GetCurrentGuardAndMoveNext();

  bool NoMoreGuard();

  void MarkChainStarted() { dispatch_guard_chain_started = true; }

  void MarkChainEnded() { dispatch_guard_chain_started = false; }

  bool ChainStarted() const { return dispatch_guard_chain_started; }

 private:
  DispatchGuardStack() = default;
  ~DispatchGuardStack() = default;
  DispatchGuardStack(const DispatchGuardStack &) = delete;
  DispatchGuardStack &operator=(const DispatchGuardStack &) = delete;
};

/// \brief RAII guard for managing guard chain lifecycle.
class COMMON_EXPORT GuardChainScope {
 public:
  GuardChainScope();
  ~GuardChainScope();
  GuardChainScope(const GuardChainScope &) = delete;
  GuardChainScope &operator=(const GuardChainScope &) = delete;
};

class COMMON_EXPORT InMsDispatchGuard {
 public:
  InMsDispatchGuard();
  ~InMsDispatchGuard();

  InMsDispatchGuard(const InMsDispatchGuard &) = delete;
  InMsDispatchGuard &operator=(const InMsDispatchGuard &) = delete;
  InMsDispatchGuard(InMsDispatchGuard &&) = delete;
  InMsDispatchGuard &operator=(InMsDispatchGuard &&) = delete;

 private:
  bool pre_in_dispatch_{false};  // Track previous state for restoration
};

COMMON_EXPORT bool IsDispatchEnable();

/// \brief Check if dispatch guard chain should be started.
/// Returns true only when guards exist and chain hasn't started yet.
COMMON_EXPORT bool HasActiveDispatchGuard();

/// \brief Get the attribute name for dispatch guard callback.
COMMON_EXPORT const std::string &GetDispatchGuardCallbackStr();

}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_UTILS_PYNATIVE_DISPATCH_DISPATCH_STACK_H_
