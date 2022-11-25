/**
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

#ifndef MINDSPORE_CORE_IR_SCOPE_H_
#define MINDSPORE_CORE_IR_SCOPE_H_
#include <string>
#include <memory>
#include <stack>
#include "mindapi/base/macros.h"

namespace mindspore {
class Scope;
using ScopePtr = std::shared_ptr<Scope>;
MS_CORE_API extern const ScopePtr kDefaultScope;

class MS_CORE_API Scope {
 public:
  /// \brief Constructor of Scope.
  ///
  /// \param[in] name The name of scope, usually using the default scope.
  explicit Scope(const std::string &name) : name_(name) {}

  /// \brief Destructor of Scope.
  ~Scope() = default;

  /// \brief Get the name of scope.
  ///
  /// \return The name of scope.
  std::string name() const { return name_; }

 private:
  std::string name_;
};

class MS_CORE_API ScopeManager {
 public:
  /// \brief Get instance of ScopeManager.
  ///
  /// \return Instance of ScopeManager.
  static ScopeManager &GetInstance() noexcept;

  /// \brief Disable the default constructor.
  ScopeManager(const ScopeManager &) = delete;
  /// \brief Disable the default copy constructor.
  ScopeManager &operator=(const ScopeManager &) = delete;
  /// \brief Destructor.
  ~ScopeManager() = default;

  /// \brief Enter the scope.
  ///
  /// \param[in] scope The scope.
  void EnterScope(const ScopePtr &scope);

  /// \brief Leave the scope.
  ///
  /// \param[in] scope The scope.
  void LeaveScope(const ScopePtr &scope) noexcept;

  /// \brief Get the current scope.
  ///
  /// \return The current scope.
  ScopePtr GetCurrentScope();

  /// \brief Clear the scope.
  void ClearScope();

 private:
  ScopeManager() = default;
  std::stack<ScopePtr> scope_stack_;
};

// ScopeGuard is a class that help generate the anf node of specified scope
// in the current c++ action scope.
class ScopeGuard {
 public:
  explicit ScopeGuard(const ScopePtr &scope) {
    scope_ = scope;
    ScopeManager::GetInstance().EnterScope(scope);
  }
  ~ScopeGuard() { ScopeManager::GetInstance().LeaveScope(scope_); }

 private:
  ScopePtr scope_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_SCOPE_H_
