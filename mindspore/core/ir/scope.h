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

#ifndef IR_SCOPE_H_
#define IR_SCOPE_H_
#include <string>
#include <memory>
#include <stack>
namespace mindspore {
class Scope;
using ScopePtr = std::shared_ptr<Scope>;
extern const ScopePtr kDefaultScope;

class Scope {
 public:
  // using the default scope
  explicit Scope(const std::string &name) : name_(name) {}
  ~Scope() = default;
  std::string name() { return name_; }
  std::string name_;
};

class ScopeManager {
 public:
  static ScopeManager &GetInstance() noexcept {
    static ScopeManager instance;
    return instance;
  }
  ScopeManager(const ScopeManager &) = delete;
  ScopeManager &operator=(const ScopeManager &) = delete;
  ~ScopeManager() = default;
  void EnterScope(const ScopePtr &scope);
  void LeaveScope(const ScopePtr &scope) noexcept;
  ScopePtr GetCurrentScope();
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
#endif  // IR_SCOPE_H_
