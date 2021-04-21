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

#include "ir/scope.h"
namespace mindspore {
const ScopePtr kDefaultScope = std::make_shared<Scope>("Default");

void ScopeManager::EnterScope(const ScopePtr &scope) {
  if (scope != kDefaultScope) {
    scope_stack_.push(scope);
  }
}

void ScopeManager::LeaveScope(const ScopePtr &scope) noexcept {
  if (scope != kDefaultScope && !scope_stack_.empty()) {
    scope_stack_.pop();
  }
}
ScopePtr ScopeManager::GetCurrentScope() {
  // if the scope stack is empty, return the default scope
  if (scope_stack_.empty()) {
    return kDefaultScope;
  }
  return scope_stack_.top();
}
void ScopeManager::ClearScope() {
  while (!scope_stack_.empty()) {
    scope_stack_.pop();
  }
}
}  // namespace mindspore
