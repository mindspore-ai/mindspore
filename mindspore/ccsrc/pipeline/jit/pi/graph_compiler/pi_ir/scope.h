/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_PI_JIT_SCOPE_H_
#define MINDSPORE_PI_JIT_SCOPE_H_

namespace mindspore {
namespace pijit {
namespace ir {
enum Scope {
  kScopeUnknown,
  kScopeConst,    // means this value will be placed in the tuple co_const.
  kScopeLocal,    // means this value will be placed in the tuple co_varnames.
  kScopeBuiltIn,  // means this value will be placed in the dict builtins_.
  kScopeGlobal,   // means this value will be placed in the dict globals_.
  kScopeName,     // means this value will be placed in the tuple co_names.
  kScopeFreeVar,  // means this value will be placed in the tuple co_freevars.
  kScopeCellVar,  // means this value will be placed in the tuple co_cellvars.
  kScopeClousre   // means this value will be placed in the tuple clousre_.
};
}  // namespace ir
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_SCOPE_H_
