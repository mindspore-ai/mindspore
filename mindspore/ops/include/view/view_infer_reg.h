/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_OPS_INCLUDE_VIEW_VIEW_INFER_REG_H_
#define MINDSPORE_OPS_INCLUDE_VIEW_VIEW_INFER_REG_H_

#include <string_view>
#include "mindapi/base/macros.h"

namespace mindspore::ops {
constexpr size_t kCompileHashPrime = 31;
constexpr size_t CompileHashString(std::string_view str) noexcept {
  size_t h = 0;
  for (char c : str) {
    h = (h * kCompileHashPrime) + static_cast<unsigned char>(c);
  }
  return h;
}

template <size_t Key>
struct ViewFunctionHolder {
  static_assert(Key != Key, "View infer function not registered!");
};

#define REG_VIEW_INFER_FUNCTION(name, infer_func)       \
  template <>                                           \
  struct ViewFunctionHolder<CompileHashString(#name)> { \
    static constexpr auto func = infer_func;            \
  }

#define CALL_VIEW_INFER_FUNCTION(name, ...) ops::ViewFunctionHolder<ops::CompileHashString(#name)>::func(__VA_ARGS__)
}  // namespace mindspore::ops
#endif  // MINDSPORE_OPS_INCLUDE_VIEW_VIEW_INFER_REG_H_
