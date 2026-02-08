/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_CORE_OP_UTILS_H
#define MINDSPORE_CORE_UTILS_CORE_OP_UTILS_H

#include <set>
#include <string>
#include "mindapi/base/macros.h"
#include "mindapi/ir/abstract.h"
#include "ir/primitive.h"
#include "ir/anf.h"

#ifndef MS_UNLIKELY
#ifdef _MSC_VER
#define MS_UNLIKELY(x) (x)
#define MS_LIKELY(x) (x)
#else
#define MS_LIKELY(x) __builtin_expect(!!(x), 1)
#define MS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif
#endif
#define MS_CHECK_VALUE(cond, msg)        \
  do {                                   \
    if (MS_UNLIKELY(!(cond))) {          \
      MS_EXCEPTION(ValueError) << (msg); \
    }                                    \
  } while (0)

#define MS_AMBIGUOUS_ELSE_BLOCKER_ \
  switch (0)                       \
  case 0:                          \
  default:
#define MS_ASSERT_TRUE(cond) \
  MS_AMBIGUOUS_ELSE_BLOCKER_ \
  if (cond)                  \
    ;                        \
  else                       \
    MS_LOG(EXCEPTION)

#define MS_ASSERT_FALSE(cond) MS_ASSERT_TRUE(!cond)

namespace mindspore::ops {
MS_CORE_API std::set<int64_t> GetInputDependValueList(const PrimitivePtr &op_prim);

template <typename T>
api::SharedPtr<T> GetOperator(const AnfNodePtr &node) {
  auto prim = GetValueNode<PrimitivePtr>(node);
  if (prim == nullptr) {
    return nullptr;
  }
  return api::MakeShared<T>(prim);
}

MS_CORE_API size_t GetInputIndexByName(const std::string &op_name, const std::string &input_name);
MS_CORE_API std::string GetInputNameByIndex(const std::string &op_name, size_t index);
MS_CORE_API size_t GetOpInputsNum(const std::string &op_name);
MS_CORE_API CNodePtr ConvertArgsToAttr(const CNodePtr &cnode);
MS_CORE_API bool HasOpDef(const std::string &op_name);
}  //  namespace mindspore::ops
#endif  //  MINDSPORE_CORE_UTILS_CORE_OP_UTILS_H
