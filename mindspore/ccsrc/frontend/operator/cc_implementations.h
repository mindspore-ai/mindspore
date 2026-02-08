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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_CC_IMPLEMENTATIONS_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_CC_IMPLEMENTATIONS_H_

#include <vector>
#include <memory>
#include "ir/anf.h"
#include "ir/value.h"
#include "utils/any.h"
#include "include/utils/visible.h"

namespace mindspore {
// namespace to support primitive operators definition
namespace prim {
using Any = mindspore::Any;
using AnyPtrList = std::vector<std::shared_ptr<Any>>;
using ValuePtrList = std::vector<ValuePtr>;
using OpsFunction = std::function<Any(const AnyPtrList &)>;
using AnfNodeOpsFunction = std::function<AnfNodePtr(const std::vector<AnfNodePtr> &)>;

ValuePtr ScalarAdd(const ValuePtrList &list);
ValuePtr ScalarSub(const ValuePtrList &list);
ValuePtr ScalarMul(const ValuePtrList &list);
ValuePtr ScalarDiv(const ValuePtrList &list);
ValuePtr ScalarMod(const ValuePtrList &list);
ValuePtr ScalarPow(const ValuePtrList &list);
ValuePtr ScalarFloorDiv(const ValuePtrList &list);
ValuePtr ScalarUAdd(const ValuePtrList &list);
ValuePtr ScalarUSub(const ValuePtrList &list);
ValuePtr ScalarLog(const ValuePtrList &list);
FRONTEND_EXPORT ValuePtr ScalarEq(const ValuePtrList &list);
ValuePtr ScalarLt(const ValuePtrList &list);
ValuePtr ScalarGt(const ValuePtrList &list);
ValuePtr ScalarNe(const ValuePtrList &list);
ValuePtr ScalarLe(const ValuePtrList &list);
ValuePtr ScalarGe(const ValuePtrList &list);
ValuePtr BoolNot(const ValuePtrList &list);
ValuePtr BoolAnd(const ValuePtrList &list);
ValuePtr BoolOr(const ValuePtrList &list);
ValuePtr BoolEq(const ValuePtrList &list);
ValuePtr BitAnd(const ValuePtrList &list);
ValuePtr BitOr(const ValuePtrList &list);
ValuePtr BitXor(const ValuePtrList &list);
ValuePtr BitLeftShift(const ValuePtrList &list);
ValuePtr BitRightShift(const ValuePtrList &list);
ValuePtr StringEq(const ValuePtrList &list);
ValuePtr StringLt(const ValuePtrList &list);
ValuePtr StringGt(const ValuePtrList &list);
ValuePtr StringLe(const ValuePtrList &list);
ValuePtr StringGe(const ValuePtrList &list);
ValuePtr StringConcat(const ValuePtrList &list);
ValuePtr StringNot(const ValuePtrList &list);
ValuePtr StringIn(const ValuePtrList &list);
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_CC_IMPLEMENTATIONS_H_
