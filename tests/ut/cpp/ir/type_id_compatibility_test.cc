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

#include "mindapi/base/type_id.h"

namespace mindspore {
static_assert(kNumberTypeDouble == 49, "Existing number type IDs must remain stable");
static_assert(kNumberTypeComplex == 50, "Existing number type IDs must remain stable");
static_assert(kNumberTypeComplex64 == 51, "Existing number type IDs must remain stable");
static_assert(kNumberTypeComplex128 == 52, "Existing number type IDs must remain stable");
static_assert(kNumberTypeInt4 == 53, "Existing number type IDs must remain stable");
static_assert(kNumberTypeGLUInt == 54, "Existing number type IDs must remain stable");
static_assert(kNumberTypeEnd == 55, "Existing number type boundary must remain stable");
static_assert(kMonadTypeBegin == 55, "Existing monad type boundary must remain stable");
static_assert(kObjectTypeMonad == 56, "Existing monad type IDs must remain stable");
static_assert(kObjectTypeUMonad == 57, "Existing monad type IDs must remain stable");
static_assert(kObjectTypeIOMonad == 58, "Existing monad type IDs must remain stable");
static_assert(kMonadTypeEnd == 59, "Existing monad type boundary must remain stable");
static_assert(kSparseTypeBegin == 59, "Existing sparse type boundary must remain stable");
static_assert(kObjectTypeCSRTensorType == 60, "Existing sparse type IDs must remain stable");
static_assert(kObjectTypeSparseTensorType == 61, "Existing sparse type IDs must remain stable");
static_assert(kObjectTypeMapTensorType == 62, "Existing sparse type IDs must remain stable");
static_assert(kSparseTypeEnd == 63, "Existing sparse type boundary must remain stable");
static_assert(kEventTypeBegin == 63, "Existing event type boundary must remain stable");
static_assert(kObjectTypeEventType == 64, "Existing event type IDs must remain stable");
static_assert(kEventTypeEnd == 65, "Existing event type boundary must remain stable");
static_assert(kNumberTypeCompatBegin == kEventTypeEnd, "Compatible number types must follow existing types");
static_assert(kNumberTypeFloat8E8M0FNU == 66, "New number types must follow existing types");
static_assert(IsNumberTypeId(kNumberTypeFloat8E8M0FNU), "E8M0 must be recognized as a number type");
static_assert(IsNumberTypeId(kNumberTypeFloat32), "Existing number types must remain recognized");
static_assert(!IsNumberTypeId(kNumberTypeCompatBegin), "Compatible number type begin is not a number type");
static_assert(!IsNumberTypeId(kNumberTypeCompatEnd), "Compatible number type end is not a number type");
static_assert(!IsNumberTypeId(kObjectTypeEventType), "Event types must not be recognized as numbers");
static_assert(!IsNumberTypeId(kObjectTypeUMonad), "Monad types must not be recognized as numbers");
}  // namespace mindspore
